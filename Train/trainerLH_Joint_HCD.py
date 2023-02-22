import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np

from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import scipy.sparse as sp
from utils.dataLoader_XR import LandmarksDataset, ToTensor, RandomScale, AugColor, Rotate
from utils.utils import genMatrixesLH, scipy_to_torch_sparse

from models.joint import Joint

from models.chebConv import Pool
import datetime
import time

def sample_sdf(y_pred, sdf_map):

    # x and y
    h = y_pred[:, :, 1]
    w = y_pred[:, :, 0]

    # map from [0, 1] to [-1, 1]
    h_mapped = 2 * h - 1
    w_mapped = 2 * w - 1

    # clamp values to [-1, 1]
    h_mapped = torch.clamp(h_mapped, min=-1, max=1)
    w_mapped = torch.clamp(w_mapped, min=-1, max=1)

    sdf_map = sdf_map.unsqueeze(1)
    # sample sdf values at prediction coordinates
    sdf_score = F.grid_sample(sdf_map, torch.stack([w_mapped, h_mapped], dim=-1).unsqueeze(1), align_corners=True)

    return sdf_score

def HCD_loss_(beta, downsampled):
    def HCD_loss(y_pred, y_true, sdf_map):

        # calculate mse
        mse_score = F.mse_loss(y_pred, y_true)

        if downsampled:
            sdf_score_lungs = sample_sdf(y_pred[:, :47, :], sdf_map[:, 0, :, :])
            sdf_score_heart = sample_sdf(y_pred[:, 47:, :], sdf_map[:, 1, :, :])
        else:
            sdf_score_lungs = sample_sdf(y_pred[:, :94, :], sdf_map[:, 0, :, :])
            sdf_score_heart = sample_sdf(y_pred[:, 94:, :], sdf_map[:, 1, :, :])

        sdf_score = torch.cat([sdf_score_lungs, sdf_score_heart], -1)
        sdf_score = sdf_score[:,0,0,:]
        sdf_score = torch.mean(sdf_score)

        loss = mse_score + (beta * sdf_score)

        return loss
    return HCD_loss

def trainer(train_dataset, val_dataset, model, config):
    torch.manual_seed(420)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total Number of Trainable Params: ', pytorch_total_params)

    print('Number of Train Examples: ', train_dataset.__len__())
    print('Number of Val Examples: ', val_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True, num_workers = 8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = config['val_batch_size'], num_workers = 8)

    optimizer = torch.optim.Adam(params = model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])

    HCD_loss_downsampled = HCD_loss_(config['beta'], downsampled=True)
    HCD_loss = HCD_loss_(config['beta'], downsampled=False)

    train_loss_avg = []
    train_kld_loss_avg = []
    train_rec_loss_avg = []
    val_loss_avg = []

    folder = os.path.join(config['dir']+"Results", config['name'])
    try:
        os.mkdir(folder)
    except:
        pass

    bestMSE = 1e12
    
    print('Training ...')
        
    scheduler = StepLR(optimizer, step_size=config['stepsize'], gamma=config['gamma'])
    pool = Pool()
    
    for epoch in range(config['epochs']):

        start = time.time()

        model.train()

        train_loss_avg.append(0)
        train_rec_loss_avg.append(0)
        train_kld_loss_avg.append(0)
        num_batches = 0
        
        for sample_batched in train_loader:
            image, sdf, target = sample_batched['image'].to(device), sample_batched['sdf'].to(device), sample_batched['landmarks'].to(device)
            out = model(image)
            target_down = pool(target, model.downsample_matrices[0])
            
            optimizer.zero_grad()

            pos7, pos6, pos5, pos4, pos3, pos2, pos1 = out

            pos1loss = HCD_loss_downsampled(pos1, target_down, sdf) / 2
            pos2loss = HCD_loss_downsampled(pos2, target_down, sdf) / 2
            pos3loss = HCD_loss_downsampled(pos3, target_down, sdf) / 2
            pos4loss = HCD_loss(pos4, target, sdf) / 2
            pos5loss = HCD_loss(pos5, target, sdf) / 2
            pos6loss = HCD_loss(pos6, target, sdf) / 2
            outloss = HCD_loss(pos7, target, sdf) / 2

            loss = outloss + pos6loss + pos5loss + pos4loss + pos3loss + pos2loss + pos1loss

            train_rec_loss_avg[-1] += outloss.item()
            
            kld_loss = -0.5 * torch.mean(torch.mean(1 + model.log_var - model.mu ** 2 - model.log_var.exp(), dim=1), dim=0)
            loss += model.kld_weight * kld_loss

            train_loss_avg[-1] += loss.item()

            loss.backward()

            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()

            train_kld_loss_avg[-1] += model.kld_weight * kld_loss.item()

            num_batches += 1

        train_loss_avg[-1] /= num_batches
        train_rec_loss_avg[-1] /= num_batches
        train_kld_loss_avg[-1] /= num_batches

        print('MSE:KLD ratio {:.5f} : {:.5f} '.format(train_rec_loss_avg[-1],train_kld_loss_avg[-1]))

        print('Epoch [%d / %d] train average reconstruction error: %f' % (epoch+1, config['epochs'], train_rec_loss_avg[-1]*512*512))

        num_batches = 0

        model.eval()
        val_loss_avg.append(0)

        with torch.no_grad():
            for sample_batched in val_loader:
                image, sdf, target = sample_batched['image'].to(device), sample_batched['sdf'].to(device), sample_batched['landmarks'].to(device)

                out = model(image)
                if len(out) > 1:
                    out = out[0]

                loss_rec = HCD_loss(out, target, sdf)

                val_loss_avg[-1] += loss_rec.item()
                num_batches += 1   
                loss_rec = 0

        val_loss_avg[-1] /= num_batches
        
        print('Epoch [%d / %d] validation average reconstruction error: %f' % (epoch+1, config['epochs'], val_loss_avg[-1] * 512 * 512))

        end = time.time()
        epoch_time = end - start
        print('Epoch time: {:.2f}s'.format(epoch_time))

        if val_loss_avg[-1] < bestMSE:
            bestMSE = val_loss_avg[-1]
            print('Model Saved MSE all time')
            torch.save(model.state_dict(), os.path.join(folder, "bestMSE.pt"))

        scheduler.step()
    
    torch.save(model.state_dict(), os.path.join(folder, "final.pt"))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type=str, default=datetime.datetime.now().strftime('%m_%d_%H_%M_%S')+'_JOINT_HCD')
    parser.add_argument("--load", help="enter the folder where the weights are saved", default = "None", type=str)
    parser.add_argument("--inputsize", default = 1024, type=int)
    parser.add_argument("--epochs", default = 2500, type = int)
    parser.add_argument("--lr", default = 1e-4, type = float)
    parser.add_argument("--stepsize", default = 50, type = int)
    parser.add_argument("--gamma", default = 0.9, type = float)
    parser.add_argument("--beta", default=0.025, type=float)
    parser.add_argument("--n_classes", default=3, type=int)

    parser.add_argument('--rtn_all', dest='rtn_all', action='store_true')
    parser.set_defaults(rtn_all=True)
    
    # Number of filters at low resolution
    parser.add_argument("--f", default = 32, type=int)
    # K-hops parameter
    parser.add_argument("--K", default = 6, type=int)
            
    # Arguments for skip connections
    parser.add_argument("--layer", default = 6, type = int)
    parser.add_argument("--w", default = 3, type = int)

    parser.add_argument('--dir', type=str, default='')
    parser.add_argument('--unet_weights', type=str, default='../weights/unet_weights/bestDice_JSRT_Padchest.pt')
    
    config = parser.parse_args()
    config = vars(config)

    print('Name: ',config['name'])
    print('Loading JSRT Padchest Dataset.. ')
    print('Beta: ', config['beta'])

    train_path = config['dir'] + "Datasets/JSRT_Padchest/Train"
    val_path = config['dir'] + "Datasets/JSRT_Padchest/Val"
    transforms_train = [RandomScale(), Rotate(3), AugColor(0.40), ToTensor()]
    transforms_val = [ToTensor()]
    A, AD, D, U, E, ED = genMatrixesLH()

    W = config['w']
    config['window'] = (W, W)

    img_path = os.path.join(train_path, 'Images')
    label_path = os.path.join(train_path, 'Landmarks')
    mask_path = os.path.join(train_path, 'Masks')
    sdf_path = os.path.join(train_path, 'SDF')

    train_dataset = LandmarksDataset(img_path=img_path,
                                     mask_path=mask_path,
                                     sdf_path=sdf_path,
                                     label_path=label_path,
                                     transform=transforms.Compose(transforms_train)
                                     )

    img_path = os.path.join(val_path, 'Images')
    label_path = os.path.join(val_path, 'Landmarks')
    mask_path = os.path.join(val_path, 'Masks')
    sdf_path = os.path.join(val_path, 'SDF')

    val_dataset = LandmarksDataset(img_path=img_path,
                                   mask_path=mask_path,
                                   sdf_path=sdf_path,
                                   label_path=label_path,
                                   transform=transforms.Compose(transforms_val)
                                   )

    N1 = A.shape[0]
    N2 = AD.shape[0]
       
    A = sp.csc_matrix(A).tocoo()
    AD = sp.csc_matrix(AD).tocoo()
    D = sp.csc_matrix(D).tocoo()
    U = sp.csc_matrix(U).tocoo()
        
    D_ = [D.copy()]
    U_ = [U.copy()]

    config['n_nodes'] = [N1, N1, N1, N2, N2, N2]
    A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]

    A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to('cuda:0') for x in X] for X in (A_, D_, U_))
    
    config['latents'] = 16
    config['batch_size'] = 4
    config['val_batch_size'] = 1
    config['weight_decay'] = 1e-5
    
    f = int(config['f'])
    print(f, 'filters')
    config['filters'] = [2, f, f, f, f//2, f//2, f//2]

    print('Model: Joint + HCD Loss')
    model = Joint(config, D_t, U_t, A_t)

    trainer(train_dataset, val_dataset, model, config)