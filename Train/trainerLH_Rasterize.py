import sys
sys.path.append('..')
import os
import torch
import torch.nn.functional as F
import argparse

from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

import scipy.sparse as sp


from sklearn.metrics import mean_squared_error
from utils.utils import scipy_to_torch_sparse, genMatrixesLH
from utils.dataLoader_XR import LandmarksDataset, ToTensor, RandomScale, AugColor, Rotate

from models.rasterize_finetune import Hybrid_Rasterize

from models.unet import DiceLoss
from torch.nn import CrossEntropyLoss

from models.chebConv import Pool
from skimage.metrics import hausdorff_distance as hd
import datetime
import time

def load_weights(model, weights):
    model.load_state_dict(torch.load(weights))
    return model

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

    dice_loss = DiceLoss().to(device)
    ce_loss = CrossEntropyLoss().to(device)

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
            image, target, mask = sample_batched['image'].to(device), sample_batched['landmarks'].to(device), sample_batched['mask'].to(device)
            out = model(image)
            
            optimizer.zero_grad()

            out, pre1, pre2 = out

            seg_1_loss = dice_loss(pre2, mask) + ce_loss(pre2, mask)
            seg_2_loss = dice_loss(pre1, mask) + ce_loss(pre1, mask)
            seg_3_loss = dice_loss(out, mask) + ce_loss(out, mask)

            loss = seg_1_loss + seg_2_loss + seg_3_loss

            train_rec_loss_avg[-1] += seg_3_loss.item()
            
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

        print('REC:KLD ratio {:.5f} : {:.5f} '.format(train_rec_loss_avg[-1],train_kld_loss_avg[-1]))

        print('Epoch [%d / %d] train average reconstruction error: %f' % (epoch+1, config['epochs'], train_rec_loss_avg[-1]*512*512))

        num_batches = 0

        model.eval()
        val_loss_avg.append(0)

        with torch.no_grad():
            for sample_batched in val_loader:
                image, target, mask = sample_batched['image'].to(device), sample_batched['landmarks'].to(device), sample_batched['mask'].to(device)

                out = model(image)
                if len(out) > 1:
                    out = out[0]

                seg_loss = dice_loss(out, mask) + ce_loss(out, mask)

                loss_rec = seg_loss
                val_loss_avg[-1] += seg_loss.item()
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
    
    parser.add_argument("--name", type=str, default=datetime.datetime.now().strftime('%m_%d_%H_%M_%S')+'_RASTERIZE')
    parser.add_argument("--load", default = "../weights/hybridgnet_weights/bestMSE_JSRT_Padchest.pt", type=str)
    parser.add_argument("--inputsize", default = 1024, type=int)
    parser.add_argument("--epochs", default = 2500, type = int)
    parser.add_argument("--lr", default = 1e-4, type = float)
    parser.add_argument("--stepsize", default = 50, type = int)
    parser.add_argument("--gamma", default = 0.9, type = float)
    parser.add_argument("--workers", default=8, type=int)

    # Number of filters at low resolution
    parser.add_argument("--f", default = 32, type=int)
    # K-hops parameter
    parser.add_argument("--K", default = 6, type=int)
            
    # Arguments for skip connections
    parser.add_argument("--layer", default = 6, type = int)
    parser.add_argument("--w", default = 3, type = int)

    # For double skip connections
    parser.add_argument("--l1", default = 6, type = int)
    parser.add_argument("--l2", default = 5, type = int)

    # Defining what model to use: Default 'HybridGNet'
    parser.add_argument('--IGSC', dest='IGSC', action='store_true')
    parser.add_argument('--no-IGSC', dest='IGSC', action='store_false')
    parser.set_defaults(IGSC=True)

    parser.add_argument('--dir', type=str, default='../')
    
    config = parser.parse_args()
    config = vars(config)

    print('Name: ', config['name'])
    print('Loading JSRT & Padchest Dataset.. ')

    train_path = config['dir'] + "Datasets/JSRT_Padchest/Train"
    val_path = config['dir'] + "Datasets/JSRT_Padchest/Val"
    transforms_train = [RandomScale(), Rotate(3), AugColor(0.40), ToTensor()]
    transforms_val = [ToTensor()]
    A, AD, D, U, E, ED = genMatrixesLH()
    print(A.shape, AD.shape, D.shape, U.shape)

    W = config['w']
    config['window'] = (W,W)
        
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

    print('Model: HybrigGNet with 2 skip connections + Rasterizer')
    model = Hybrid_Rasterize(config, D_t, U_t, A_t)

    model = load_weights(model, config['load'])

    trainer(train_dataset, val_dataset, model, config)