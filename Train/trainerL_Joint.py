import os

import torch
import torch.nn.functional as F
import argparse
import numpy as np
import random

from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

import scipy.sparse as sp

from sklearn.metrics import mean_squared_error
from utils.utils import genMatrixesLungs, scipy_to_torch_sparse
from utils.dataLoader_XR import LandmarksDataset, ToTensor, RandomScale, AugColor, Rotate

from models.joint import Joint

from models.chebConv import Pool
import datetime
import time


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
            image, target = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)
            out = model(image)
            target_down = pool(target, model.downsample_matrices[0])
            
            optimizer.zero_grad()

            pos7, pos6, pos5, pos4, pos3, pos2, pos1 = out

            pos1loss = F.mse_loss(pos1, target_down) / 2
            pos2loss = F.mse_loss(pos2, target_down) / 2
            pos3loss = F.mse_loss(pos3, target_down) / 2
            pos4loss = F.mse_loss(pos4, target) / 2
            pos5loss = F.mse_loss(pos5, target) / 2
            pos6loss = F.mse_loss(pos6, target) / 2
            outloss = F.mse_loss(pos7, target) / 2

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
                image, target = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)

                out = model(image)
                if len(out) > 1:
                    out = out[0]

                out = out.reshape(-1, 2)
                target = target.reshape(-1, 2)

                loss_rec = mean_squared_error(out.cpu().numpy(), target.cpu().numpy())
                val_loss_avg[-1] += loss_rec
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
    
    parser.add_argument("--name", type=str, default=datetime.datetime.now().strftime('%m_%d_%H_%M_%S')+'_JOINT')
    parser.add_argument("--load", help="enter the folder where the weights are saved", default = "None", type=str)
    parser.add_argument("--inputsize", default = 1024, type=int)
    parser.add_argument("--epochs", default = 2500, type = int)
    parser.add_argument("--lr", default = 1e-4, type = float)
    parser.add_argument("--stepsize", default = 50, type = int)
    parser.add_argument("--gamma", default = 0.9, type = float)
    parser.add_argument("--n_classes", default=2, type=int)

    parser.add_argument('--rtn_all', dest='rtn_all', action='store_true')
    parser.set_defaults(rtn_all=True)
    
    # Number of filters at low resolution
    parser.add_argument("--f", default = 32, type=int)
    # K-hops parameter
    parser.add_argument("--K", default = 6, type=int)
            
    # Arguments for skip connections
    parser.add_argument("--layer", default = 6, type = int)
    parser.add_argument("--w", default = 3, type = int)

    parser.add_argument('--dir', type=str, default='../')
    parser.add_argument('--unet_weights', type=str, default='../models/unet_weights/bestDice_Montgomery_Shenzen.pt')
    
    config = parser.parse_args()
    config = vars(config)

    print(config['name'])

    W = config['w']
    config['window'] = (W, W)

    print('Loading Montgomery Shenzen Dataset.. ')

    config['inputsize'] = 1024
    train_path = config['dir'] + "Datasets/Montgomery_Shenzen/Train"
    val_path = config['dir'] + "Datasets/Montgomery_Shenzen/Val"
    transforms_train = [RandomScale(), Rotate(3), AugColor(0.40), ToTensor()]
    transforms_val = [ToTensor()]
    A, AD, D, U, E, ED = genMatrixesLungs()

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

    print('Model: Joint')
    model = Joint(config, D_t, U_t, A_t)

    trainer(train_dataset, val_dataset, model, config)