import sys
sys.path.append('..')
import os
import torch
import argparse
import datetime

from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

from models.unet import UNet, DiceLoss
from torch.nn import CrossEntropyLoss

from utils.dataLoader_XR import LandmarksDataset, ToTensor, RandomScale, AugColor, Rotate

from medpy.metric.binary import dc
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
    val_dice_avg = []

    folder = os.path.join(config['dir']+"Results", config['name'])
    try:
        os.mkdir(folder)
    except:
        pass

    best = 0
    
    print('Training ...')

    dice_loss = DiceLoss().to(device)
    ce_loss = CrossEntropyLoss().to(device)

    scheduler = StepLR(optimizer, step_size=config['stepsize'], gamma=config['gamma'])
    
    for epoch in range(config['epochs']):

        start = time.time()

        model.train()

        train_loss_avg.append(0)
        num_batches = 0
        
        for sample_batched in train_loader:
            image, mask, id = sample_batched['image'].to(device), sample_batched['mask'].to(device), sample_batched['id']

            out = model(image)

            optimizer.zero_grad()

            loss = dice_loss(out, mask) + ce_loss(out, mask)
            train_loss_avg[-1] += loss.item()

            loss.backward()
            optimizer.step()

            num_batches += 1

        train_loss_avg[-1] /= num_batches
        num_batches = 0

        model.eval()
        val_dice_avg.append(0)

        with torch.no_grad():
            for sample_batched in val_loader:
                image, mask, id = sample_batched['image'].to(device), sample_batched['mask'].cpu().numpy(), sample_batched['id']

                out = model(image)
                seg = torch.argmax(out[0, :, :, :], dim=0).cpu().numpy()
                dcl = dc(seg == 1,  mask[0, :, :] == 1)

                val_dice_avg[-1] += dcl
                num_batches += 1   
                loss_rec = 0

        val_dice_avg[-1] /= num_batches
        
        print('Epoch [%d / %d] validation Dice: %f' % (epoch + 1, config['epochs'], val_dice_avg[-1]))

        end = time.time()
        epoch_time = end - start
        print('Epoch time: {:.2f}s'.format(epoch_time))

        if val_dice_avg[-1] > best:
            best = val_dice_avg[-1]
            print('Model Saved Dice')
            out = "bestDice.pt"
            torch.save(model.state_dict(), os.path.join(folder, out))

        scheduler.step()
    
    torch.save(model.state_dict(), os.path.join(folder, "final.pt"))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type=str, default=datetime.datetime.now().strftime('%m_%d_%H_%M_%S')+'_UNET')
    parser.add_argument("--load", help="enter the folder where the weights are saved", default = "None", type=str)
    parser.add_argument("--inputsize", default=1024, type=int)
    parser.add_argument("--epochs", default = 2500, type = int)
    parser.add_argument("--lr", default = 1e-4, type = float)
    parser.add_argument("--stepsize", default = 3000, type = int)
    parser.add_argument("--gamma", default = 0.1, type = float)
    parser.add_argument('--dir', type=str, default='../')
    parser.add_argument('--rtn_all', dest='rtn_all', action='store_true')
    parser.set_defaults(rtn_all=False)
    parser.add_argument("--n_classes", default=2, type=int)
    
    config = parser.parse_args()
    config = vars(config)

    print('Name: ',config['name'])
    print('Loading Montgomery Shenzen Dataset.. ')

    train_path = config['dir'] + "Datasets/Montgomery_Shenzen/Train"
    val_path = config['dir'] + "Datasets/Montgomery_Shenzen/Val"
    transforms_train = [RandomScale(), Rotate(3), AugColor(0.40), ToTensor()]
    transforms_val = [ToTensor()]


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

    config['batch_size'] = 4
    config['val_batch_size'] = 1
    config['weight_decay'] = 1e-5

    model = UNet(n_classes=config['n_classes'], rtn_all=config['rtn_all'])

    trainer(train_dataset, val_dataset, model, config)