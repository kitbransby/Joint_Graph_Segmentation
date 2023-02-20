import torch
import torch.nn as nn
import torch.nn.functional as F
from models.chebConv import ChebConv, Pool
from models.modelutils import residualBlock
import torchvision.ops.roi_align as roi_align

import numpy as np
from models.unet import UNet

class EncoderConv(nn.Module):
    def __init__(self, config, latents = 64, hw = 32, unet=None):
        super(EncoderConv, self).__init__()
        
        self.latents = latents
        self.c = 4
        
        self.size = self.c * np.array([2,4,8,16,32], dtype = np.intc)
        
        self.maxpool = nn.MaxPool2d(2)

        self.unet_decoder_feats = unet
        
        self.dconv_down1 = residualBlock(1+config['n_classes'], self.size[0])
        self.dconv_down2 = residualBlock(self.size[0], self.size[1])
        self.dconv_down3 = residualBlock(self.size[1], self.size[2])
        self.dconv_down4 = residualBlock(self.size[2], self.size[3])
        self.dconv_down5 = residualBlock(self.size[3], self.size[4])
        self.dconv_down6 = residualBlock(self.size[4], self.size[4])
        
        self.fc_mu = nn.Linear(in_features=self.size[4]*hw*hw, out_features=self.latents)
        self.fc_logvar = nn.Linear(in_features=self.size[4]*hw*hw, out_features=self.latents)

    def forward(self, x):

        unet0 = self.unet_decoder_feats(x)

        x = torch.cat([unet0, x], dim=1)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        
        conv5 = self.dconv_down5(x)
        x = self.maxpool(conv5)
        
        conv6 = self.dconv_down6(x)
        
        x = conv6.view(conv6.size(0), -1)

        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
                
        return x_mu, x_logvar, conv1, conv2, conv3, conv4, conv5, conv6

    
class Hourglass_Concat(nn.Module):
    def __init__(self, config, downsample_matrices, upsample_matrices, adjacency_matrices):
        super(Hourglass_Concat, self).__init__()
        
        self.config = config
        hw = config['inputsize'] // 32
        self.z = config['latents']

        self.unet = self.load_unet(config['unet_weights'], config['n_classes'], config['rtn_all'])
        self.encoder = EncoderConv(config=config, latents = self.z, hw = hw, unet=self.unet)
        
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.kld_weight = 1e-5
                
        n_nodes = config['n_nodes']
        self.filters = config['filters']
        self.K = config['K'] # orden del polinomio
        self.ventana = config['window']
        
        # Genero la capa fully connected del decoder
        outshape = self.filters[-1] * n_nodes[-1]          
        self.dec_lin = torch.nn.Linear(self.z, outshape)

        self.normalization1u = torch.nn.InstanceNorm1d(self.filters[1])
        self.normalization2u = torch.nn.InstanceNorm1d(self.filters[1])
        self.normalization3u = torch.nn.InstanceNorm1d(self.filters[2])
        self.normalization4u = torch.nn.InstanceNorm1d(self.filters[3])
        self.normalization5u = torch.nn.InstanceNorm1d(self.filters[4])
        self.normalization6u = torch.nn.InstanceNorm1d(self.filters[5])

        outsize1 = self.encoder.size[4]
        outsize2 = self.encoder.size[4]
        outsize3 = self.encoder.size[3]
        outsize4 = self.encoder.size[2]
        outsize5 = self.encoder.size[1]
        outsize6 = self.encoder.size[0]

        print('Encoder sizes: ', self.encoder.size)

        self.graphConv_up6 = ChebConv(self.filters[6], self.filters[5], self.K)
        self.graphConv_up5 = ChebConv(self.filters[5] + outsize1 + 2, self.filters[4], self.K)
        self.graphConv_up4 = ChebConv(self.filters[4] + outsize2 + 2, self.filters[3], self.K)
        self.graphConv_up3 = ChebConv(self.filters[3] + outsize3 + 2, self.filters[2], self.K)
        self.graphConv_up2 = ChebConv(self.filters[2] + outsize4 + 2, self.filters[1], self.K)
        self.graphConv_up1 = ChebConv(self.filters[1] + outsize5 + 2, self.filters[1], self.K)

        self.graphConv_out6 = ChebConv(self.filters[5], self.filters[0], 1, bias=False)
        self.graphConv_out5 = ChebConv(self.filters[4], self.filters[0], 1, bias=False)
        self.graphConv_out4 = ChebConv(self.filters[3], self.filters[0], 1, bias=False)
        self.graphConv_out3 = ChebConv(self.filters[2], self.filters[0], 1, bias=False)
        self.graphConv_out2 = ChebConv(self.filters[1], self.filters[0], 1, bias=False)
        self.graphConv_out1 = ChebConv(self.filters[1], self.filters[0], 1, bias=False)
        self.graphConv_out0 = ChebConv(self.filters[1] + outsize6 + 2, self.filters[0], 1, bias=False)

        self.pool = Pool()

        self.reset_parameters()

    def load_unet(self, unet_weights, n_classes, rtn_all):
        unet = UNet(n_classes=n_classes, rtn_all=rtn_all)
        unet.load_state_dict(torch.load(unet_weights))
        for param in unet.parameters():
            param.requires_grad = False
        return unet

        
    def reset_parameters(self):
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)


    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) 
    
    def lookup(self, pos, layer, salida = (1,1)):
        B = pos.shape[0]
        N = pos.shape[1]
        F = layer.shape[1]
        h = layer.shape[-1]
        
        ## Scale from [0,1] to [0, h]
        pos = pos * h
        
        _x1 = (self.ventana[0] // 2) * 1.0
        _x2 = (self.ventana[0] // 2 + 1) * 1.0
        _y1 = (self.ventana[1] // 2) * 1.0
        _y2 = (self.ventana[1] // 2 + 1) * 1.0

        boxes = []
        for batch in range(0, B):
            x1 = pos[batch,:,0].reshape(-1, 1) - _x1
            x2 = pos[batch,:,0].reshape(-1, 1) + _x2
            y1 = pos[batch,:,1].reshape(-1, 1) - _y1
            y2 = pos[batch,:,1].reshape(-1, 1) + _y2
            
            aux = torch.cat([x1, y1, x2, y2], axis = 1)            
            boxes.append(aux)
        skip = roi_align(layer, boxes, output_size = salida, aligned=True)
        vista = skip.view([B, N, -1])

        return vista
        
    def forward(self, x):
        self.mu, self.log_var, conv1, conv2, conv3, conv4, conv5, conv6 = self.encoder(x)

        #print('Actual encoder size: ',[e.shape for e in [conv1, conv2, conv3, conv4, conv5, conv6]])


        if self.training:
            z = self.sampling(self.mu, self.log_var)
        else:
            z = self.mu
        x = self.dec_lin(z)
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1, self.filters[-1])

        x = self.graphConv_up6(x, self.adjacency_matrices[5]._indices())
        x = self.normalization6u(x)
        x = F.relu(x)

        pos1 = self.graphConv_out6(x, self.adjacency_matrices[5]._indices())
        skip1 = self.lookup(pos1, conv6)
        x = torch.cat((x, skip1, pos1), dim=2)
        
        x = self.graphConv_up5(x, self.adjacency_matrices[4]._indices())
        x = self.normalization5u(x)
        x = F.relu(x)
        
        pos2 = self.graphConv_out5(x, self.adjacency_matrices[4]._indices())
        skip2 = self.lookup(pos2, conv5)

        x = torch.cat((x, skip2, pos2), dim=2)

        x = self.graphConv_up4(x, self.adjacency_matrices[3]._indices())
        x = self.normalization4u(x)
        x = F.relu(x)

        pos3 = self.graphConv_out4(x, self.adjacency_matrices[3]._indices())
        skip3 = self.lookup(pos3, conv4)
        x = torch.cat((x, skip3, pos3), dim=2)

        x = self.pool(x, self.upsample_matrices[0])

        x = self.graphConv_up3(x, self.adjacency_matrices[2]._indices())
        x = self.normalization3u(x)
        x = F.relu(x)

        pos4 = self.graphConv_out3(x, self.adjacency_matrices[1]._indices())
        skip4 = self.lookup(pos4, conv3)
        x = torch.cat((x, skip4, pos4), dim=2)

        x = self.graphConv_up2(x, self.adjacency_matrices[1]._indices())
        x = self.normalization2u(x)
        x = F.relu(x)

        pos5 = self.graphConv_out2(x, self.adjacency_matrices[1]._indices())
        skip5 = self.lookup(pos5, conv2)
        x = torch.cat((x, skip5, pos5), dim=2)

        x = self.graphConv_up1(x, self.adjacency_matrices[1]._indices())
        x = self.normalization1u(x)
        x = F.relu(x)

        pos6 = self.graphConv_out1(x, self.adjacency_matrices[1]._indices())
        skip6 = self.lookup(pos6, conv1)
        x = torch.cat((x, skip6, pos6), dim=2)
        
        pos7 = self.graphConv_out0(x, self.adjacency_matrices[1]._indices())
        
        return pos7, pos6, pos5, pos4, pos3, pos2, pos1