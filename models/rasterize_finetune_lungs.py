import torch
import torch.nn as nn
import torch.nn.functional as F
from models.chebConv import ChebConv, Pool
from models.modelutils import residualBlock
import torchvision.ops.roi_align as roi_align
from rasterizer.polygon import SoftPolygon

import numpy as np



class EncoderConv(nn.Module):
    def __init__(self, latents = 64, hw = 32):
        super(EncoderConv, self).__init__()
        
        self.latents = latents
        self.c = 4
        
        self.size = self.c * np.array([2,4,8,16,32], dtype = np.intc)
        
        self.maxpool = nn.MaxPool2d(2)
        
        self.dconv_down1 = residualBlock(1, self.size[0])
        self.dconv_down2 = residualBlock(self.size[0], self.size[1])
        self.dconv_down3 = residualBlock(self.size[1], self.size[2])
        self.dconv_down4 = residualBlock(self.size[2], self.size[3])
        self.dconv_down5 = residualBlock(self.size[3], self.size[4])
        self.dconv_down6 = residualBlock(self.size[4], self.size[4])
        
        self.fc_mu = nn.Linear(in_features=self.size[4]*hw*hw, out_features=self.latents)
        self.fc_logvar = nn.Linear(in_features=self.size[4]*hw*hw, out_features=self.latents)

    def forward(self, x):
        x = self.dconv_down1(x)
        x = self.maxpool(x)
        #print('E1: ', x.shape)

        x = self.dconv_down2(x)
        x = self.maxpool(x)
        #print('E2: ', x.shape)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        #print('E3: ', x.shape)
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        #print('E4: ', x.shape)
        
        conv5 = self.dconv_down5(x)
        x = self.maxpool(conv5)
        #print('E5: ', x.shape)
        
        conv6 = self.dconv_down6(x)
        #print('E6: ', x.shape)
        
        x = conv6.view(conv6.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        #print('FLATTEN: ', x.shape)

        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        #print('MU AND LOGVAR: ', x_mu.shape, x_logvar.shape)
                
        return x_mu, x_logvar, conv3, conv4, conv5, conv6

    
class Hybrid_Rasterize(nn.Module):
    def __init__(self, config, downsample_matrices, upsample_matrices, adjacency_matrices):
        super(Hybrid_Rasterize, self).__init__()
        
        self.config = config
        hw = config['inputsize'] // 32
        self.z = config['latents']
        self.encoder = EncoderConv(latents = self.z, hw = hw)

        self.RASTERIZER = SoftPolygon(mode="mask", inv_smoothness=0.1)
        
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
                
        self.normalization2u = torch.nn.InstanceNorm1d(self.filters[1])
        self.normalization3u = torch.nn.InstanceNorm1d(self.filters[2])
        self.normalization4u = torch.nn.InstanceNorm1d(self.filters[3])
        self.normalization5u = torch.nn.InstanceNorm1d(self.filters[4])
        self.normalization6u = torch.nn.InstanceNorm1d(self.filters[5])
        
        if config['l1'] == 6 and config['l2'] == 5:
            outsize1 = self.encoder.size[4]
            outsize2 = self.encoder.size[4]
            print('6-5')
        elif config['l1'] == 5 and config['l2'] == 4:
            outsize1 = self.encoder.size[4]
            outsize2 = self.encoder.size[3]
            print('5-4')
        else:
            outsize1 = self.encoder.size[3]
            outsize2 = self.encoder.size[2]
            print('4-3')           
                    
        # Guardo las capas de convoluciones en grafo
        self.graphConv_up6 = ChebConv(self.filters[6], self.filters[5], self.K)

        self.graphConv_up5 = ChebConv(self.filters[5], self.filters[4], self.K)
        
        # Deep supervised 1
        self.graphConv_pre1 = ChebConv(self.filters[4], self.filters[0], 1, bias = False)
        # Merges
        self.graphConv_up4 = ChebConv(self.filters[4] + outsize1 + 2, self.filters[3], self.K)
        
        self.graphConv_up3 = ChebConv(self.filters[3], self.filters[2], self.K)
        
        # Deep supervised 2
        self.graphConv_pre2 = ChebConv(self.filters[2], self.filters[0], 1, bias = False)
        # Merges
        self.graphConv_up2 = ChebConv(self.filters[2] + outsize2 + 2, self.filters[1], self.K)
        
        self.graphConv_up1 = ChebConv(self.filters[1], self.filters[0], 1, bias = False)
        
        self.pool = Pool()
        
        self.reset_parameters()


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

        #print('_x1 _x2 _y1 _y2: ',_x1 ,_x2 ,_y1 ,_y2)
        
        boxes = []
        for batch in range(0, B):
            x1 = pos[batch,:,0].reshape(-1, 1) - _x1
            x2 = pos[batch,:,0].reshape(-1, 1) + _x2
            y1 = pos[batch,:,1].reshape(-1, 1) - _y1
            y2 = pos[batch,:,1].reshape(-1, 1) + _y2
            
            aux = torch.cat([x1, y1, x2, y2], axis = 1)            
            boxes.append(aux)

        #print('LAYER shape: ', layer.shape)
        #print('boxes length: ', len(boxes))
                    
        skip = roi_align(layer, boxes, output_size = salida, aligned=True)

        #print('skip shape: ', skip.shape)

        vista = skip.view([B, N, -1])

        #print('vista shape: ', vista.shape)

        return vista
        
    def forward(self, x):
        self.mu, self.log_var, conv3, conv4, conv5, conv6 = self.encoder(x)

        if self.training:
            z = self.sampling(self.mu, self.log_var)
        else:
            z = self.mu

        #print('SAMPLING: ', z.shape)
            
        x = self.dec_lin(z)
        x = F.relu(x)
        #print('SAMPLING LINEAR: ', x.shape)
        
        x = x.reshape(x.shape[0], -1, self.filters[-1])
        #print('SAMPLING RESHAPE: ', x.shape)
        
        x = self.graphConv_up6(x, self.adjacency_matrices[5]._indices())
        x = self.normalization6u(x)
        x = F.relu(x)
        #print('D6: ', x.shape)
        
        x = self.graphConv_up5(x, self.adjacency_matrices[4]._indices())
        x = self.normalization5u(x)
        x = F.relu(x)
        #print('D5: ', x.shape)
        
        pos1 = self.graphConv_pre1(x, self.adjacency_matrices[3]._indices()) # Positions where to look
        #print('SUPERVISION 1: ', pos1.shape)
        
        if self.config['l1'] == 6:
            skip = self.lookup(pos1, conv6) 
        elif self.config['l1'] == 5:
            skip = self.lookup(pos1, conv5) 
        else:
            skip = self.lookup(pos1, conv4)
        #print('SKIP 1: ', skip.shape)
            
        x = torch.cat((x, skip, pos1), axis=2) # Concatenating features
        #print('CONCAT 1: ', x.shape)
        
        x = self.graphConv_up4(x, self.adjacency_matrices[3]._indices())
        x = self.normalization4u(x)
        x = F.relu(x)
        #print('D4: ', x.shape)
                
        x = self.pool(x, self.upsample_matrices[0])
        
        # Sigue
        x = self.graphConv_up3(x, self.adjacency_matrices[2]._indices())
        x = self.normalization3u(x)
        x = F.relu(x)
        #print('D3: ', x.shape)
        
        pos2 = self.graphConv_pre2(x, self.adjacency_matrices[1]._indices()) # Sin relu y sin bias
        #print('SUPERVISION 2: ', pos2.shape)
        
        if self.config['l2'] == 5:
            skip2 = self.lookup(pos2, conv5)        
        elif self.config['l2'] == 4:
            skip2 = self.lookup(pos2, conv4)
        else:
            skip2 = self.lookup(pos2, conv3)
        #print('SKIP 2: ', skip2.shape)
            
        x = torch.cat((x, skip2, pos2), axis=2)
        #print('CONCAT 2: ', x.shape)
        
        x = self.graphConv_up2(x, self.adjacency_matrices[1]._indices())
        x = self.normalization2u(x)
        x = F.relu(x)
        #print('D2: ', x.shape)
        
        x = self.graphConv_up1(x, self.adjacency_matrices[0]._indices()) # Sin relu y sin bias
        #print('D1: ', x.shape)

        pos1_raster = self.rasterize(pos1, 1024, scale=2)
        pos2_raster = self.rasterize(pos2, 1024)
        x_raster = self.rasterize(x, 1024)
        
        return x_raster, pos1_raster, pos2_raster

    def rasterize(self, x, resolution, scale=1, offset=-0.50):

        #print('RAZZ x: ', x.shape)

        # RLUNG 44, LLUNG 94, HEART 120

        RLUNG = x[:, :44//scale, :]
        LLUNG = x[:, 44//scale:, :]

        RLUNG_raster = self.RASTERIZER(RLUNG * float(resolution) + offset, resolution, resolution, 0.1)
        LLUNG_raster = self.RASTERIZER(LLUNG * float(resolution) + offset, resolution, resolution, 0.1)

        LUNG_raster = RLUNG_raster + LLUNG_raster
        LUNG_raster = torch.clip(LUNG_raster, 0, 1)

        BACKGROUND = torch.ones((x.shape[0], resolution, resolution), device=x.device)
        BACKGROUND -= LUNG_raster
        BACKGROUND = torch.clip(BACKGROUND, 0, 1)

        raster = torch.stack([BACKGROUND, LUNG_raster], dim=1)

        return raster