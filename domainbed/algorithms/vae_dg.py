import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from domainbed import networks

"""
This code contains the VAE architecture from the project Variational Autoencoder (VAE) + Transfer learning (ResNet + VAE)
https://github.com/hsinyilin19/ResNetVAE/tree/master
"""

def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape

def convtrans2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = ((img_size[0] - 1) * stride[0] - 2 * padding[0] + kernel_size[0],
                (img_size[1] - 1) * stride[1] - 2 * padding[1] + kernel_size[1])
    return outshape

class ResNet_VAE(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, fc_hidden1=1024, fc_hidden2=768, CNN_embed_dim=256):
        super(ResNet_VAE, self).__init__()
        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim
        self.num_classes = num_classes
        self.loss_multiplier_y = hparams.loss_multiplier_y
        self.loss_multiplier_kl = hparams.loss_multiplier_kl
        self.qy = qy(CNN_embed_dim, self.num_classes)

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        self.resnet = networks.Featurizer(input_shape, hparams)
        if hparams.model in ['resnet50', 'resnet152']:
            in_features = 2048
        else:
            in_features = 512

        self.fc1 = nn.Linear(in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )

    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)        
        x_reconst = self.decode(z)
        y_hat = self.qy(z)
        return x_reconst, mu, logvar, y_hat
 
    def classifier(self, x):
        with torch.no_grad():
            z_q_loc, _ = self.encode(x)
            z = z_q_loc
            logits = self.qy(z)
        return logits

    def loss_function(self, x, y):
        recon_x, mu, logvar, y_hat = self.forward(x)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        CE_y = F.cross_entropy(y_hat, y, reduction='sum')
        total_loss = recon_loss + self.loss_multiplier_kl * KLD + self.loss_multiplier_y * CE_y
        return total_loss, recon_loss , self.loss_multiplier_kl * KLD , self.loss_multiplier_y * CE_y
        
class qy(nn.Module):
  def __init__(self, latent_dim, num_classes):
    super(qy, self).__init__()
    self.fc1 = nn.Linear(latent_dim, num_classes)
    self.relu = nn.ReLU()
  
  def forward(self, z):
    h = self.relu(z)
    loc_y = self.fc1(h)
    return loc_y
  
