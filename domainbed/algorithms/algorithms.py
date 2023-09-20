# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed.algorithms.vae_dg import *
from domainbed import networks
from domainbed.optimizers import get_optimizer


def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone


class ERM(Algorithm):
    """
    Baseline --- Empirical Risk Minimization (ERM) 
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class VAE_DG(Algorithm):
    """
    Our proposed method
    Variational Autoencoders for Domain Generalization
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
        CNN_embed_dim = 256   # latent dim extracted by 2D CNN
        self.network = ResNet_VAE(input_shape, num_classes, hparams, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, CNN_embed_dim=CNN_embed_dim)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        loss, recon_loss , KLD_loss, y_loss = self.network.loss_function(all_x, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"total_loss": loss.item(),
                "recon_loss": recon_loss.item(),
                "KLD_loss": KLD_loss.item(),
                "y_loss": y_loss.item()}

    def predict(self, x):
        return self.network.classifier(x)

