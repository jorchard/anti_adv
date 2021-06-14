
from typing import List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as T

import matplotlib.pyplot as plt
from tqdm import tqdm

class MyNet(nn.Module):
    '''
     net = MyNet(img_size=28)
     
     Creates a neural network to do classification on MNIST.
     It assumes the images will be (img_size)x(img_size).
     
     It projects to a latent space.
     From that latent space, it:
      1) projects to an output classification layer (log softmax), and
      2) projects back down through the network to a reconstruction of the input.
     
    '''
    def __init__(self, img_size=28, latent_dim=5, device=None):
        super().__init__()

        # set device up front
        self.device = device

        # Discriminative network
        self.D = nn.ModuleList()
        
        # Input -> Hidden 1
        self.D.append(nn.Linear(img_size**2, 100))
        self.D.append(nn.ReLU())
        
        # Hidden 1 -> Hidden 2
        self.D.append(nn.Linear(100, latent_dim))
        self.D.append(nn.ReLU())
        
        # Classifier
        self.classifier = nn.Sequential(
                            nn.Linear(latent_dim, 10),
                            nn.LogSoftmax(dim=-1),
                            )

        # The generative network
        self.G = nn.ModuleList()
        
        # Hidden 2 -> Hidden 1
        self.G.append(nn.Linear(latent_dim, 100))
        self.G.append(nn.ReLU())
        
        # Hidden 1 -> Input
        self.G.append(nn.Linear(100, img_size**2))
        self.G.append(nn.Sigmoid())
        
        # Tie the weights of D and G
        # param required b/c ...
        self.G[-2].weight = torch.nn.Parameter(self.D[0].weight.transpose(1,0))
        self.G[-4].weight = torch.nn.Parameter(self.D[2].weight.transpose(1,0))
        self.G[-4].bias.data = self.D[0].bias.data
        
        self.classifier_loss = nn.NLLLoss()
        self.recon_loss = nn.BCELoss()
        self.losses = []
        self.to(device)
        
        # checkpoint states
        self.optimizer = None
        
        
    def forward(self, x):
        '''Latent -> Classification'''
        return self.classifier(self.discriminate(x))

    
    def ae(self,x):
        return self.generate(self.discriminate(x))

    
    def discriminate(self, x):
        '''Input -> Latent'''
        for d in self.D:
            x = d(x)
        return x


    def generate(self, z):
        '''
         Latent -> Input
         x_had = net.generate(z)
         
         Runs the network in reverse, generating a batch of inputs from
         a batch of latent vectors.
         
         Inputs:
          z      (D,latent_dim) tensor of latent vectors
          
         Outputs:
          x_hat  (D,784) tensor containing the batch of inputs
        '''
        for g in self.G:
            z = g(z)
        return z
    
      
    def learn(self, 
              dl, 
              optimizer=None, 
              epochs=10, 
              beta=0.,
              checkpoints=[],
              path: str=None):
        '''
         net.learn(dl, optimizer=None, epochs=10, beta=0.)
         
         Train the network on the dataset represented by the DataLoader dl.
         The default optimizer is Adam().
         
         The targets for the dataset are assumed to be class indices.
         
         beta is the weight for the reconstruction loss.
         
         Args:
         
             checkpoints (Boolean|List[int]): if True, save every 10-epochs. if List[int], save each listed epoch.
             path (str): optional path to save model checkpoints.
        '''
        if optimizer is None:
            print('Need to specify an optimizer')
            return
        else:
            self.optimizer = optimizer
        
        for epoch in tqdm(range(epochs)):
            total_loss = 0.
            count = 0.
            for x, t in dl:
                x = x.to(device)   # for use with a GPU
                t = t.to(device)
                z = self.discriminate(x)
                y = self.classifier(z)
                xhat = self.generate(z)
                loss = self.classifier_loss(y, t) + beta*self.recon_loss(xhat, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()*len(t)
                count += 1.
                # set generator weights to equal discirminator weights
                # hack fix
                self.G[0].weight.data = self.D[2].weight.data.transpose(1,0)
                self.G[-2].weight.data = self.D[0].weight.data.transpose(1,0)
                # delete above hack
            self.losses.append(total_loss/len(dl.dataset))
            #print(f'Epoch: {epoch}, loss: {total_loss/count}')
            if checkpoints:
                self.checkpoint(epoch, checkpoints, path)
                
        plt.figure(figsize=(4,4))
        plt.plot(self.losses); plt.yscale('log');


    def checkpoint(self, 
                   epoch: int, 
                   checkpoints: List[int]=[],
                   path: str="model-checkpoints.pt"):
        """ Save model checkpoints. 
        
            Args:
                epoch (int): Current training epoch.
                checkpoints (List[int]): list of epochs to save model at. if True, save every 10.model 
                path (str): path to save model.pt, not required if checkpoints is False.
                
        """
        
        def save_checkpoint(self, epoch, path):
            path = path.split(".pt")[0] + f"-{epoch}" + ".pt"
            torch.save({'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': self.losses[-1],
                        }, 
        
                        path) 
        if checkpoints is True:
            if epoch % 10 == 0: 
                save_checkpoint(self, epoch, path)
        elif epoch in checkpoints:
            save_checkpoint(self, epoch, path)     