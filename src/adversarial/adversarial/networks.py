from typing import List, Tuple

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# import torchvision.transforms as T

import matplotlib.pyplot as plt
from tqdm import tqdm


class MyNet(nn.Module):
    """
    net = MyNet(img_size=28)

    Creates a neural network to do classification on MNIST.
    It assumes the images will be (img_size)x(img_size).

    It projects to a latent space.
    From that latent space, it:
     1) projects to an output classification layer (log softmax), and
     2) projects back down through the network to a reconstruction of the input.

    """

    def __init__(self, img_size=28, latent_dim=5, device=None):
        super().__init__()

        # set device up front
        self.device = device

        # Discriminative network
        self.D = nn.ModuleList()

        # Input -> Hidden 1
        self.D.append(nn.Linear(img_size ** 2, 100))
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
        self.G.append(nn.Linear(100, img_size ** 2))
        self.G.append(nn.Sigmoid())

        # Tie the weights of D and G
        # param required b/c ...
        self.G[-2].weight = torch.nn.Parameter(self.D[0].weight.transpose(1, 0))
        self.G[-4].weight = torch.nn.Parameter(self.D[2].weight.transpose(1, 0))
        self.G[-4].bias.data = self.D[0].bias.data

        self.classifier_loss = nn.NLLLoss()
        self.recon_loss = nn.BCELoss()
        self.losses = []
        self.to(device)

        # checkpoint states
        self.optimizer = None

    def forward(self, x):
        """Latent -> Classification"""
        return self.classifier(self.discriminate(x))

    def ae(self, x):
        return self.generate(self.discriminate(x))

    def discriminate(self, x):
        """Input -> Latent"""
        for d in self.D:
            x = d(x)
        return x

    def generate(self, z):
        """
        Latent -> Input
        x_had = net.generate(z)

        Runs the network in reverse, generating a batch of inputs from
        a batch of latent vectors.

        Inputs:
         z      (D,latent_dim) tensor of latent vectors

        Outputs:
         x_hat  (D,784) tensor containing the batch of inputs
        """
        for g in self.G:
            z = g(z)
        return z

    def learn(
        self, dl, optimizer=None, epochs=10, beta=0.0, checkpoints=[], path: str = None
    ):
        """
        net.learn(dl, optimizer=None, epochs=10, beta=0.)

        Train the network on the dataset represented by the DataLoader dl.
        The default optimizer is Adam().

        The targets for the dataset are assumed to be class indices.

        beta is the weight for the reconstruction loss.

        Args:

            checkpoints (Boolean|List[int]): if True, save every 10-epochs. if List[int], save each listed epoch.
            path (str): optional path to save model checkpoints.
        """
        if optimizer is None:
            print("Need to specify an optimizer")
            return
        else:
            self.optimizer = optimizer

        for epoch in tqdm(range(epochs)):
            total_loss = 0.0
            count = 0.0
            for x, t in dl:
                x = x.to(self.device)  # for use with a GPU
                t = t.to(self.device)
                z = self.discriminate(x)
                y = self.classifier(z)
                xhat = self.generate(z)
                loss = self.classifier_loss(y, t) + beta * self.recon_loss(xhat, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(t)
                count += 1.0
                # set generator weights to equal discirminator weights
                # hack fix
                self.G[0].weight.data = self.D[2].weight.data.transpose(1, 0)
                self.G[-2].weight.data = self.D[0].weight.data.transpose(1, 0)
                # delete above hack
            self.losses.append(total_loss / len(dl.dataset))
            # print(f'Epoch: {epoch}, loss: {total_loss/count}')
            if checkpoints:
                self.checkpoint(epoch, checkpoints, path)

        plt.figure(figsize=(4, 4))
        plt.plot(self.losses)
        plt.yscale("log")

    def checkpoint(
        self,
        epoch: int,
        checkpoints: List[int] = [],
        path: str = "model-checkpoints.pt",
    ):
        """Save model checkpoints.

        Args:
            epoch (int): Current training epoch.
            checkpoints (List[int]): list of epochs to save model at. if True, save every 10.model
            path (str): path to save model.pt, not required if checkpoints is False.

        """

        def save_checkpoint(self, epoch, path):
            path = path.split(".pt")[0] + f"-{epoch}" + ".pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": self.losses[-1],
                },
                path,
            )

        if checkpoints is True:
            if epoch % 10 == 0:
                save_checkpoint(self, epoch, path)
        elif epoch in checkpoints:
            save_checkpoint(self, epoch, path)


class GoodfellowG(nn.Module):
    """[summary]

        Note:  This network assumes rectangular 1-channel inputs.
    Args:
        nn ([type]): [description]
    """

    def __init__(self, img_size, latent_dim, hidden_dim, device):
        super().__init__()

        self.device = device
        self.img_size = img_size
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),  # swap to leaky?
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, img_size),
            nn.Tanh(),
        )

    def forward(self, z):
        """Forward pass of generator, taking p(z) noise input.

        Args:
            z ([torch.tensor]): noise seed tensor

        Returns:
            torch.tensor: tensor of self.img_size
        """
        return self.model(z)

    def sample_latent_space(
        self, latent_dim: int = None, batch_size: int = 32
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Sample a normal distribution for latent space vectors

        Note, that this assumes row-based data (as expected in Pytorch).

        Args:
            latent_size ): latent space vector, size: batch_size X latent_size

        Returns:
            z, u Tuple[torch.tensor]: batch_size X img_size tensor, and list of zeros as labels
        """

        if latent_dim is None:
            latent_dim = self.latent_dim

        return torch.randn(batch_size, latent_dim), torch.zeros(batch_size, 1)


class GoodfellowD(nn.Module):
    def __init__(self, img_size=28, latent_dim=5, device=None):
        super().__init__()

        self.device = device
        self.img_size = img_size
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(img_size, latent_dim),  # [784, 5]
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(latent_dim, latent_dim),  # [5 , 5]
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(latent_dim, 1),  # [5 , 1]
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Sequential model forward inference.

        Args:
            x (np.ndarray): data, as flattened row-vectors (image tensor)

        Returns:
            float: probability score(s) in range [0, 1],
            clamped for space efficiency
        """

        return self.model(x).clamp(1e-9)


class GoodfellowGAN(nn.Module):
    def __init__(
        self,
        G,
        D,
        img_size: int = 784,
        latent_dim: int = 10,
        hidden_dim: int = 5,
        device=None,
    ):
        super().__init__()

        # set device up front
        self.device = device if device else "cpu"

        self.img_size = img_size

        # Generator and Discriminator
        self.G = G
        self.D = D

        # Instantiate criterion used for both D and G
        self.criterion = nn.BCELoss()

        # History
        self.losses = {"G": [], "D": []}

        self.to(device)

    def discriminate(self, x):
        """Input -> Latent"""
        for d in self.D:
            x = d(x)
        return x

    def train(
        self,
        dl: DataLoader,
        optimizer=None,
        epochs=10,
        batch_size: int = 32,
        checkpoints=[],
        path: str = None,
    ):
        """
        net.learn(dl, optimizer=None, epochs=10, beta=0.)

        Train the network on the dataset represented by the DataLoader dl.
        The default optimizer is Adam().

        The targets for the dataset are assumed to be class indices.

        beta is the weight for the reconstruction loss.

        Args:

            checkpoints (Boolean|List[int]): if True, save every 10-epochs. if List[int], save each listed epoch.
            path (str): optional path to save model checkpoints.
        """
        if optimizer is None:
            print("Need to specify an optimizer")
            return
        else:
            self.optimizer = optimizer

        for epoch in tqdm(range(epochs)):

            running_d_loss, running_g_loss = 0.0, 0.0
            for x, t in dl:
                # x = real images
                # t = real labels
                # z = fake images (noise sample)
                # u = fake labels

                # sample latent space to get p(z)

                # Get probability scores for real and fake data
                real_logits = self.D(x)

                z, u = self.G.sample_latent_space(batch_size=batch_size)
                z.to(self.device)
                u.to(self.device)
                fake_imgs = self.G(z)
                fake_logits = self.D(fake_imgs).to(torch.float32)

                d_real_loss = self.criterion(
                    real_logits.to(torch.float32).view(-1), t.to(torch.float32)
                )

                d_fake_loss = self.criterion(
                    fake_logits.to(torch.float32).view(-1), u.to(torch.float32).view(-1)
                )
                d_loss = d_real_loss + d_fake_loss

                # # Backpropagation and update discriminator
                self.optimizer.zero_grad()
                d_loss.backward()
                self.optimizer.step()

                # Resample a NEW set of z inputs
                z, u = self.G.sample_latent_space(batch_size=batch_size)
                z.to(self.device)
                u.to(self.device)
                fake_imgs = self.G(z).to(self.device)
                fake_logits = self.D(fake_imgs).to(self.device)

                g_loss = self.criterion(
                    fake_logits.to(torch.float32).view(-1), t.to(torch.float32)
                )

                # backprop generator and update
                self.optimizer.zero_grad()
                g_loss.backward()
                self.optimizer.step()

                # Keep track of losses and global step
                running_g_loss += g_loss.item()
                running_d_loss += d_loss.item()

            # self.losses.append(total_loss / len(dl.dataset))
            # self.losses["G"].append(g_loss.item() / len(dl.dataset))
            # self.losses["D"].append(d_loss.item() / len(dl.dataset))
            print(f"Epoch: {epoch}, d_loss: {d_loss.item()}")
            if checkpoints:
                self.losses["G"].append(g_loss.item())
                self.losses["D"].append(d_loss.item())
                self.checkpoint(epoch, checkpoints, path)

        # plt.figure(figsize=(4, 4))
        # plt.plot(self.losses)
        # plt.yscale("log")

    def checkpoint(
        self,
        epoch: int,
        checkpoints: List[int] = [],
        path: str = "model-checkpoints.pt",
    ):
        """Save model checkpoints.

        Args:
            epoch (int): Current training epoch.
            checkpoints (List[int]): list of epochs to save model at. if True, save every 10.model
            path (str): path to save model.pt, not required if checkpoints is False.

        """

        def save_checkpoint(self, epoch, path):
            assert path is not None

            outpath = path.split(".pt")[0] + f"-{epoch}" + ".pt"
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Created directory to save checkpoints: {path}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "g_loss": self.losses["G"][-1],
                    "d_loss": self.losses["D"][-1],
                },
                outpath,
            )

        if checkpoints is True:
            if epoch % 10 == 0:
                save_checkpoint(self, epoch, path)
        elif epoch in checkpoints:
            save_checkpoint(self, epoch, path)