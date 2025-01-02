import math
import time
from physioex.train.networks.base import SleepModule
import torch
import torch.nn.functional as F
from physioex.train.networks.old_prototype_kl import ProtoSleepNet, NN

from scipy.signal import spectrogram
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

class SampleReconstructor(SleepModule):
    def __init__(self, module_config: dict):
        super(SampleReconstructor, self).__init__(NN(module_config), module_config)
        self.step = 0
        self.num_bins = 64
        self.bin_edges = torch.linspace(0.0, 1.0, self.num_bins + 1, device="cuda")
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.bandwidth = (self.bin_edges[1] - self.bin_edges[0])

    def log_reconstructions(self, x, x_hat, log : str):        
        # x shape : (N, hidden_size)
        # convert the model parameters to numpy

        x = x.clone().detach().cpu().numpy()
        x_hat = x_hat.clone().detach().cpu().numpy()
        
        # y label equal to prototype index, x label equal to the feature index
        # display the cbar with diverging colors
        fig, axes = plt.subplots(4, 5, figsize=(25, 20))
        fig_s, axes_s = plt.subplots(4, 5, figsize=(25, 20))
        axes = axes.flatten()
        axes_s = axes_s.flatten()
        for i, (ax, ax_s) in enumerate(zip(axes, axes_s)):
            x_max = max(x[i].max(), x_hat[i].max())
            x_min = min(x[i].min(), x_hat[i].min())
            sns.heatmap(x[i], ax=ax, label="Original", vmin=x_min, vmax=x_max, cmap="coolwarm", cbar=False)
            sns.heatmap(x_hat[i], ax=ax_s, label="Reconstructed", vmin=x_min, vmax=x_max, cmap="coolwarm", cbar=False)

        self.logger.experiment.add_figure(f"{log}-original", fig, self.step )
        self.logger.experiment.add_figure(f"{log}-reconstruction", fig_s, self.step)
        plt.close()    

    
    def training_step(self, batch, batch_idx):
        # get the logged metrics
        if "val_loss" not in self.trainer.logged_metrics:
            self.log("val_loss", float("inf"))
        # Logica di training
        inputs, _ = batch
        x_hat, x = self.forward(inputs)

        return self.compute_loss(x_hat, x, "train")

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        inputs, _ = batch
        x_hat, x = self.forward(inputs)

        return self.compute_loss(x_hat, x, "val")

    def test_step(self, batch, batch_idx):
        # Logica di training
        inputs, _ = batch
        x_hat, x = self.forward(inputs)

        return self.compute_loss(x_hat, x, "test", log_metrics=True)
    
    def calculate_entropy(self, x):
        # x_shape = [-1, 5*128]
        x = torch.abs(x)
        D = torch.sum(x, dim=-1)
        x = torch.einsum("ij,i->ij", x, 1.0 / D)

        entropy = x * torch.log(x + 1e-8)
        entropy = -torch.sum(entropy, dim=-1)
        return entropy

    def compute_loss(self, 
                    Sxx_hat, 
                    Sxx, 
                    log: str = "train", 
                    log_metrics: bool = False):
        self.step += 1
        
        if self.step % 250 == 0:
            self.log_reconstructions(Sxx, Sxx_hat, log)

        Sxx = Sxx.reshape(-1, 5*128)
        Sxx_hat = Sxx_hat.reshape(-1, 5*128)
        mse_loss = torch.nn.functional.mse_loss(Sxx, Sxx_hat)

        x_hat_entropy = self.calculate_entropy(Sxx_hat)
        x_entropy = self.calculate_entropy(Sxx)
        
        entropy_loss = torch.nn.functional.mse_loss(x_hat_entropy, x_entropy)

        total_loss = mse_loss + entropy_loss

        # Log individual losses
        self.log(f"{log}_mse_loss", mse_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{log}_entropy_loss", entropy_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{log}_total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=5000):
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        # shape x : (batch_size, seq_len, hidden_size)
        x = x + self.pe[:, : x.size(1)]
        return x           

class NN(nn.Module):
    def __init__(self, config: dict):
        super(NN, self).__init__()
        # Load the pre-trained ProtoSleepNet model
        self.proto_model = ProtoSleepNet.load_from_checkpoint(config["proto_ck"], 
                                                              module_config=config)

        # Freeze the parameters of the ProtoSleepNet model
        for param in self.proto_model.parameters():
            param.requires_grad = False

        self.pe = PositionalEncoding(128)
        
        # Define the reconstruction layers using transposed convolutions
        self.reconstruction_layer = nn.Sequential(
            nn.Linear(256, 128*5),
            nn.ReLU(),
            nn.LayerNorm(128*5),
            nn.Dropout(0.25),
        )

        self.residual_layer = nn.Sequential(
            nn.Conv2d( 1, 8, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d( 8, 16, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d( 16, 1, 5, 1, 2),
            nn.ReLU(),
        )
        
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=128,
                                                            nhead=4, 
                                                            dim_feedforward=256,
                                                            activation='relu',
                                                            batch_first=True
                                                            )
        
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)

        nn.init.constant_(self.residual_layer[0].weight, 0.0)
        nn.init.constant_(self.residual_layer[0].bias, 0.0)
        nn.init.constant_(self.residual_layer[3].weight, 0.0)
        nn.init.constant_(self.residual_layer[3].bias, 0.0)
        nn.init.constant_(self.residual_layer[6].weight, 0.0)
        nn.init.constant_(self.residual_layer[6].bias, 0.0)

    def deconvolve(self, z):        
        # Reconstruct the original sample from the combined tensor
        z_hat = self.reconstruction_layer(z)
        z_hat = z_hat.reshape(-1, 5, 128)
        z_hat = self.pe(z_hat)
        z_hat = self.transformer(z_hat)
        z_hat = z_hat.reshape(-1, 1, 5, 128)
        
        return (z_hat + self.residual_layer(z_hat))
    
    def forward(self, x):
        # x = [batch_size, seq_len, n_channels, 3000]
        print("X", x.shape)
        #print("X2", x.reshape(-1,3,5,129).shape)
        #start_forward = time.time()
        x = x.reshape(-1, 1, 600)   # [batch_size*seq_len*num_sections, 1, 600]
        
        # Get the prototypes from the frozen ProtoSleepNet model
        with torch.no_grad():
            p = self.proto_model.nn.epoch_encoder.conv1(x)  # Shape: [batch_size*seq_len*num_sections, 256]
            proto, res, _ = self.proto_model.nn.epoch_encoder.prototype(p)  # Shape: [batch_size*seq_len*num_sections, 256]


        start_spectrogram = time.time()
        _, _, Sxx = spectrogram(
            x.detach().cpu().numpy().astype(np.double),
            fs=100,
            window="hamming",
            nperseg=200,
            noverlap=100,
            nfft=256,
        )
        #print(f)
        Sxx = Sxx[:,:, :128]
        # log_10 scale the spectrogram safely (using epsilon)
        Sxx = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)
        Sxx = torch.tensor(Sxx, dtype=torch.float32, device=x.device)
        Sxx = Sxx.permute(0,1,3,2)
        #end_transform = time.time()
        #print(f"Transform time: {end_transform - start_spectrogram}")
            
        z = proto + res
        Sxx_hat = self.deconvolve(z)

        #end_forward = time.time()
        #print(f"Forward time: {end_forward - start_forward}")
        #print(f"Percentage of time in calculating spectrogram: {((end_transform - start_spectrogram) / (end_forward - start_forward) * 100):.1f}%")
        return Sxx_hat, Sxx
