import math
from physioex.train.networks.base import SleepModule
import torch
import torch.nn.functional as F
from physioex.train.networks.prototype_kl import ProtoSleepNet, NN
from scipy.signal import welch

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
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            sns.lineplot(x = range(600), y = x[i].flatten(), ax=ax, color="blue", label="Original")
            sns.lineplot(x = range(600), y = x_hat[i].flatten(), ax=ax, color="red", label="Reconstructed")

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        self.logger.experiment.add_figure(f"{log}-reconstruction", fig, self.step )
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
        x = torch.abs(x)
        D = torch.sum(x, dim=-1)
        x = torch.einsum("ij,i->ij", x, 1.0 / D)

        entropy = x * torch.log(x + 1e-8)
        entropy = -torch.sum(entropy, dim=-1)
        return entropy
    
    def welch_psd(self, x, fs=1.0, nperseg=256, noverlap=None, nfft=None):
        if noverlap is None:
            noverlap = nperseg // 2
        if nfft is None:
            nfft = nperseg

        batch_size, signal_len = x.size()   #x shape: [-1, 600]
    
        # Window function
        window = torch.hamming_window(nperseg, periodic=False).to(x.device)
        # Calculate the number of segments
        step = nperseg - noverlap
        shape = (batch_size, (x.size(1) - noverlap) // step, nperseg)   #shape shape: [-1, 11, 100]
        strides = (x.stride(0), x.stride(1) * step, x.stride(1))    #strides shape: [600, 50, 1]
        segments = torch.as_strided(x, size=shape, stride=strides)  #segments shape: [-1, 11, 100]

        # Apply window to each segment
        segments = segments * window
    
        # Compute FFT and power spectral density
        fft_segments = torch.fft.rfft(segments, n=nfft)
        psd = (fft_segments.abs() ** 2) / (fs * window.sum() ** 2)

        # Average over segments
        psd = psd.mean(dim=1)   #psd shape: [-1, 129]
    
        # Frequency axis
        freqs = torch.fft.rfftfreq(nfft, 1 / fs)
    
        return freqs, psd

    def compute_loss(self, 
                    x_hat, 
                    x, 
                    log: str = "train", 
                    log_metrics: bool = False):
        self.step += 1
        
        if self.step % 250 == 0:
            self.log_reconstructions(x, x_hat, log)

        x = x.reshape(-1, 600)
        mse_loss = torch.nn.functional.mse_loss(x, x_hat)

        '''x_hat_entropy = self.calculate_entropy(x_hat)
        x_entropy = self.calculate_entropy(x)
        entropy_loss = torch.nn.functional.mse_loss(x_entropy, x_hat_entropy)*10

        x_hat_std = torch.std(x_hat, dim=-1)
        x_std = torch.std(x, dim=-1)
        std_loss = torch.nn.functional.mse_loss(x_std, x_hat_std)'''

        '''_, x_psd_cuda = self.welch_psd(x, fs=100, nperseg=100, noverlap=50, nfft=256)
        _, x_hat_psd_cuda = self.welch_psd(x_hat, fs=100, nperseg=100, noverlap=50, nfft=256)'''

        #print("PSD_cuda: ", x_psd_cuda.shape, x_hat_psd_cuda.shape)

        _, x_psd = welch(x.clone().detach().cpu().numpy(), fs=100, nperseg=100, noverlap=50, nfft=256)
        _, x_hat_psd = welch(x_hat.clone().detach().cpu().numpy(), fs=100, nperseg=100, noverlap=50, nfft=256)

        #print("PSD: ", x_psd.shape, x_hat_psd.shape)

        psd_loss = torch.nn.functional.mse_loss(torch.tensor(x_psd), torch.tensor(x_hat_psd))       
        #psd_loss_cuda = torch.nn.functional.mse_loss(x_psd_cuda, x_hat_psd_cuda)
        #print("PSD: ", psd_loss.shape, psd_loss_cuda.shape)
       # psd_diff = psd_loss - psd_loss_cuda
        #print("PSD Diff: ", psd_diff)

        total_loss = mse_loss + psd_loss#+ entropy_loss + std_loss

        # Log individual losses
        self.log(f"{log}_mse_loss", mse_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{log}_psd_loss", psd_loss, prog_bar=True, on_step=True, on_epoch=True)
        #self.log(f"{log}_entropy_loss", entropy_loss, prog_bar=True, on_step=True, on_epoch=True)
        #self.log(f"{log}_std_loss", std_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{log}_total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{log}_acc", -total_loss, prog_bar=True, on_step=True, on_epoch=True)

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
        #print("Proto CK: ", config["proto_ck"])
        # Load the pre-trained ProtoSleepNet model
        self.proto_model = ProtoSleepNet.load_from_checkpoint(config["proto_ck"], 
                                                              module_config=config)

        # Freeze the parameters of the ProtoSleepNet model
        for param in self.proto_model.parameters():
            param.requires_grad = False

        self.pe = PositionalEncoding(100)
        
        # Define the reconstruction layers using transposed convolutions
        self.reconstruction_layer = nn.Sequential(
            nn.Linear(256, 100*6),
            nn.ReLU(),
            nn.LayerNorm(100*6),
            nn.Dropout(0.25),
        )

        self.residual_layer = nn.Sequential(
            nn.Conv1d( 1, 8, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Conv1d( 8, 16, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d( 16, 1, 5, 1, 2),
            nn.ReLU(),
        )
        
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=100,
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
        z_hat = z_hat.reshape(-1, 6, 100)
        z_hat = self.pe(z_hat)
        z_hat = self.transformer(z_hat)
        z_hat = z_hat.reshape(-1, 1, 600)
        
        return (z_hat + self.residual_layer(z_hat)).reshape(-1, 600)
    
    def forward(self, x):
        # x = [batch_size, seq_len, n_channels, 3000]
        
        x = x.reshape(-1, 1, 600)   # [batch_size*seq_len*num_sections, 1, 600]

        # Get the prototypes from the frozen ProtoSleepNet model
        with torch.no_grad():
            p = self.proto_model.nn.epoch_encoder.conv1(x)  # Shape: [batch_size*seq_len*num_sections, 256]
            proto, res, _ = self.proto_model.nn.epoch_encoder.prototype(p)  # Shape: [batch_size*seq_len*num_sections, 256]
            
        z = proto + res
        x_hat = self.deconvolve(z)

        return x_hat, x
