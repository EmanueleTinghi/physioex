import math
from physioex.train.networks.base import SleepModule
import torch
import torch.nn.functional as F
from physioex.train.networks.prototype_kl import ProtoSleepNet
from scipy.signal import welch

import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

class SampleReconstructor(SleepModule):
    def __init__(self, module_config: dict):
        super(SampleReconstructor, self).__init__(NN(module_config), module_config)
        self.step = 0
        self.section_length = module_config["proto_lenght"]
        self.num_sections = 3000 // self.section_length
        self.N = module_config["N"]

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
            sns.lineplot(x = range(self.section_length), y = x[i].flatten(), ax=ax, color="blue", label="Original")
            sns.lineplot(x = range(self.section_length), y = x_hat[i].flatten(), ax=ax, color="red", label="Reconstructed")

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

    def compute_loss(self, 
                    x_hat, 
                    x, 
                    log: str = "train", 
                    log_metrics: bool = False):
        self.step += 1
        
        if self.step % 250 == 0:
            self.log_reconstructions(x, x_hat, log)
        
        x = x.reshape(-1, self.section_length)
        mse_loss = torch.nn.functional.mse_loss(x, x_hat)

        total_loss = mse_loss

        # Log individual losses
        self.log(f"{log}_mse_loss", mse_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{log}_total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{log}_acc", -total_loss, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss       

class NN(nn.Module):
    def __init__(self, config: dict):
        super(NN, self).__init__()
        self.N = config["N"]
        self.section_length = config["proto_lenght"]
        self.num_sections = 3000 // self.section_length
        self.proto_dim = 64

        # Load the pre-trained ProtoSleepNet model
        self.proto_model = ProtoSleepNet.load_from_checkpoint(config["proto_ck"], 
                                                              module_config=config)

        # Freeze the parameters of the ProtoSleepNet model
        for param in self.proto_model.parameters():
            param.requires_grad = False

        self.amplitude_layer = nn.Sequential(
            nn.Linear(self.proto_dim, self.proto_dim*4),
            nn.ReLU(),
            nn.LayerNorm(self.proto_dim*4),
            nn.Dropout(0.3),
            nn.Linear(self.proto_dim*4, 100),
            nn.Dropout(0.2)
        )

        self.phase_layer = nn.Sequential(
            nn.Linear(self.proto_dim, self.proto_dim*4),
            nn.ReLU(),
            nn.LayerNorm(self.proto_dim*4),
            nn.Dropout(0.3),
            nn.Linear(self.proto_dim*4, 100),
            nn.Dropout(0.2)
        )

        self.freqs = torch.arange(0.5, 50.5, 0.5)
        self.t = torch.linspace(0, self.section_length, steps = self.section_length, device = "cuda").unsqueeze(0).unsqueeze(0)

    def sinusoid_extractor(self, z):
        # z shape: [batch_size*seq_len*num_sections*n_channels, proto_dim]
        amps = self.amplitude_layer(z)
        phases = self.phase_layer(z)
        return amps, phases
    
    def print_memory_usage(self):
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

    def signal_reconstruction(self, amps, phases):
        sinusoids = amps.unsqueeze(-1) * torch.sin(2 * torch.pi * self.freqs.unsqueeze(-1) * self.t + phases.unsqueeze(-1))
        signal = sinusoids.sum(dim=1, keepdim=True)  # [-1, 1, section_length]
        signal = (signal + self.residual_layer(signal)).reshape(-1, self.section_length)
        return signal

    def signal_builder(self, z):
        # z shape: [batch_size*seq_len*num_sections*n_channels, proto_dim]
        amps, phases = self.sinusoid_extractor(z)
        z_hat = self.signal_reconstruction(amps, phases)
        return z_hat

    def forward(self, x):
        # x = [batch_size, seq_len, n_channels, 3000]
        
        x = x.reshape(-1, 1, self.section_length)   # [batch_size*seq_len*num_sections*n_channels, 1, section_length] (num_sections = 3000/section_length)

        # Get the prototypes from the frozen ProtoSleepNet model
        with torch.no_grad():
            p = self.proto_model.nn.epoch_encoder.conv1(x)  # Shape: [batch_size*seq_len*num_sections*n_channels, proto_dim]
            proto, res, _ = self.proto_model.nn.epoch_encoder.prototype(p)  # Shape: [batch_size*seq_len*num_sections*n_channels, proto_dim]
            
        z = proto + res
        x_hat = self.signal_builder(z)

        return x_hat, x
