import math
from physioex.train.networks.base import SleepModule
import torch
import torch.nn.functional as F
from physioex.train.networks.spectrogram import ProtoSleepNet, PositionalEncoding
from scipy.signal import welch

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SampleReconstructor(SleepModule):
    def __init__(self, module_config: dict):
        super(SampleReconstructor, self).__init__(NN(module_config), module_config)
        self.step = 0
        self.section_length = module_config["proto_lenght"]
        self.num_sections = 3000 // self.section_length
        self.N = module_config["N"]
        
    def log_reconstructions(self, x, x_hat, target, output, log):        
        # x shape : (N, hidden_size)
        # convert the model parameters to numpy

        x = x.clone().detach().cpu().numpy()
        x_hat = x_hat.clone().detach().cpu().numpy()
        target = target.clone().detach().cpu().numpy()
        output = output.clone().detach().cpu().numpy()

        
        target = np.expand_dims(target, axis=-1)
        target = np.repeat(target, self.N, axis=-1)
        target = target.flatten()
        
        output = np.argmax(output, axis=-1)
        output = np.expand_dims(output, axis=-1)
        output = np.repeat(output, self.N, axis=-1)
        output = output.flatten()

        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x = x[indices]
        x_hat = x_hat[indices]
        target = target[indices]
        output = output[indices]

        # y label equal to prototype index, x label equal to the feature index
        # display the cbar with diverging colors
        fig, axes = plt.subplots(4, 5, figsize=(25, 20))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            sns.lineplot(x=range(self.section_length), y=x[i].flatten(), ax=ax, color="blue", label="Original")
            sns.lineplot(x=range(self.section_length), y=x_hat[i].flatten(), ax=ax, color="red", label="Reconstructed")
            
            # Add title for each subplot
            ax.set_title(f"Target: {target[i]}, Output: {output[i]}")
            
        # Create a single global legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', title="Legend")
        
        self.logger.experiment.add_figure(f"{log}-reconstruction", fig, self.step)
        plt.close()
    
    def training_step(self, batch, batch_idx):
        # get the logged metrics
        if "val_loss" not in self.trainer.logged_metrics:
            self.log("val_loss", float("inf"))
        # Logica di training
        inputs, targets = batch
        x_hat, x, clf = self.forward(inputs)

        return self.compute_loss(x_hat, x, targets, clf, "train")

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        inputs, targets = batch
        x_hat, x, clf = self.forward(inputs)

        return self.compute_loss(x_hat, x, targets, clf, "val")

    def test_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets = batch
        x_hat, x, clf  = self.forward(inputs)

        return self.compute_loss(x_hat, x, targets, clf, "test", log_metrics=True)

    def compute_loss(self, 
                    x_hat, 
                    x,
                    targets,
                    clf,
                    log: str = "train", 
                    log_metrics: bool = False):
        self.step += 1
        
        if self.step % 250 == 0:
            self.log_reconstructions(x, x_hat, targets, clf, log)
        
        x = x.reshape(-1, self.section_length)
        mse_loss = torch.nn.functional.mse_loss(x, x_hat)
        '''std_x = torch.sqrt(torch.var(x, dim=1))
        std_x_hat = torch.sqrt(torch.var(x_hat, dim=1))
        std_loss = torch.nn.functional.mse_loss(std_x, std_x_hat)'''   

        total_loss = mse_loss# + std_loss

        # Log individual losses
        self.log(f"{log}_mse_loss", mse_loss, prog_bar=True, on_step=True, on_epoch=True)
        #self.log(f"{log}_std_loss", std_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{log}_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{log}_acc", -total_loss, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss       

class NN(nn.Module):
    def __init__(self, config: dict):
        super(NN, self).__init__()
        self.N = config["N"]
        self.section_length = config["proto_lenght"]
        self.num_sections = 3000 // self.section_length
        self.num_components = 300
        self.hidden_size = 25
        self.dim_ff = 128  
        self.proto_dim = 64
        self.nhead = 2

        # Load the pre-trained ProtoSleepNet model
        self.proto_model = ProtoSleepNet.load_from_checkpoint(config["proto_ck"], 
                                                              module_config=config)

        # Freeze the parameters of the ProtoSleepNet model
        for param in self.proto_model.parameters():
            param.requires_grad = False

        self.reconstruction_linear = nn.Sequential(
            nn.Linear(self.proto_dim, self.num_components),
            nn.ReLU(),
            nn.LayerNorm(self.num_components),
            nn.Dropout(0.5)
        )

        self.encoder = nn.TransformerEncoderLayer(
            d_model= self.hidden_size*2,
            nhead=self.nhead,
            dim_feedforward=self.dim_ff,
            dropout=0.25,
            batch_first=True
        )
       
        self.encoder = nn.TransformerEncoder( self.encoder, num_layers=2 )

        self.pe = PositionalEncoding(self.hidden_size*2, 1024)

    def reconstructor(self, x):
        x_hat = self.reconstruction_linear(x)
        x_hat = x_hat.reshape(-1, self.num_components//self.hidden_size, self.hidden_size)
        x_hat = x_hat.repeat(1,1,2)
        x_hat = self.pe(x_hat)
        x_hat = self.encoder(x_hat)
        x_hat = x_hat.reshape(-1, self.num_components//self.hidden_size, 2, self.hidden_size)
        x_hat = x_hat.mean(dim=2)
        return x_hat        

    def forward(self, x):
        # x: x[0] preprocessing xsleepnet, x[1] preprocessing raw
        batch_size, seq_len, _, time_steps = x[1].size()

        xsleepnet = x[0]
        raw = x[1].reshape(batch_size*seq_len, time_steps//100, 100)

        window_size = 3
        step_size = 2

        num_windows = (raw.size(1) - window_size) // step_size + 1

        raw = torch.as_strided(
            raw,
            size=(batch_size*seq_len, num_windows, window_size * raw.size(2)),
            stride=(raw.stride(0), step_size * raw.stride(1), raw.stride(2)),
        )

        raw = torch.cat([raw, raw[:, -1:, :]], dim=1)  # (bs*sl, 15, 300)
        # Get the prototypes from the frozen ProtoSleepNet model
        with torch.no_grad():
            proto, clf, _, res, indexes = self.proto_model.nn.encode(xsleepnet)
    
        batch_indices = torch.arange(batch_size*seq_len).unsqueeze(-1)
        original_section = raw[batch_indices, indexes] 
        z = proto + res
        x_hat = self.reconstructor(z).reshape(-1, self.num_components)

        return x_hat, original_section, clf
