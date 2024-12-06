import math
from physioex.train.networks.base import SleepModule
import torch
from physioex.train.networks.prototype_kl import ProtoSleepNet, NN

import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

class SampleReconstructor(SleepModule):
    def __init__(self, module_config: dict):
        super(SampleReconstructor, self).__init__(NN(module_config), module_config)
        self.step = 0

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
            sns.lineplot(x = range(600), y = x[i].flatten(), ax=ax, color="blue")
            sns.lineplot(x = range(600), y = x_hat[i].flatten(), ax=ax, color="red")

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

        x = x.reshape(-1, 600)
        loss = torch.nn.functional.mse_loss(x, x_hat)
        self.log(f"{log}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{log}_acc", -loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss
    
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
        print("Proto CK: ", config["proto_ck"])
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
        
        self.transformer_layer = nn.TransformerEncoderLayer(dmodel=100,
                                                            nhead=8, 
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
        # Concatenate prototypes and residuals along the channel dimension
        #z = prototypes + residuals

        # Reconstruct the original sample from the combined tensor
        z_hat = self.reconstruction_layer(z)
        z_hat = z_hat.reshape(-1, 6, 100)
        z_hat = self.pe(z_hat)
        z_hat = self.transformer(z_hat)
        z_hat = z_hat.reshape(-1, 1, 600)
        
        return (z_hat + self.residual_layer(z_hat)).reshape(-1, 600)
    
    def forward(self, x):
        # x = [batch_size, seq_len, n_channels, 3000]
        
        x = x.reshape(-1, 1, 600)

        # Get the prototypes from the frozen ProtoSleepNet model
        with torch.no_grad():
            p = self.proto_model.nn.epoch_encoder.conv1(x)
            proto, res, _ = self.proto_model.nn.epoch_encoder.prototype(p)

        #print("Proto shape: ", proto.shape)
        #print("Residual shape: ", res.shape)

        z = proto + res
        z_hat = self.deconvolve(z)

        # Decode the input tensor
        #inv_prototypes = self.decode(prototypes)

        # Reconstruct the original sample
        x_hat = self.transformer(z_hat)

        return x_hat, x
