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
    """
    Initializes the SampleReconstructor class.
    Args:
        module_config (dict): Configuration dictionary containing the parameters for the reconstructor.
    """
    def __init__(self, module_config: dict):
        super(SampleReconstructor, self).__init__(NN(module_config), module_config)
        self.step = 0                                           # Initialize step to track the training process
        self.section_length = module_config["proto_lenght"]     # Used to split the signal into smaller parts
        self.num_sections = 3000 // self.section_length         # Calculate the number of sections (assuming a fixed 3000 length signal)
        self.N = module_config["N"]                             # Extract the number of prototypes or samples (N)

    """
    Logs the original and reconstructed signals along with target and output labels.
    Args:
        x (Tensor): Original signals
        x_hat (Tensor): Reconstructed signals
        target (Tensor): True labels.
        output (Tensor): Predicted labels.
        log (str): Log tag to differentiate between different runs or experiments.
    """
    def log_reconstructions(self, x, x_hat, target, output, log):        
        x = x.clone().detach().cpu().numpy()                # Original signals
        x_hat = x_hat.clone().detach().cpu().numpy()        # Reconstructed signals
        target = target.clone().detach().cpu().numpy()      # True labels
        output = output.clone().detach().cpu().numpy()      # Predicted labels

        # Prepare target and output for visualization:
        # Expand target and output to match the batch size     
        target = np.expand_dims(target, axis=-1)
        target = np.repeat(target, self.N, axis=-1)
        target = target.flatten()
        
        output = np.argmax(output, axis=-1)
        output = np.expand_dims(output, axis=-1)
        output = np.repeat(output, self.N, axis=-1)
        output = output.flatten()

        # Shuffle indices to randomize the order of plotting
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x = x[indices]
        x_hat = x_hat[indices]
        target = target[indices]
        output = output[indices]

        # y label equal to prototype index, x label equal to the feature index
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

    """
    Defines the logic for the training step.
    Args:
        batch (tuple): Input batch containing (inputs, targets).
        batch_idx (int): Index of the current batch (not used).
    Returns:
        loss (Tensor): Total loss for the current batch.
    """ 
    def training_step(self, batch, batch_idx):
        # get the logged metrics
        if "val_loss" not in self.trainer.logged_metrics:
            self.log("val_loss", float("inf"))
        # Logica di training
        inputs, targets = batch
        x_hat, x, clf = self.forward(inputs)

        return self.compute_loss(x_hat, x, targets, clf, "train")
    
    """
    Defines the logic for the validation step.
    Args:
        batch (tuple): Input batch containing (inputs, targets).
        batch_idx (int): Index of the current batch (not used).
    Returns:
        loss (Tensor): Total loss for the current batch.
    """ 
    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        inputs, targets = batch
        x_hat, x, clf = self.forward(inputs)

        return self.compute_loss(x_hat, x, targets, clf, "val")
    
    """
    Defines the logic for the test step.
    Args:
        batch (tuple): Input batch containing (inputs, targets).
        batch_idx (int): Index of the current batch (not used).
    Returns:
        loss (Tensor): Total loss for the current batch.
    """ 
    def test_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets = batch
        x_hat, x, clf  = self.forward(inputs)

        return self.compute_loss(x_hat, x, targets, clf, "test", log_metrics=True)
   
    """
    Computes the loss for the reconstruction task and logs the values.

    Args:
        x_hat (Tensor): The reconstructed signals, shape (batch_size, seq_len, section_size).
        x (Tensor): The original signals, shape (batch_size, seq_len, section_size).
        targets (Tensor): The true labels associated with the signals.
        clf (Tensor): Predicted class associated to the signal.
        log (str): A string to specify whether the loss is for training, validation, or test.
        log_metrics (bool): Whether to log additional metrics.
    Returns:
        Tensor: The total computed loss (sum of the MSE loss).
    """
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
        # Compute the Mean Squared Error (MSE) loss between the original and reconstructed signals
        mse_loss = torch.nn.functional.mse_loss(x, x_hat)

        total_loss = mse_loss
        # Log the MSE loss and the total loss for tracking
        self.log(f"{log}_mse_loss", mse_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{log}_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        # Log the accuracy metric, although it’s the negative of the total loss (to minimize it)
        # (This is unusual for accuracy as it typically ranges from 0 to 1, but used here for tracking purposes)
        self.log(f"{log}_acc", -total_loss, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss       


class NN(nn.Module):
    """
    The NN model integrates a frozen ProtoSleepNet for prototype extraction and a Transformer-based 
    reconstruction module. It processes input sequences, extracts prototype representations, and 
    reconstructs them using learned transformations.
    Args:
        config (dict): Configuration dictionary containing model parameters.
    """
    def __init__(self, config: dict):
        super(NN, self).__init__()
        self.N = config["N"]
        self.section_length = config["proto_lenght"]
        self.num_sections = 3000 // self.section_length # Number of sections in the input sequence
        self.num_components = 300                       # Number of components in the reconstructed signal
        self.hidden_size = 25                           # Hidden dimension size
        self.dim_ff = 128                               # Feedforward layer size
        self.proto_dim = 64                             # Dimension of the extracted prototypes
        self.nhead = 2                                  # Number of attention heads in the transformer

        # Load the pre-trained ProtoSleepNet model
        self.proto_model = ProtoSleepNet.load_from_checkpoint(config["proto_ck"], 
                                                              module_config=config)
        # Freeze the parameters of the ProtoSleepNet model
        for param in self.proto_model.parameters():
            param.requires_grad = False
        # Define a linear layer to transform prototype embeddings for reconstruction
        self.reconstruction_linear = nn.Sequential(
            nn.Linear(self.proto_dim, self.num_components),
            nn.ReLU(),
            nn.LayerNorm(self.num_components),
            nn.Dropout(0.5)
        )
         # Define a transformer encoder layer with multi-head attention for processing reconstructions
        self.encoder = nn.TransformerEncoderLayer(
            d_model= self.hidden_size*2,
            nhead=self.nhead,
            dim_feedforward=self.dim_ff,
            dropout=0.25,
            batch_first=True
        )
       # Stack multiple transformer encoder layers
        self.encoder = nn.TransformerEncoder( self.encoder, num_layers=2 )
        # Positional encoding to provide order information to the transformer
        self.pe = PositionalEncoding(self.hidden_size*2, 1024)

    """
    Reconstructs the original sequence from prototype representations.
    Args:
        x (Tensor): Prototype representation of shape (batch_size, proto_dim).
    Returns:
        Tensor: Reconstructed sequence of shape (batch_size, num_sections, hidden_size).
    """
    def reconstructor(self, x):
        # Apply linear transformation to project prototype embeddings to the target space
        x_hat = self.reconstruction_linear(x)
        # Reshape to prepare for transformer processing
        x_hat = x_hat.reshape(-1, self.num_components//self.hidden_size, self.hidden_size)
        # Duplicate the representation along the last dimension
        x_hat = x_hat.repeat(1,1,2)
        # Apply positional encoding
        x_hat = self.pe(x_hat)
        # Pass through the transformer encoder
        x_hat = self.encoder(x_hat)

        x_hat = x_hat.reshape(-1, self.num_components//self.hidden_size, 2, self.hidden_size)
        # Aggregate across the duplicated dimension
        x_hat = x_hat.mean(dim=2)
        return x_hat 
           
    """
    Forward pass of the NN model.
    Args:
        x (tuple): A tuple containing:
            - x[0]: Processed input for ProtoSleepNet (batch_size, seq_len, features).
            - x[1]: Raw signal input (batch_size, seq_len, channels, time_steps).
    Returns:
        Tuple[Tensor, Tensor, Tensor]: 
            - Reconstructed prototype-based representation.
            - Original sections of raw input corresponding to selected prototypes.
            - Classification output from ProtoSleepNet.
    """
    def forward(self, x):
        # x: x[0] preprocessing xsleepnet, x[1] preprocessing raw
        batch_size, seq_len, _, time_steps = x[1].size()
        # Preprocessed input for ProtoSleepNet
        xsleepnet = x[0]
        # Reshape raw input
        raw = x[1].reshape(batch_size*seq_len, time_steps//100, 100)
        # Define sliding window parameters
        window_size = 3
        step_size = 2
        # Compute the number of windows
        num_windows = (raw.size(1) - window_size) // step_size + 1
        # Create overlapping windows using strided operations
        raw = torch.as_strided(
            raw,
            size=(batch_size*seq_len, num_windows, window_size * raw.size(2)),
            stride=(raw.stride(0), step_size * raw.stride(1), raw.stride(2)),
        )
        # Append the last window to ensure all sections have equal length
        raw = torch.cat([raw, raw[:, -1:, :]], dim=1)  # (bs*sl, 15, 300)
        # Get the prototypes from the frozen ProtoSleepNet model
        with torch.no_grad():
            proto, clf, _, res, indexes = self.proto_model.nn.encode(xsleepnet)

        # Extract the corresponding original sections from the raw input
        batch_indices = torch.arange(batch_size*seq_len).unsqueeze(-1)
        original_section = raw[batch_indices, indexes]
        # Compute the full prototype representation
        z = proto + res
        # Reconstruct the sequence from the learned prototype representation
        x_hat = self.reconstructor(z).reshape(-1, self.num_components)

        return x_hat, original_section, clf
