from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

import math
import numpy as np

from collections import OrderedDict

from physioex.train.networks.seqsleepnet import LearnableFilterbank, AttentionLayer
from physioex.train.networks.base import SleepModule

import seaborn as sns
import matplotlib.pyplot as plt

class ProtoSleepNet(SleepModule):
    """
    The ProtoSleepNet class is designed to implement the ProtoSleepNet architecture,
    which combines prototype learning with sequence modeling for sleep stage classification.
    It uses prototypes to learn representations of each epoch and applies a cross-entropy 
    classification loss along with additional regularization terms.
    Args:
        module_config (dict): Configuration dictionary containing model hyperparameters such as:
            - section_length (int): Length of each section (input signal segment).
            - N (int): Number of prototypes to learn.
    """
    def __init__(self, module_config : dict):
        super(ProtoSleepNet, self).__init__(NN(module_config), module_config)
        self.prototype = self.nn.epoch_encoder.prototype        #Learned prototype embeddings from the EpochEncoder.
        self.section_length = module_config["section_length"]   #Length of each section in the input signal.
        self.num_sections = 3000 // self.section_length         #Number of sections, derived from the input length and section length.
        self.N = module_config["N"]                             #Number of prototypes to learn.
        self.step = 0                                           #Current step in training, used for logging.
        self.loss_function = nn.CrossEntropyLoss()              #Loss function used for classification.

    """
    Logs the learned prototypes as a heatmap to visualize their distribution across features.
    """
    def log_prototypes(self):        
        # Retrieve the learned prototypes from the model
        prototypes = self.prototype.prototypes.weight.detach().cpu().numpy()
        # Create and log a heatmap of the prototypes (N x hidden_size)  
        sns.heatmap(prototypes, cmap="RdPu", cbar=True)
        plt.xlabel("Features")
        plt.ylabel("Prototype")
        plt.title("Prototypes")
        # Log the figure to the experiment at the current step
        self.logger.experiment.add_figure(f"Prototypes", plt.gcf(), self.step )
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

        # Get the inputs and targets from the batch
        inputs, targets = batch
        # Encode the input data and compute prototypes, classification outputs, and losses
        prototype, clf, loss, residuals, _ = self.encode(inputs)
        # Compute the total loss based on prototypes, classification, and residuals
        return self.compute_loss(prototype, clf, targets, loss, residuals)

    """
    Defines the logic for the validation step.
    Args:
        batch (tuple): Input batch containing (inputs, targets).
        batch_idx (int): Index of the current batch (not used).
    Returns:
        loss (Tensor): Total loss for the current batch.
    """ 
    def validation_step(self, batch, batch_idx):
        # Get the inputs and targets from the batch
        inputs, targets = batch
        # Encode the input data and compute prototypes, classification outputs, and losses
        prototype, clf, loss, residuals, _ = self.encode(inputs)
        # Compute the total loss based on prototypes, classification, and residuals for validation
        return self.compute_loss(prototype, clf, targets, loss, residuals, "val")

    """
    Defines the logic for the test step.
    Args:
        batch (tuple): Input batch containing (inputs, targets).
        batch_idx (int): Index of the current batch (not used).
    Returns:
        loss (Tensor): Total loss for the current batch.
    """ 
    def test_step(self, batch, batch_idx):
        # Get the inputs and targets from the batch
        inputs, targets = batch
        # Encode the input data and compute prototypes, classification outputs, and losses
        prototype, clf, loss, residuals, _ = self.encode(inputs)
        # Compute the total loss based on prototypes, classification, and residuals for testing
        return self.compute_loss(prototype, clf, targets, loss, residuals, "test", log_metrics=True)
    
    """
    Computes the total loss including classification, prototype learning, and regularization terms.
    Args:
        prototypes (Tensor): The prototypes learned from the epochs.
        outputs (Tensor): The classification outputs from the model.
        targets (Tensor): The ground truth labels.
        encoding_loss (Tensor): The loss associated with the prototype learning.
        residuals (Tensor): The residuals from the reconstruction of the prototypes.
        log (str): A string to specify whether the loss is for training, validation, or test.
        log_metrics (bool): Whether to log additional metrics.
    Returns:
        total_loss (Tensor): The final total loss for the batch.
    """
    def compute_loss(
        self,
        prototypes,
        outputs,
        targets, 
        encoding_loss,
        residuals,
        log: str = "train",
        log_metrics: bool = False,
    ):
        self.step += 1

        # Log prototypes every 250 steps
        if self.step % 250 == 0:
            self.log_prototypes()
        
        batch_size, seq_len, n_class = outputs.size()
        _,_,N_proto,_ = prototypes.size()

        learned_prototypes = self.prototype.prototypes.weight.detach().cpu().numpy()
        # Calculate the variance of the learned prototypes across the feature dimension
        variance_learned_prototypes = learned_prototypes.var(axis=0).mean()
        #prototipation
        prototypes = prototypes.reshape(batch_size * seq_len * N_proto, -1)   # shape : [batch_size * seq_len, 256]
        residuals = residuals.reshape(batch_size * seq_len * N_proto, -1)     # shape : [batch_size * seq_len, proto_size]
        #classification
        outputs = outputs.reshape(-1, n_class)  # shape : [batch_size * seq_len, n_class]
        targets = targets.reshape(-1)           # shape : [batch_size * seq_len]
        # compute KL-divergence between resiudals and normal distribution
        normal_dist = dist.Normal(loc=prototypes, scale=1.0)
        z = prototypes + residuals
        kl_loss = normal_dist.log_prob(z)
        kl_loss = kl_loss.sum(dim=-1).mean() * 0.003
        # Gaussian regularization on prototypes
        gauss_dist = dist.Normal(loc=0.0, scale=1.0)
        proto_gauss_loss = gauss_dist.log_prob(prototypes)
        proto_gauss_loss = proto_gauss_loss.sum(dim=-1).mean() * 0.003    
        # Compute cross-entropy loss for classification
        ce_loss = self.loss_function(outputs, targets)
        # Compute total loss as a combination of classification loss, encoding loss, and regularization
        total_loss = ce_loss + encoding_loss - kl_loss - proto_gauss_loss
        # Log metrics for the current step
        self.log(f"{log}_encoding_loss", encoding_loss, prog_bar=False, on_epoch=False, on_step=True)
        self.log(f"{log}_ce_loss", ce_loss, prog_bar=False, on_epoch=False, on_step=True)
        self.log(f"{log}_kl_loss", kl_loss, prog_bar=False, on_epoch=False, on_step=True)
        self.log(f"{log}_proto_gauss_loss", proto_gauss_loss, prog_bar=False, on_epoch=False, on_step=True)
        self.log(f"{log}_loss", total_loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log(f"{log}_acc", self.wacc(outputs, targets), prog_bar=True, on_epoch=True, on_step=True)
        self.log(f"{log}_f1", self.wf1(outputs, targets), prog_bar=False, on_epoch=True, on_step=True)
        self.log(f"prototype_variance", variance_learned_prototypes, prog_bar=False, on_epoch=True, on_step=False)
        # Log additional metrics if required
        if log_metrics and self.n_classes > 1:
            self.log(f"{log}_ck", self.ck(outputs, targets))
            self.log(f"{log}_pr", self.pr(outputs, targets))
            self.log(f"{log}_rc", self.rc(outputs, targets))
            self.log(f"{log}_macc", self.macc(outputs, targets))
            self.log(f"{log}_mf1", self.mf1(outputs, targets))
        return total_loss
        

class NN(nn.Module):
    """
    The NN class implements a neural network for processing sequential data with
    prototype learning and transformer-based encoding. It consists of:
        - EpochEncoder: to encode epochs into prototypes.
        - SequenceEncoder: to model the sequence of prototypes using transformers.
        - Classifier: to classify the encoded sequence.

    The network uses a combination of prototype learning and transformer-based encoding 
    to generate a sequence representation and output class probabilities.
    Args:
        config (dict): Dictionary containing configuration for model hyperparameters.
    """
    def __init__( self, 
        config : dict
    ):
        super(NN, self).__init__()
        
        assert 3000 % config["section_length"] == 0, "The prototype lenght must be a divisor of 3000"

        self.epoch_encoder = EpochEncoder( 
            hidden_size = config["section_length"], # Section length as hidden size
            attention_size = 128,                   # Attention size for hard attention
            N = config["N"],                        # Number of prototypes to select
            n_prototypes = config["n_prototypes"],  # Number of prototypes to learn
            temperature = 1.0                       # Temperature for Gumbel-Softmax
        )
        # Initialize the SequenceEncoder to process the sequence of prototypes
        self.sequence_encoder = SequenceEncoder(
            hidden_size = self.epoch_encoder.out_size,  # Hidden size from EpochEncoder output
            nhead = 4                                   # Number of attention heads
        )
        # Classifier layer to predict output classes from sequence encoding
        self.clf = nn.Linear( self.epoch_encoder.out_size, config["n_classes"] )
         # Store section length for later use
        self.section_length = config["section_length"]

    """
    Encodes the input sequence into prototypes and a classification output.
    Args:
        x (Tensor): Input tensor of shape (batch_size, seq_len, n_chan, 3000, 2, n_fbins).
    Returns:
        proto (Tensor): Prototypes for each epoch, shape (batch_size, seq_len, N, hidden_size).
        clf (Tensor): Predicted classification output, shape (batch_size, seq_len, n_classes).
        loss (Tensor): Combined loss from the prototype learning.
        residual (Tensor): Residuals from prototype reconstruction.
        indexes (Tensor): Indices of selected prototypes.
    """
    def encode( self, x ):
        batch_size, seq_len, _, _, _ = x.size()
        # Extract sections of length 2 from the input sequence
        sections = x.unfold(dimension=3, size=2, step=2).transpose(4, 5)  # shape: (batch_size, seq_len, n_chan, 14, 2, n_fbins)
        # Copy the 29th section to the 30th place, maintaining sequence length
        last_section = x[:, :, :, 28:29, :].unsqueeze(3)  # shape: (batch_size, seq_len, n_chan, 1, 1, n_fbins)
        last_section = last_section.expand(-1, -1, -1, 1, 2, -1)  # shape: (batch_size, seq_len, n_chan, 1, 2, n_fbins)
        sections = torch.cat((sections, last_section), dim=3)  # shape: (batch_size, seq_len, n_chan, 15, 2, n_fbins)
        # Pass the sections through the EpochEncoder
        proto, residual, loss, indexes = self.epoch_encoder( sections )    # proto shape : (batch_size, seq_len, N, hidden_size)
        # Aggregate the prototypes to form an epoch-level representation
        N = proto.size(2)
        p_clf = torch.sum( proto, dim=2 ) / N   # shape : (batch_size, seq_len, hidden_size)      
        # Pass the aggregated prototypes through the SequenceEncoder      
        clf = self.sequence_encoder( p_clf ).reshape( batch_size*seq_len, -1 )    # shape : (batch_size * seq_len, hidden_size)
        # Classify the sequence representation
        clf = self.clf( clf ).reshape( batch_size, seq_len, -1 )

        return proto, clf, loss, residual, indexes

    """
    Forward pass of the network.
    Args:
        x (Tensor): Input tensor of shape (batch_size, seq_len, n_chan, 3000, 2, n_fbins).
    Returns:
        y (Tensor): Classification output of shape (batch_size, seq_len, n_classes).
    """           
    def forward( self, x ):        
        x, y = self.encode( x )                   
        return y
    

class SequenceEncoder( nn.Module ):
    """
    The SequenceEncoder applies positional encoding and uses a TransformerEncoder 
    to process sequential data. The encoder learns to model dependencies across 
    the sequence using multi-head attention.
    Args:
        hidden_size (int): Dimension of the input and output embeddings for the transformer.
        nhead (int): Number of attention heads for multi-head attention.
    """
    def __init__(self,
        hidden_size : int,
        nhead : int,
    ):
        super(SequenceEncoder, self).__init__()
        assert  hidden_size % nhead == 0, "Hidden size must be a divisor of the number of heads"
        # Create a positional encoding layer for embedding the sequence positions
        self.pe = PositionalEncoding(hidden_size, 1024)
        # Create a transformer encoder layer with specified hidden size, number of heads, and feedforward dimension
        self.encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size, 
            dropout=0.25,
            batch_first=True
        )
        # Stack multiple transformer encoder layers (here 2 layers)
        self.encoder = nn.TransformerEncoder( self.encoder, num_layers=2 )

    """
    Forward pass of the SequenceEncoder.
    Args:
        x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
    Returns:
        Tensor: Encoded sequence of shape (batch_size, seq_len, hidden_size).
    """
    def forward( self, x):
        # Apply positional encoding to the input sequence to embed position information
        x = self.pe(x)
        # Pass the sequence through the transformer encoder
        x = self.encoder(x)
        return x
    

class PrototypeLayer( nn.Module ):
    """
    The PrototypeLayer maps input representations to the closest learned prototype
    and computes the loss
    Args:
        output_size (int): Dimensionality of prototype vectors.
        n_prototypes (int): Number of prototype vectors.
        commitment_cost (float): Scaling factor for the latent loss.
    """ 
    def __init__( self, 
        output_size : int,
        n_prototypes : int = 1,
        commitment_cost : float = 0.25
        ):
        super(PrototypeLayer, self).__init__()

        self.proto_dim = output_size
        self.proto_num = n_prototypes 
        #Learnable prototypes
        self.prototypes = nn.Embedding(self.proto_num, self.proto_dim)
        self.commitment_cost = commitment_cost

    """
    Forward pass of the PrototypeLayer.
    Args:
        x (Tensor): Input tensor of shape (batch_size, seq_len, N, output_size).
    Returns:
        tuple: (proto, residuals, loss)
            - proto (Tensor): Quantized prototype representations (batch_size, seq_len, N, output_size).
            - residuals (Tensor): Difference between input and selected prototype (batch_size, seq_len, N, output_size).
            - loss (Tensor): loss value.
    """
    def forward( self, x ):
        # x shape : (batch_size, seq_len, N, proto_dim)
        x_shape = x.shape
        # Flatten the input for distance computation
        x = x.reshape(-1, self.proto_dim).contiguous()

        # Compute squared Euclidean distance between input and prototypes
        dist = (torch.sum(x**2, dim=1, keepdim=True) # shape: (batch_size * seq_len * N, proto_num)
                    + torch.sum(self.prototypes.weight**2, dim=1)
                    - 2 * torch.matmul(x, self.prototypes.weight.t()))
        # Find the closest prototype for each input
        proto_indexes = torch.argmin(dist, dim=1).unsqueeze(1) # shape : (batch_size * seq_len * N, 1)
        # Create one-hot encoding for selected prototypes
        prototype_OHE = torch.zeros(proto_indexes.shape[0], self.proto_num).to(x.device) # shape: (batch_size * seq_len * N, proto_num)
        prototype_OHE.scatter_(1, proto_indexes, 1)
        # Retrieve the corresponding prototypes
        proto = torch.matmul(prototype_OHE, self.prototypes.weight).view(x_shape)   #shape: (batch_size, seq_len * N, output_size)
        x = x.view(x_shape)
        # Compute the loss components
        e_latent_loss = F.mse_loss(proto.detach(), x)
        q_latent_loss = F.mse_loss(proto, x.detach())

        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        proto = x + (proto - x).detach()
        #Compute residuals
        residuals = x - proto

        return proto, residuals, loss
        
        
class EpochEncoder( nn.Module ):
    """
    The EpochEncoder processes an entire EEG epoch by encoding individual sections, 
    selecting the most relevant ones using hard attention, and mapping them to learned prototypes.
    Args:
        hidden_size (int): Size of section embeddings.
        attention_size (int): Size of the attention representation.
        N (int): Number of sections selected by hard attention.
        n_prototypes (int): Number of prototype representations in the codebook.
        temperature (float): Temperature for Gumbel-Softmax selection.
    """
    def __init__( self, 
        hidden_size : int,
        attention_size : int,
        N : int = 1, 
        n_prototypes : int = 25, 
        temperature : float = 1.0 
        ):
        
        super(EpochEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.N = N
        
        # Extract frequency-related features from each section
        self.encoder = SectionEncoder()
        # Determine the output size of the SectionEncoder dynamically
        self.out_size = self.encoder(torch.randn(1, 2,  129)).shape[1]
        # Hard attention mechanism to select the most relevant sections
        self.sampler = HardAttentionLayer(self.out_size, attention_size, N, temperature)
        # Prototype layer to learn meaningful signal representations
        self.prototype = PrototypeLayer( self.out_size, n_prototypes )

    """
    Forward pass of the EpochEncoder.
    Args:
        x (Tensor): Input tensor of shape (batch_size, seq_len, n_chan, 15, 2, n_fbins).
    Returns:
        (proto, residual, loss, indexes)
            - proto (Tensor): Prototype representations (batch_size, seq_len, N, out_size).
            - residual (Tensor): Residuals between input and prototypes (batch_size, seq_len, N, out_size).
            - loss (Tensor): Loss computed from prototype matching.
            - indexes (Tensor): Indices of the selected sections.
    """  
    def forward( self, x ):
        batch_size, seq_len, n_chan, tot_sections, section_length, n_fbins = x.size()
        x = x.reshape(-1, section_length, n_fbins) #output shape: (batch_size*seq_lem*n_chan*tot_sections, section_length, n_fbins)
        # Encode the sections using SectionEncoder
        x = self.encoder( x ) # shape : (batch_size*seq_len*n_chan*tot_sections, out_size)

        x = x.reshape( batch_size*seq_len, tot_sections*n_chan, self.out_size )
        # Apply hard attention to select the most relevant sections
        sampled_x, indexes = self.sampler( x ) # shape : (batch_size * seq_len, N, out_size)

        x = sampled_x.reshape( batch_size, seq_len, self.N, self.out_size )
        # Map selected representations to learned prototypes
        proto, residual, loss = self.prototype( x )
        # Reshape prototypes and residuals to match expected output dimensions
        proto = proto.reshape( batch_size, seq_len, self.N, self.out_size )
        residual = residual.reshape( batch_size, seq_len, self.N, self.out_size )
        
        return proto, residual, loss, indexes
    

class SectionEncoder(nn.Module):
    """
    Section Encoder module that processes input signals through a learnable filterbank,
    a bidirectional LSTM, and a soft attention mechanism to extract meaningful features.
    """
    def __init__(self):
        super(SectionEncoder, self).__init__()
        # Attention mechanism to focus on relevant parts of the sequence
        self.soft_attention = AttentionLayer(hidden_size=64, attention_size=128)
        # Learnable filterbank for feature extraction in the frequency domain
        self.learn_filterbank = LearnableFilterbank(F=129, in_chan=1, nfilt=32)
        # Bidirectional LSTM for sequential encoding
        self.bi_lstm = nn.LSTM(
            input_size=32, 
            hidden_size=32,  
            num_layers=2,  
            batch_first=True, 
            bidirectional=True,  # Use bidirectional processing
            dropout=0.25  
        )

    """
    Forward pass of the SectionEncoder.
    Args:
        x (Tensor): Input tensor of shape (batch_size*seq_len*n_chan*tot_sections, section_length, 129).
    Returns:
        Tensor: Encoded feature representation of shape (batch_size, 64).
    """
    def forward(self, x):
        # Apply learnable filterbank transformation
        x = self.learn_filterbank(x)  # Output shape: (batch_size, section_length, 32)
        # Process sequence with a bidirectional LSTM
        x, _ = self.bi_lstm(x)  # Output shape: (batch_size, section_length, 64)
        # Apply attention to extract relevant features
        x = self.soft_attention(x)  # Output shape: (batch_size, 64)
        return x


class HardAttentionLayer(nn.Module):
    """
    Hard Attention Layer that selects a subset of important elements from an input sequence using 
    a learned attention mechanism. The selection process is based on the Gumbel-Softmax trick, 
    enabling discrete selection while keeping the operation differentiable.
    Args:
        hidden_size (int): Dimension of input embeddings.
        attention_size (int): Dimension of attention projections.
        N (int): Number of elements to select from the sequence.
        temperature (float): Temperature parameter for the Gumbel-Softmax.
        encoding (nn.Module, optional): Optional encoding module (not used in current implementation).
    """
    def __init__(self, 
            hidden_size : int,
            attention_size : int, 
            N : int = 1,
            temperature : float = 1.0,
            encoding : nn.Module = None
        ):
        super(HardAttentionLayer, self).__init__()

        #controls the sharpness of the Gumbel-Softmax function
        self.temperature = temperature  
        # Adds positional information to the input embeddings
        self.pe = PositionalEncoding(hidden_size, 100)
        # Defines how many elements are selected from the sequence
        self.N = N
        # Linear layers for computing attention query and key representations
        self.Q = nn.Linear(hidden_size, attention_size * N, bias = False)
        self.K = nn.Linear(hidden_size, attention_size * N, bias = False)

    """
    Forward pass of the HardAttentionLayer.
    Args:
        x (Tensor): Input tensor of shape (batch_size*seq_len, section_num*n_chan, hidden_size).
    Returns:
        x (Tensor): Selected elements from the input tensor after attention selection.
        indexes (Tensor): Indices of the selected elements.
    """
    def forward(self, x):
        # x shape : (batch_size * seq_len, section_num*n_chan, hidden_size)
        batch_size, section_num, hidden_size = x.size()
        # encode the sequence with positional encoding
        pos_emb = self.pe(x)
        # calculate the query and key -> output shape: (batch_size, section_num, attention_size * N)
        Q = self.Q(pos_emb)
        K = self.K(pos_emb)
        # Reshape and transpose for multi-head attention structure
        Q = Q.reshape( batch_size, section_num, self.N, -1 ).transpose(1, 2)
        K = K.reshape( batch_size, section_num, self.N, -1 ).transpose(1, 2)
        # Compute attention scores using scaled dot-product attention
        attention = torch.einsum( "bnsh,bnth -> bnst", Q, K ) / ( hidden_size ** (1/2) )
        attention = torch.sum(attention, dim=-1) / section_num
        # Reshape attention scores to match the section dimension
        logits = attention.reshape( batch_size * self.N, section_num )                
        # apply the Gumbel-Softmax trick to select the N most important elements
        alphas = torch.nn.functional.gumbel_softmax(logits, tau=self.temperature, hard=True)
        alphas = alphas.reshape( batch_size, self.N, section_num )
        # Extract indices of the selected elements
        indexes = torch.argmax(alphas, dim=-1)
        # select N elements from the sequence x using alphas
        x = torch.einsum( "bns, bsh -> bnh", alphas, x )
        #x shape: (batch_size*seq_len, N, hidden_size)
        return x, indexes
    

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for transformers.
    Positional encoding is added to the input embeddings to provide information 
    about the relative positions of elements in a sequence.
    Args:
        hidden_size (int): The dimensionality of the model's hidden representations.
        max_len (int, optional): The maximum sequence length the model can handle. Defaults to 5000.
    """
    def __init__(self, hidden_size, max_len=5000):
        super().__init__()

        # Create matrix of [max_len, hidden_size] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, hidden_size)
        # Create a tensor representing positions (shape: [max_len, 1])
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the denominator term for the sinusoidal function
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        # Apply sine to even indices (0, 2, 4, ...) and cosine to odd indices (1, 3, 5, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension (shape: [1, max_len, hidden_size])
        pe = pe.unsqueeze(0)
        # Register 'pe' as a buffer
        self.register_buffer("pe", pe, persistent=False)

    """
    Adds positional encoding to the input tensor.
    Args:
        x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
    Returns:
        Tensor: The input tensor with positional encoding added.
    """
    def forward(self, x):
        # x shape : (batch_size, seq_len, hidden_size)
        # Add the positional encoding
        x = x + self.pe[:, : x.size(1)]
        return x
