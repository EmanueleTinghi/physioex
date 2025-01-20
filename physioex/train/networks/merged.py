from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

import math

from collections import OrderedDict

from physioex.train.networks.base import SleepModule

# from 30-seconds of signal we need to extract the 5 second subsequence
# that is more relevant for the classification task

import seaborn as sns
import matplotlib.pyplot as plt

class ProtoSleepNet(SleepModule):
    def __init__(self, module_config : dict):
        super(ProtoSleepNet, self).__init__(NN(module_config), module_config)

        self.prototypes = self.nn.epoch_encoder.prototype
        self.section_length = module_config["proto_lenght"]
        self.num_sections = 3000 // self.section_length
        self.N = module_config["N"]
        self.step = 0

    def log_prototypes_variance(self, log: str):
        # Ensure the variance values are tracked across epochs
        if not hasattr(self, "variance_history"):
            self.variance_history = []

        # Prototypes shape: (N, hidden_size)
        prototypes = self.prototypes.prototypes.weight.detach().cpu().numpy()

        # Calculate variance between prototypes (axis 0: prototypes)
        variance_between_prototypes = prototypes.var(axis=0).mean()

        # Append the variance to the history
        self.variance_history.append(variance_between_prototypes)

        # Create a plot for the variance history
        plt.figure()
        plt.plot(self.variance_history, marker="o")
        plt.xlabel("Log number")
        plt.ylabel("Variance")
        plt.title("Variance Between Prototypes Over Teps")

        # Log the figure
        self.logger.experiment.add_figure(f"{log}-proto-variance-history", plt.gcf(), self.step)
        plt.close()

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

    def log_prototypes(self, log : str):        
        # prototypes shape : (N, hidden_size)
        # convert the model parameters to numpy
        prototypes = self.prototypes.prototypes.weight.detach().cpu().numpy()

        # create the heatmap of the prototypes
        # y label equal to prototype index, x label equal to the feature index
        # display the cbar with diverging colors
        sns.heatmap(prototypes, cmap="RdPu", cbar=True)
        plt.xlabel("Features")
        plt.ylabel("Prototype")
        
        plt.title("Prototypes")
        self.logger.experiment.add_figure(f"{log}-age-corr", plt.gcf(), self.step )
        plt.close()
 
    def training_step(self, batch, batch_idx):
        # get the logged metrics
        if "val_loss" not in self.trainer.logged_metrics:
            self.log("val_loss", float("inf"))
        # Logica di training
        inputs, targets = batch
        prototype, clf, loss, residuals, reconstructed_section, original_section = self.encode(inputs)

        return self.compute_loss(prototype, clf, targets, loss, residuals, reconstructed_section, original_section)

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        inputs, targets = batch
        prototype, clf, loss, residuals, reconstructed_section, original_section = self.encode(inputs)

        return self.compute_loss(prototype, clf, targets, loss, residuals, reconstructed_section, original_section, "val")

    def test_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets = batch

        prototype, clf, loss, residuals, reconstructed_section, original_section = self.encode(inputs)

        return self.compute_loss(prototype, clf, targets, loss, residuals, reconstructed_section, original_section, "test", log_metrics=True)       

    def compute_loss(
        self,
        prototypes,
        outputs,
        targets, 
        encoding_loss,
        residuals,
        reconstructed_section, 
        original_section,
        log: str = "train",
        log_metrics: bool = False,
    ):
        self.step += 1
        reconstructed_section = reconstructed_section.reshape(-1, self.section_length)
        original_section = original_section.reshape(-1, self.section_length)
        
        if self.step % 250 == 0:
            #self.log_prototypes(log)
            #self.log_prototypes_variance(log)
            self.log_reconstructions(original_section, reconstructed_section, log)
        
        batch_size, seq_len, n_class = outputs.size()
        #print("prototo shape", prototypes.size())
        _,_,N_proto,_ = prototypes.size()

        #prototipation
        prototypes = prototypes.reshape(batch_size * seq_len * N_proto, -1)   # shape : [batch_size * seq_len, 256]
        residuals = residuals.reshape(batch_size * seq_len * N_proto, -1)     # shape : [batch_size * seq_len, proto_size]

        #classification
        outputs = outputs.reshape(-1, n_class)  # shape : [batch_size * seq_len, n_class]
        targets = targets.reshape(-1)           # shape : [batch_size * seq_len]

        # compute KL-divergence between resiudals and normal distribution
        normal_dist = dist.Normal(loc=prototypes, scale=1.0)
        z = prototypes + residuals
 
        #TODO log prob che z appartenga alla distribuzione
        kl_loss = normal_dist.log_prob(z)
        kl_loss = kl_loss.sum(dim=-1).mean() * 0.003

        #
        gauss_dist = dist.Normal(loc=0.0, scale=1.0)
        proto_gauss_loss = gauss_dist.log_prob(prototypes)
        proto_gauss_loss = proto_gauss_loss.sum(dim=-1).mean() * 0.003

        reconstruction_loss = torch.nn.functional.mse_loss(original_section, reconstructed_section)
        std_x = torch.sqrt(torch.var(original_section, dim=1))
        std_x_hat = torch.sqrt(torch.var(reconstructed_section, dim=1))
        std_loss = torch.nn.functional.mse_loss(std_x, std_x_hat)

        ce_loss = self.loss(prototypes, outputs, targets)
        total_loss = encoding_loss + ce_loss - kl_loss - proto_gauss_loss + 2*reconstruction_loss + std_loss

        self.log(f"{log}_encoding_loss", encoding_loss, prog_bar=False, on_epoch=False, on_step=True)
        self.log(f"{log}_ce_loss", ce_loss, prog_bar=False, on_epoch=False, on_step=True)
        self.log(f"{log}_kl_loss", kl_loss, prog_bar=False, on_epoch=False, on_step=True)
        self.log(f"{log}_proto_gauss_loss", proto_gauss_loss, prog_bar=False, on_epoch=True, on_step=True)
        self.log(f"{log}_rec_loss", reconstruction_loss, prog_bar=False, on_epoch=True, on_step=True)
        self.log(f"{log}_rec_std_loss", std_loss, prog_bar=False, on_epoch=True, on_step=True)
        self.log(f"{log}_total_loss", total_loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log(f"{log}_acc", self.wacc(outputs, targets), prog_bar=True, on_epoch=True, on_step=True)
        self.log(f"{log}_f1", self.wf1(outputs, targets), prog_bar=False, on_epoch=True, on_step=True)

        if log_metrics and self.n_classes > 1:
            self.log(f"{log}_ck", self.ck(outputs, targets))
            self.log(f"{log}_pr", self.pr(outputs, targets))
            self.log(f"{log}_rc", self.rc(outputs, targets))
            self.log(f"{log}_macc", self.macc(outputs, targets))
            self.log(f"{log}_mf1", self.mf1(outputs, targets))
        return total_loss
        
class NN(nn.Module):
    def __init__( self, 
        config : dict
    ):
        super(NN, self).__init__()
        
        assert 3000 % config["proto_lenght"] == 0, "The prototype lenght must be a divisor of 3000"

        self.epoch_encoder = EpochEncoder( 
            hidden_size = config["proto_lenght"],    #3000 // config["proto_lenght"], 
            attention_size = 128, 
            N = config["N"], 
            n_prototypes = config["n_prototypes"], 
            temperature = 1.0 
        )
        
        self.sequence_encoder = SequenceEncoder(
            hidden_size = self.epoch_encoder.out_size,
            nhead = 4
        )

        self.section_reconstructor = SectionReconstructor(
            N = config["N"],
            section_length = config["proto_lenght"],
            num_sections = 3000//config["proto_lenght"],
            proto_dim = self.epoch_encoder.out_size
        )
        
        self.clf = nn.Linear( self.epoch_encoder.out_size, config["n_classes"] )
        self.section_length = config["proto_lenght"]
    
    def encode( self, x ):
        # x shape : (batch_size, seq_len, n_chan, n_samp)
        batch_size, seq_len, _, _ = x.size()
        
        proto, residual, loss, x_embedding, sampled_x = self.epoch_encoder( x )    # proto shape : (batch_size, seq_len, N, hidden_size)

        reconstructed_section = self.section_reconstructor(x_embedding)
        
        # we selected N prototypes from each epoch in the sequence
        
        # sum the prototypes to get the epoch representation
        N = proto.size(2)
        p_clf = torch.sum( proto, dim=2 ) / N   # shape : (batch_size, seq_len, hidden_size)        
                
        # encode the sequence with the transformer        
        clf = self.sequence_encoder( p_clf ).reshape( batch_size*seq_len, -1 )    # shape : (batch_size * seq_len, hidden_size)
        
        clf = self.clf( clf ).reshape( batch_size, seq_len, -1 )

        #return proto, clf, loss
        return proto, clf, loss, residual, reconstructed_section, sampled_x
            
    def forward( self, x ):        
        x, y = self.encode( x )                   
        return y

class SectionReconstructor(nn.Module):
    def __init__(self,
        N : int,
        section_length: int,
        num_sections = int,
        proto_dim = int
    ):
        super(SectionReconstructor, self).__init__()
        self.N = N
        self.section_length = section_length
        self.num_sections = num_sections
        self.proto_dim = proto_dim
        self.num_components = 100

        self.amplitude_layer = nn.Sequential(
            nn.Linear(self.proto_dim, self.proto_dim*4),
            nn.ReLU(),
            nn.LayerNorm(self.proto_dim*4),
            nn.Dropout(0.4),
            nn.Linear(self.proto_dim*4, self.proto_dim*2),
            nn.ReLU(),
            nn.LayerNorm(self.proto_dim*2),
            nn.Dropout(0.5),
            nn.Linear(self.proto_dim*2, self.num_components),
            nn.Dropout(0.4)
        )
        for layer in self.amplitude_layer:
            if hasattr(layer, "bias") and layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
                layer.bias.requires_grad = False

        self.phase_layer = nn.Sequential(
            nn.Linear(self.proto_dim, self.proto_dim*4),
            nn.ReLU(),
            nn.LayerNorm(self.proto_dim*4),
            nn.Dropout(0.4),
            nn.Linear(self.proto_dim*4, self.proto_dim*2),
            nn.ReLU(),
            nn.LayerNorm(self.proto_dim*2),
            nn.Dropout(0.5),
            nn.Linear(self.proto_dim*2, self.num_components),
            nn.Dropout(0.4)
        )
        for layer in self.phase_layer:
            if hasattr(layer, "bias") and layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
                layer.bias.requires_grad = False

        self.freqs = torch.arange(0.5, self.num_components//2 + 0.5, 0.5, device="cuda")
        self.t = torch.linspace(0, self.section_length, steps = self.section_length, device = "cuda").unsqueeze(0).unsqueeze(0)

    def sinusoid_extractor(self, z):
        # z shape: [batch_size*seq_len*num_sections*n_channels, proto_dim]
        amps = self.amplitude_layer(z)
        phases = self.phase_layer(z)
        return amps, phases

    def signal_reconstruction(self, amps, phases):
        sinusoids = amps.unsqueeze(-1) * torch.sin(2 * torch.pi * self.freqs.unsqueeze(-1) * self.t + phases.unsqueeze(-1))
        signal = sinusoids.sum(dim=1)  # [-1, section_length]
        return signal

    def signal_builder(self, z):
        # z shape: [batch_size*seq_len*num_sections*n_channels, proto_dim]
        amps, phases = self.sinusoid_extractor(z)
        z_hat = self.signal_reconstruction(amps, phases)
        return z_hat
     
    def forward(self, x):
        x_hat = self.signal_builder(x.reshape(-1, self.proto_dim))
        return x_hat

class SequenceEncoder( nn.Module ):
    def __init__(self,
        hidden_size : int,
        nhead : int,
    ):
        super(SequenceEncoder, self).__init__()
        assert  hidden_size % nhead == 0, "Hidden size must be a divisor of the number of heads"
        
        self.pe = PositionalEncoding(hidden_size, 1024)
        
        # autoregressive model for the sequence trasnformer based on the transformer architecture
        self.encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size, 
            dropout=0.25,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder( self.encoder, num_layers=1 )

    def forward( self, x):
        # x shape : (batch_size, seq_len, hidden_size)
        
        # encode the sequence with positional encoding
        x = self.pe(x)
        
        # encode the sequence with the transformer
        x = self.encoder(x)
        
        return x
    

class PrototypeLayer( nn.Module ):
    def __init__( self, 
        output_size : int,  #protoype dimension
        n_prototypes : int = 1, # number of prototypes to learn
        commitment_cost : float = 0.25 # beta parameter for the VQ-VAE
        ):
        super(PrototypeLayer, self).__init__()

        self.proto_dim = output_size
        self.proto_num = n_prototypes

        self.prototypes = nn.Embedding(self.proto_num, self.proto_dim)
        self.commitment_cost = commitment_cost

    def forward( self, x ):
        # x shape : (batch_size, seq_len * N, output_size)
        x_shape = x.shape

        x = x.reshape(-1, self.proto_dim).contiguous()  # shape : (batch_size * seq_len * N, output_size)

        # Calculate distances
        dist = (torch.sum(x**2, dim=1, keepdim=True) # shape: (batch_size * seq_len * N, proto_num)
                    + torch.sum(self.prototypes.weight**2, dim=1)
                    - 2 * torch.matmul(x, self.prototypes.weight.t()))

        proto_indices = torch.argmin(dist, dim=1).unsqueeze(1) # shape : (batch_size * seq_len * N, 1)
        prototype_OHE = torch.zeros(proto_indices.shape[0], self.proto_num).to(x.device) # shape: (batch_size * seq_len * N, proto_num)
        prototype_OHE.scatter_(1, proto_indices, 1)

        proto = torch.matmul(prototype_OHE, self.prototypes.weight).view(x_shape)   #shape: (batch_size, seq_len * N, output_size)
        x = x.view(x_shape)

        e_latent_loss = F.mse_loss(proto.detach(), x)
        q_latent_loss = F.mse_loss(proto, x.detach())

        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        proto = x + (proto - x).detach()

        residuals = x - proto
        x_embedding = x

        return proto, residuals, loss, x_embedding
        
        
class EpochEncoder( nn.Module ):
    def __init__( self, 
        hidden_size : int, #size of the sections
        attention_size : int,
        N : int = 1, # number of elements to select from the epoch
        n_prototypes : int = 25, # number of elements to learn
        temperature : float = 1.0
        ):
        
        super(EpochEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.N = N
        
        # we need to extract frequency-related features from the signal
        self.conv1 = nn.Sequential( OrderedDict([
            ("conv1", nn.Conv1d(1, 32, 3, padding=1)),
            ("relu1", nn.ReLU()),
            ("maxpool1", nn.MaxPool1d(4)),
            ("batchnorm1", nn.BatchNorm1d(32)),
            ("conv2", nn.Conv1d(32, 64, 3, padding=1)),
            ("relu2", nn.ReLU()),
            ("maxpool2", nn.MaxPool1d(hidden_size // 100)),
            ("batchnorm2", nn.BatchNorm1d(64)),
            ("conv3", nn.Conv1d(64, 128, 3, padding=1)),
            ("relu3", nn.ReLU()),
            ("maxpool3", nn.MaxPool1d(4)),
            ("batchnorm3", nn.BatchNorm1d(128)),
            ("flatten", nn.Flatten()),
            ("fc", nn.Linear(128*6, 256)),
            ("dropout", nn.Dropout(0.3)),
            ("fc2", nn.Linear(256, 128))
        ]))
        
        self.out_size = self.conv1(torch.randn(1, 1,  hidden_size)).shape[1]
        
        self.sampler = HardAttentionLayer(hidden_size, attention_size, N, temperature)
        
        self.prototype = PrototypeLayer( self.out_size, n_prototypes )
        
    def forward( self, x ):
        # x shape : (batch_size, seq_len, n_chan, n_samp)
        batch_size, seq_len, n_chan, n_samp = x.size()
        assert n_samp % self.hidden_size == 0, "Hidden size must be a divisor of the number of samples"

        x = x.reshape( batch_size * seq_len, n_chan*(n_samp//self.hidden_size), -1 )

        sampled_x = self.sampler( x ) # shape : (batch_size * seq_len, N, hidden_size)

        x = sampled_x.reshape( batch_size * seq_len * self.N, 1,  -1 )
        
        x = self.conv1( x ) # shape : (batch_size * seq_len * self.N, out_size)

        x = x.reshape( batch_size, seq_len*self.N, self.out_size )
        
        proto, residual, loss, x_embedding = self.prototype( x )
        
        proto = proto.reshape( batch_size, seq_len, self.N, self.out_size )
        residual = residual.reshape( batch_size, seq_len, self.N, self.out_size )
        
        return proto, residual, loss, x_embedding, sampled_x

class HardAttentionLayer(nn.Module):
    def __init__(self, 
            hidden_size : int,
            attention_size : int, 
            N : int = 1, # number of elements to select
            temperature : float = 1.0,
            encoding : nn.Module = None
        ):
        super(HardAttentionLayer, self).__init__()
        
        self.temperature = temperature
        
        self.pe = PositionalEncoding(hidden_size, 100)
        
        self.N = N

        self.Q = nn.Linear(hidden_size, attention_size * N, bias = False)
        self.K = nn.Linear(hidden_size, attention_size * N, bias = False)

    def forward(self, x):
        # x shape : (batch_size * seq_len, section_num (n_chan*(n_samp//hidden_size)), hidden_size)
        batch_size, section_num, hidden_size = x.size()

        # encode the sequence with positional encoding
        pos_emb = self.pe(x)

        # calculate the query and key -> output shape: (batch_size, section_num, attention_size * N)
        Q = self.Q(pos_emb)
        K = self.K(pos_emb)
        
        Q = Q.reshape( batch_size, section_num, self.N, -1 ).transpose(1, 2)
        K = K.reshape( batch_size, section_num, self.N, -1 ).transpose(1, 2)
        
        attention = torch.einsum( "bnsh,bnth -> bnst", Q, K ) / ( hidden_size ** (1/2) )
        attention = torch.sum(attention, dim=-1) / section_num

        # attention shape : (batch_size * N, section_num)
        logits = attention.reshape( batch_size * self.N, section_num )                
        # apply the Gumbel-Softmax trick to select the N most important elements
        alphas = torch.nn.functional.gumbel_softmax(logits, tau=self.temperature, hard=True)
        alphas = alphas.reshape( batch_size, self.N, section_num )
        
        # select N elements from the sequence x using alphas
        x = torch.einsum( "bns, bsh -> bnh", alphas, x )
        
        return x

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
