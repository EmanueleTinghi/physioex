#!/usr/bin/env python3
from physioex.data.datamodule import PhysioExDataModule
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

class SectionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SectionEncoder, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, section):
        # section is of shape (1, 5, 129), where 5 is the sequence length and 129 is the embedding dimension
        section = section.squeeze(1)  # Remove the extra dimension to make it (batch_size, 5, 129)
        
        # Pass through the transformer encoder
        encoded_section = self.transformer_encoder(section)  # Expects input of shape (batch_size, 5, 129)

        # Apply linear transformation to the encoded output
        return self.linear(encoded_section)

class BERT_EEG(nn.Module):
    def __init__(self, section_encoder, bert_model, contrastive_margin=1.0):
        super(BERT_EEG, self).__init__()
        self.section_encoder = section_encoder  # Transformer to encode individual EEG sections
        self.bert = bert_model                  # BERT model to process encodings
        self.contrastive_margin = contrastive_margin
        self.loss_fn = nn.TripletMarginLoss(margin=self.contrastive_margin)
    
    def section_epoch(self, epoch):
        # Eliminate the last row (remove the 129th row) to get shape (1, 29, 128)
        epoch = epoch[:, :, :-1]

        # Split the epoch (1, 29, 128) into sections of (1, 5, 128)
        sections = [epoch[:, i:i + 5, :] for i in range(0, epoch.size(1) - 5 + 1, 5)]

        # If the last section has fewer than 5 timesteps, pad it with zeros
        if epoch.size(0) % 5 != 0:
            last_section = epoch[:, -(epoch.size(1) % 5):, :]  # Assuming epoch is of shape (batch_size, 29, 129)
            padding = torch.zeros((last_section.size(0), 5 - last_section.size(1), last_section.size(2))).to(epoch.device)

            # Concatenate the last_section and the padding
            last_section = torch.cat((last_section, padding), dim=1)

            sections.append(last_section)
        
        return sections
    
    def forward(self, epoch, next_epoch):
        # Section the epochs
        epoch_sections = self.section_epoch(epoch)
        next_epoch_sections = self.section_epoch(next_epoch)

        # Encode each section using the section encoder
        epoch_encoding = (torch.stack([self.section_encoder(section) for section in epoch_sections])).view(6, 5, 64)
        next_epoch_encoding = torch.stack([self.section_encoder(section) for section in next_epoch_sections]).view(6, 5, 64)

        # Pass the epoch encodings through BERT
        bert_output = self.bert(inputs_embeds=epoch_encoding)
        bert_output = bert_output.last_hidden_state

        return bert_output, next_epoch_encoding

# Example training loop
def train_epoch(model, optimizer, epochs, loss_function, data_loader):
    for epoch_idx in range(epochs):
        for inputs, targets in data_loader:
            flattened_inputs = inputs.view(-1, 1, 29, 129)
            epochs_iterator = iter(flattened_inputs)
            next_epoch_iterator = iter(flattened_inputs)

            for epoch in epochs_iterator:
                optimizer.zero_grad()
                next_epoch = next(next_epoch_iterator)  # Get the next element from the iterator
                if next_epoch is None:
                    break

                predicted_next_epoch, next_epoch = model(epoch, next_epoch)

                # Compute loss based on the differences between predicted next epoch and actual next epoch     
                loss = loss_function(predicted_next_epoch, next_epoch)

                # Backpropagation
                loss.backward()
                optimizer.step()
        
        print(f"Epoch {epoch_idx + 1}/{epochs}, Loss: {loss.item()}")

def main():
    print("MAIN-START")
    datamodule = PhysioExDataModule(
        datasets=["hmc"],
        versions=None,
        batch_size=15,
        selected_channels=["EEG"],
        sequence_length=10,
        data_folder="/mnt/guido-data/",
        preprocessing = "xsleepnet",
        target_transform= None
    )

    data_loader = datamodule.train_dataloader()
    # Create the model
    input_dim = 128  # Example dimension, adjust based on your EEG data
    hidden_dim = 64
    section_encoder = SectionEncoder(input_dim, hidden_dim)

    # BERT configuration
    bert_config = BertConfig(hidden_size=hidden_dim, num_attention_heads=4, num_hidden_layers=2)
    bert_model = BertModel(bert_config)

    # Initialize the BERT-EEG model
    model = BERT_EEG(section_encoder, bert_model)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Example data_loader creation
    # Assuming data_loader returns (current_epoch, next_epoch, negative_epoch)
    # current_epoch, next_epoch, and negative_epoch ar§e torch tensors containing EEG sections
    # Train the model
    epochs = 10

    train_epoch(model, optimizer, epochs, loss_function=torch.nn.MSELoss(), data_loader=data_loader)

if __name__ == "__main__":
    main()
