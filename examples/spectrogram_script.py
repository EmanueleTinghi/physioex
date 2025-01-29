from typing import List, Callable
from physioex.data import PhysioExDataset
from physioex.data.datamodule import PhysioExDataModule
from physioex.train.networks import config as network_config
from physioex.train.bin.parser import deep_update
import importlib
import numpy as np
from physioex.train.utils.train import train

class EmanueleDataset( PhysioExDataset ):
    def __init__(
        self,
        datasets: List[str],
        data_folder: str,
        preprocessing: str = "raw",
        selected_channels: List[int] = ["EEG"],
        sequence_length: int = 21,
        target_transform: Callable = None,
        hpc: bool = False,
        indexed_channels: List[int] = ["EEG", "EOG", "EMG", "ECG"],
        task: str = "sleep",
    ):
        super().__init__(
            datasets = datasets,
            data_folder = data_folder,
            preprocessing = preprocessing,
            selected_channels = selected_channels,
            sequence_length = sequence_length,
            target_transform = target_transform,
            hpc = hpc,
            indexed_channels = indexed_channels,
            task = task,
        )

        self.emanuele_data = PhysioExDataset(
            datasets = datasets,
            data_folder = data_folder,
            preprocessing = "emanuele",
            selected_channels = selected_channels,
            sequence_length = sequence_length,
            target_transform = target_transform,
            hpc = hpc,
            indexed_channels = indexed_channels,
            task = task,
        )
        return

    def __getitem__(self, idx):
        X, _ = super().__getitem__(idx)
        '''print("self.L:", self.L)
        X = X.reshape( self.L, len(self.channels_index), 5, 600)'''
        y, _ = self.emanuele_data.__getitem__(idx)      
        print("X shape:", X.shape)
        print("y shape:", y.shape)  
        '''
        X = X.permute(0, 2, 1, 3)
        y = y.permute(0, 2, 1, 3, 4)
        
        X = X.reshape( self.L*5, len(self.channels_index), 600)
        y = y.reshape( self.L*5, len(self.channels_index), 5, 19)'''
        
        return X, y
    
emanuele_data = EmanueleDataset(
    datasets = ["mass"],
    data_folder = "/mnt/guido-data/",
    preprocessing = "emanuele",
    selected_channels = ["EEG", "EOG", "EMG"],
    sequence_length = 21,
    indexed_channels = ["EEG", "EOG", "EMG"],
)

datamodule = PhysioExDataModule(
    datasets = emanuele_data,
    batch_size = 32,
)

default_config = network_config["default"].copy()
config = network_config[ "protosleepnet_reconstructor_mass"]

deep_update(default_config, config)
config = default_config

config["model_kwargs"]["in_channels"] = 3
config["model_kwargs"]["sequence_length"] = 21

module, class_name = config["model"].split(":")
config["model"] = getattr(importlib.import_module(module), class_name)

train(
    datasets = datamodule,
    model_class = config["model"],
    model_config = config["model_kwargs"],
    model = None,
    monitor = "val_loss",
)