from physioex.train.networks import seqsleepnet
import torch
import torch.nn as nn

class SeqSleepNetEpochSequenceConcScl(seqsleepnet.SeqSleepNet):
    def __init__(self, module_config=seqsleepnet.module_config):
        print("Init conc")
        super(SeqSleepNetEpochSequenceConcScl, self).__init__(module_config, NetEpochSequenceConcScl(module_config))

class NetEpochSequenceConcScl(seqsleepnet.Net):
    def __init__(self, module_config=seqsleepnet.module_config):
        super().__init__(module_config)
        self.sequence_encoder = SequenceEncoderConc(module_config)


    def encode(self, x):
        batch, L, nchan, T, F = x.size()
        x_epoch = x.reshape(-1, nchan, T, F)
        x_epoch = self.epoch_encoder(x_epoch)
        x_epoch = x_epoch.reshape(batch, L, -1)

        x_sequence = self.sequence_encoder.encode(x_epoch)

        x=torch.cat((x_epoch,x_sequence), dim=-1)
        y = self.sequence_encoder.clf(x)
        return x, y
    
class SequenceEncoderConc(seqsleepnet.SequenceEncoder):
    def __init__(self, module_config):
        super().__init__(module_config)
                      
        self.LSTM = nn.GRU(
            input_size=2 * module_config["seqnhidden1"],
            hidden_size=module_config["seqnhidden2"],
            num_layers=module_config["seqnlayer2"],
            batch_first=True,
            bidirectional=True,
        )  

        self.clf = nn.Linear(
            module_config["seqnhidden2"] * 4, module_config["n_classes"]
        )