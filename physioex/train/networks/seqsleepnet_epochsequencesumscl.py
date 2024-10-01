from physioex.train.networks import seqsleepnet

class SeqSleepNetEpochSequenceSumScl(seqsleepnet.SeqSleepNet):
    def __init__(self, module_config=seqsleepnet.module_config):
        print("Init sum")
        super(SeqSleepNetEpochSequenceSumScl, self).__init__(module_config, NetEpochSequenceSumScl(module_config))

class NetEpochSequenceSumScl(seqsleepnet.Net):
    def __init__(self, module_config=seqsleepnet.module_config):
        super().__init__(module_config)

    def encode(self, x):
        batch, L, nchan, T, F = x.size()
        x_epoch = x.reshape(-1, nchan, T, F)
        x_epoch = self.epoch_encoder(x_epoch)
        x_epoch = x_epoch.reshape(batch, L, -1)

        x_sequence = self.sequence_encoder.encode(x_epoch)
        x=x_epoch+x_sequence

        y = self.sequence_encoder.clf(x)
        return x, y