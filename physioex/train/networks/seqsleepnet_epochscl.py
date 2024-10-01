from physioex.train.networks import seqsleepnet


class SeqSleepNetEpochScl(seqsleepnet.SeqSleepNet):
    def __init__(self, module_config):
        super(SeqSleepNetEpochScl, self).__init__(module_config, NetEpochScl(module_config))


class NetEpochScl(seqsleepnet.Net):
    def __init__(self, module_config=seqsleepnet.module_config):
        super().__init__(module_config)

    def encode(self, x):
        batch, L, nchan, T, F = x.size()

        x = x.reshape(-1, nchan, T, F)
        epoch_embeddings = self.epoch_encoder(x)
        epoch_embeddings = epoch_embeddings.reshape(batch, L, -1)

        sequence_embeddings = self.sequence_encoder.encode(epoch_embeddings)
        sequence_output = self.sequence_encoder.clf(sequence_embeddings)

        return epoch_embeddings, sequence_output