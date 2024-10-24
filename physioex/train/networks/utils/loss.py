from abc import ABC, abstractmethod
from typing import Dict

import torch
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ClassWeightedReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from torch import nn


class PhysioExLoss(ABC):

    @abstractmethod
    def forward(self, emb, preds, targets):
        pass


class SimilarityCombinedLoss(nn.Module, PhysioExLoss):
    def __init__(self):
        super(SimilarityCombinedLoss, self).__init__()
        self.miner = miners.MultiSimilarityMiner()
        weights = params.get("class_weights") if params is not None else torch.ones(5) 
        weights = weights if weights is not None else torch.ones(5)  
        print(f"weights: {weights}")
        self.contr_loss = losses.TripletMarginLoss(
            distance=CosineSimilarity(),
            reducer=ClassWeightedReducer(weights),
            embedding_regularizer=LpRegularizer(),
        )

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, emb, preds, targets):
        loss = self.ce_loss(preds, targets)
        hard_pairs = self.miner(emb, targets)
        
        contr_loss_value = self.contr_loss(emb, targets, hard_pairs)
        
        return loss + contr_loss_value

class CrossEntropyLoss(nn.Module, PhysioExLoss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, emb, preds, targets):
        return self.ce_loss(preds, targets)


class HuberLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()

        self.loss = nn.HuberLoss(delta=5)

    def forward(self, emb, preds, targets):
        return self.loss(preds, targets) / 112.5


class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()

        # mse
        self.loss = nn.MSELoss()

    def forward(self, emb, preds, targets):
        return self.loss(preds, targets)


config = {"cel": CrossEntropyLoss, "scl": SimilarityCombinedLoss, "reg": RegressionLoss}
