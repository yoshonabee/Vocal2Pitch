import torch
import torch.nn as nn

import random

from pytorch_trainer.criterion import DefaultCriterion

class ResampleCriterion(DefaultCriterion):
    CRITERIONS = {
        "crossentropy": nn.CrossEntropyLoss,
        'bceloss': nn.BCELoss
    }

    def __init__(self, inbalance_ratio, *args, **kwargs):
        super(ResampleCriterion, self).__init__(*args, **kwargs)
        self.inbalance_ratio = inbalance_ratio

    def forward(self, model, data):
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device).float()

        pred = model(x)

        resampled_pred, resampled_y = self.resample(pred, y)
        loss = self.criterion(resampled_pred, resampled_y)

        return loss, pred, y

    def resample(self, pred, y):
        r_pred = pred.view(-1)
        r_y = y.view(-1)

        if self.inbalance_ratio <= 0:
            return r_pred, r_y

        positive_indices = (r_y == 1).nonzero().view(-1).tolist()
        negative_indices = (r_y == 0).nonzero().view(-1).tolist()

        if len(positive_indices) * self.inbalance_ratio > len(negative_indices):
            if len(negative_indices) * self.inbalance_ratio > len(positive_indices):
                return r_pred, r_y
            else:
                n = int(len(negative_indices) * self.inbalance_ratio)

                positive_indices = random.sample(positive_indices, n)

                all_indices = positive_indices + negative_indices
                
                return r_pred[all_indices], r_y[all_indices]
        else:
            n = int(len(positive_indices) * self.inbalance_ratio)

            negative_indices = random.sample(negative_indices, n)

            all_indices = positive_indices + negative_indices
            
            return r_pred[all_indices], r_y[all_indices]

        

