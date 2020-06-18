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

        self.pitch_criterion = nn.L1Loss().to(self.device)

    def forward(self, model, data):
        x, onset_y, pitch_y = data
        x = x.to(self.device)
        onset_y = onset_y.to(self.device)
        pitch_y = pitch_y.to(self.device)

        onset_out, pitch_out = model(x)

        resampled_onset_out, resampled_onset_y = self.resample(onset_out, onset_y)
        onset_out, onset_y = onset_out.view(-1), onset_y.view(-1)
        pitch_out, pitch_y = pitch_out.view(-1), pitch_y.view(-1)

        onset_loss = self.criterion(resampled_onset_out, resampled_onset_y)
        pitch_loss = self.pitch_criterion(pitch_out, pitch_y)

        loss = onset_loss + pitch_loss

        return loss, onset_out, onset_y, onset_loss, pitch_loss

    def resample(self, pred, y):
        #r_pred = pred.view(-1, pred.size(-1))

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

        

