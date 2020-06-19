import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerEncoder, TransformerDecoder

class CNN_Transformer(nn.Module):
    def __init__(self):
        super(CNN_Transformer, self).__init__()

        self.onset_cnn = nn.Sequential(
            nn.Conv1d(1, 64, 128, padding=31, stride=64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 16, padding=7, stride=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 8, padding=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256)
        )

        self.transformer_encoder = TransformerEncoder(blocks=3, model_dim=256, q_dim=32, h=8, dff=1024)

        self.onset_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self._down_sampling_factor = 64 * 4 * 2

    @property
    def down_sampling_factor(self):
        return self._down_sampling_factor
    
    def forward(self, x):
        x = x.unsqueeze(1) # (B, L) -> (B, 1, L)
        onset = self.onset_cnn(x) # (B, 1, L) -> (B, C, L // down_sampling_factor (L'))
        onset = onset.transpose(1, 2) # (B, C, L') -> (B, L', C)
        onset = self.transformer_encoder(onset) # (B, L', C) -> (B, L', 128)

        onset = onset.view(onset.size(0), onset.size(1), -1)
        onset_out = self.onset_classifier(onset) # (B, L', 128) -> (B, L', 1)

        pitch_out = self.pitch_predictor(onset)

        return onset_out, pitch_out
