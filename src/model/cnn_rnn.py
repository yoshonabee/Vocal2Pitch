import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_RNN(nn.Module):
    def __init__(self, layers_config=None):
        super(CNN_RNN, self).__init__()

        layers = [
            getattr(nn, layer.pop('name'))(**layer)
            for layer in layers_config
        ]

        #layers.append(nn.Dropout(0.5))

        self.cnn = nn.Sequential(*layers)
        self.lstm = nn.LSTM(self.channels[-1], 256, batch_first=True, bidirectional=True, num_layers=3)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self._down_sampling_factor = 1
        for layer in layers:
            if "stride" in layer:
                self._down_sampling_factor *= layer['stride']

    @property
    def down_sampling_factor(self):
        return self._down_sampling_factor
    
    def forward(self, x):
        x = x.unsqueeze(1) # (B, L) -> (B, 1, L)
        x = self.cnn(x) # (B, 1, L) -> (B, C, L // down_sampling_factor (L'))
        x = x.transpose(1, 2) # (B, C, L') -> (B, L', C)

        x, _ = self.lstm(x) # (B, L', C) -> (B, L', 2, 128)
        x = x.view(x.size(0), x.size(1), -1)
        out = self.classifier(x) # (B, L', 128) -> (B, L', 1)

        return out
