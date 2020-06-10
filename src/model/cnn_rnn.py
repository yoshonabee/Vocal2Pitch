import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_RNN(nn.Module):
    def __init__(self, layers_config=None):
        super(CNN_RNN, self).__init__()

        layers = []
        cnn_output_dim = 1

        for layer in layers_config:
            if "stride" in layer:

            name = layer.pop("name")
            if "conv" in name.lower():
                cnn_output_dim = layer['out_channels']
            layers.append(getattr(nn, name)(**layer))

        #layers.append(nn.Dropout(0.5))

        self.cnn = nn.Sequential(*layers)

        self.lstm = nn.LSTM(cnn_output_dim, 64, batch_first=True, bidirectional=True, num_layers=3)

        self.classifier = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.transpose(1, 2) # (B, L, C) -> (B, C, L)
        x = self.cnn(x) # (B, C, L) -> (B, C', L)
        x = x.transpose(1, 2) # (B, C', L) -> (B, L, C')

        x, _ = self.lstm(x) # (B, L, C') -> (B, L, 2, 64)
        x = x.view(x.size(0), x.size(1), -1)
        out = self.classifier(x) # (B, L, 128) -> (B, L, 1)

        return out