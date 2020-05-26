import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, channels=None, kernel_size=None, strides=None, dropout=None, layers_config=None):
        super(CNN, self).__init__()

        if dropout is None:
            dropout = 0

        self.dropout = dropout

        if layers_config is None:
            
            self.channels = channels
            self.strides = strides

            if isinstance(kernel_size, int):
                self.kernel_size = [kernel_size for _ in range(len(self.strides))]
            else:
                self.kernel_size = kernel_size

            try:
                assert len(self.kernel_size) == len(self.strides)
                assert len(self.channels) == len(self.strides)
            except Exception as e:
                raise ValueError("length of kernel_size, channels, strides must all be the same")
        else:
            self.channels = [layer['channels'] for layer in layers_config]
            self.kernel_size = [layer['kernel_size'] for layer in layers_config]
            self.strides = [layer['strides'] for layer in layers_config]

        layers = [
            nn.Conv1d(1, self.channels[0], self.kernel_size[0], self.strides[0], (self.kernel_size[0] - 1) // 2, bias=False),
            nn.ReLU()
        ]

        for i in range(1, len(self.strides)):
            paddings = (self.kernel_size[i] - 1) // 2

            layers.extend([
                nn.BatchNorm1d(self.channels[i - 1]),
                nn.Conv1d(self.channels[i - 1], self.channels[i], self.kernel_size[i], self.strides[i], paddings, bias=False),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])

        self.cnn = nn.Sequential(*layers)
        self.classifier = nn.Linear(self.channels[-1], 2)

    @property
    def down_sampling_factor(self):
        down_sampling_factor = 1

        for stride in self.strides:
            down_sampling_factor *= stride

        return down_sampling_factor
    
    def forward(self, x):
        x = x.unsqueeze(1) # (B, L) -> (B, 1, L)
        x = self.cnn(x) # (B, 1, L) -> (B, C, L // down_sampling_factor (L'))
        x = x.transpose(1, 2) # (B, C, L') -> (B, L', C)
        out = self.classifier(x) # (B, L', C) -> (B, L', 2)

        return out
