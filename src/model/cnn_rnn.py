import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_RNN(nn.Module):
    def __init__(self, channels=None, kernel_size=None, strides=None, dropout=None, layers_config=None):
        super(CNN_RNN, self).__init__()

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
            ])
        #layers.append(nn.Dropout(0.5))
        self.cnn = nn.Sequential(*layers)
        self.lstm = nn.LSTM(self.channels[-1] + 22, 512, batch_first=True, bidirectional=True, num_layers=3)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    @property
    def down_sampling_factor(self):
        down_sampling_factor = 1

        for stride in self.strides:
            down_sampling_factor *= stride

        return down_sampling_factor
    
    def forward(self, x, feature):
        x = x.unsqueeze(1) # (B, L) -> (B, 1, L)
        x = self.cnn(x) # (B, 1, L) -> (B, C, L // down_sampling_factor (L'))
        x = x.transpose(1, 2) # (B, C, L') -> (B, L', C)
        x = torch.cat([x, feature], -1) # (B, L', C) -> (B, L', C')

        x, _ = self.lstm(x) # (B, L', C') -> (B, L', 2, 128)
        x = x.view(x.size(0), x.size(1), -1)
        out = self.classifier(x) # (B, L', 128) -> (B, L', 1)

        return out
