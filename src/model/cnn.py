import torch
import torch.nn as nn
import torch.nn.functional as F

import json

class CNN(nn.Module):
    def __init__(self, in_channel=None, output_dim=None, model_config=None):
        super(CNN, self).__init__()

        layers = [
            nn.Conv2d(1, 32, (3, 7), 1),
            nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Conv2d(32, 64, (3, 3), 1),
            nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, (3, 3), 1, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), 1, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), 1, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3)
        ]

        self.net = nn.Sequential(*layers)

        layers = [
            nn.Linear(7168, 1),
            nn.Sigmoid()
        ]

        self.out = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.out(x)
