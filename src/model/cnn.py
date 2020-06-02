import torch
import torch.nn as nn
import torch.nn.functional as F

import json

class CNN(nn.Module):
    def __init__(self, in_channel=None, output_dim=None, model_config=None):
        super(CNN, self).__init__()

        layers = [
            nn.Conv2d(1, 10, (3, 7), 1),
            nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Conv2d(10, 20, (3, 3), 1),
            nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Dropout(0.3),
            nn.Conv2d(20, 60, (3, 3), 1, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(60),
            nn.Conv2d(60, 60, (3, 3), 1, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(60),
            nn.Conv2d(60, 60, (3, 3), 1, (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(60),
            nn.Dropout(0.3)
        ]

        self.net = nn.Sequential(*layers)

        layers = [
            nn.Linear(3360, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ]

        self.out = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.out(x)
