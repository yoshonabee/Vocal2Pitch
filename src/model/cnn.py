import torch
import torch.nn as nn
import torch.nn.functional as F

import json

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        layers = [
            nn.Conv2d(1, 10, (3, 7), 1),
            nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Conv2d(10, 20, (3, 3), 1),
            nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Dropout(0.3),
            nn.Conv2d(20, 60, (3, 3), 1, (1, 1))
        ]

        self.cnns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(),
                    nn.BatchNorm2d(60),
                    nn.Conv2d(60, 60, (3, 3), 1, (1, 1)),
                )
                for i in range(3)
            ]
        )

        self.before_out = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(60),
            nn.Dropout(0.3)
        )

        self.net = nn.Sequential(*layers)

        self.out = nn.Sequential(
            nn.Linear(16560, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.tanh()
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x)
        for layer in self.cnns:
            x = x + layer(x)
        x = self.before_out(x)
        x = x.view(x.size(0), -1)
        print(x.size())
        return self.out(x)
