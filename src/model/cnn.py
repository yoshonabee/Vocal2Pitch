import torch
import torch.nn as nn
import torch.nn.functional as F

import json

class CNN(nn.Module):
    def __init__(self, in_channel=None, output_dim=None, model_config=None):
        super(CNN, self).__init__()

        self.model_config = model_config

        self._down_sampling_rate = 1

        self.cnn = None
        self.classifier = None
        self._parse_model_config()

    def _parse_model_config(self):
        cnn_layers = []
        classifier_layers = []
        config = json.load(open(self.model_config))

        for layer in config['feature_extractor']:
            cnn_layers.append(getattr(nn, layer.pop("name"))(**layer))

            if "stride" in layer:
                self._down_sampling_rate *= layer['stride']

        print(cnn_layers)

        for layer in config['classifier']:
            classifier_layers.append(getattr(nn, layer.pop("name"))(**layer))

        self.cnn = nn.Sequential(*cnn_layers)
        self.classifier = nn.Sequential(*classifier_layers)

    @property
    def down_sampling_rate(self):
        return self._down_sampling_rate
    
    def forward(self, x):
        # print(x.shape)
        x = x.unsqueeze(1) # (B, L) -> (B, 1, L)
        x = self.cnn(x) # (B, 1, L) -> (B, C, L')
        x = x.transpose(1, 2) # (B, C, L') -> (B, L', C)
        out = self.classifier(x) # (B, L', C) -> (B, L', 2)

        return out
