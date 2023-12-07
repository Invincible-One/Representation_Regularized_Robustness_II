import torch.nn as nn


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, config, output_dim=2):
        super(FullyConnectedNetwork, self).__init__()

        layers_dict = dict()
        prev_dim = input_dim

        for idx, n_neurons in enumerate(config):
            layers_dict[f"fc{idx + 1}"] = nn.Linear(prev_dim, n_neurons)
            layers_dict[f"relu{idx + 1}"] = nn.ReLU()
            prev_dim = n_neurons

        self.layers = nn.ModuleDict(layers_dict)
        self.classifier = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        x = self.classifer(x)
        
        return x


class VGG_like(nn.Module):
    def __init__(self, input_dim, config, output_dim=2):
        super(VGG_like, self).__init__()
        self.input_dim = input_dim
        self.config = config

        self.features = self._make_layer()
        self.classifier = nn.Linear(self.config[-1], output_dim)

    def _make_layer(self):
        layers_dict = dict()
        in_channels = self.input_dim[0]
        
        idx = 1
        for x in self.config[:-1]:
            if x == 'M':
                layers_dict[f"maxpool{idx}"] = nn.MaxPool2d(kernel_size=2, stride=2)
            else:
                layers_dict[f"conv{idx}"] = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                layers_dict[f"relu{idx}"] = nn.ReLU()
                in_channels = x
                idx += 1

        return nn.ModuleDict(layers_dict)

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

