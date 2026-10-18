"""PyTorch equivalents of the sequential models used by MininetFed."""
import torch
from torch import nn
from torch.nn import functional as F


class MininetCNN(nn.Module):
    def __init__(self, dataset, num_classes=10):
        super().__init__()
        dataset = dataset.lower().replace('-', '')
        if dataset == 'mnist':
            channels, pools, dense, spatial = [64, 128, 128], {0, 1, 2}, [], 28
        elif dataset == 'cifar10':
            channels, pools, dense, spatial = [64, 128, 256, 256, 512, 512, 512, 512], {0, 1, 3, 5, 7}, [4096, 4096], 32
        elif dataset == 'cinic10':
            channels, pools, dense, spatial = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512], {1, 3, 6, 9, 12}, [4096, 4096], 32
        else:
            raise ValueError(f'Unsupported reference dataset: {dataset}')
        self.augment = dataset != 'mnist'
        layers, previous = [], 1 if dataset == 'mnist' else 3
        for i, width in enumerate(channels):
            layers.extend([nn.Conv2d(previous, width, 3, padding=1), nn.ReLU()])
            if i in pools:
                layers.append(nn.MaxPool2d(2))
                spatial //= 2
            previous = width
        layers.append(nn.Flatten())
        previous *= spatial * spatial
        for width in dense:
            layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(0.5)])
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.layers = nn.Sequential(*layers)
        for layer in self.layers:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def augment_inputs(self, x):
        if self.training and self.augment:
            flip = torch.rand(len(x), 1, 1, 1, device=x.device) < 0.5
            x = torch.where(flip, x.flip(-1), x)
            theta = torch.eye(2, 3, device=x.device).expand(len(x), -1, -1).clone()
            theta[:, :, 2] = torch.empty(len(x), 2, device=x.device).uniform_(-0.2, 0.2)
            grid = F.affine_grid(theta, x.shape, align_corners=False)
            x = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        return x

    def forward(self, x):
        return self.layers(self.augment_inputs(x))
