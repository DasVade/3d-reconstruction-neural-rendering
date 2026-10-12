import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def positional_encoding(x, num_frequencies=6, incl_input=True):
    results = []
    if incl_input:
        results.append(x)

    for i in range(num_frequencies):
        freq = 2.0 ** i * np.pi
        results.append(torch.sin(freq * x))
        results.append(torch.cos(freq * x))

    return torch.cat(results, dim=-1)

class model_2d(nn.Module):
    def __init__(self, filter_size=128, num_frequencies=6):
        super().__init__()
        input_dim = (num_frequencies * 2 + 1) * 2
        self.layer_in = nn.Linear(input_dim, filter_size)
        self.act1 = nn.ReLU()
        self.layer = nn.Linear(filter_size, filter_size)
        self.act2 = nn.ReLU()
        self.layer_out = nn.Linear(filter_size, 3)
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_in(x)
        x = self.act1(x)
        x = self.layer(x)
        x = self.act2(x)
        x = self.layer_out(x)
        x = self.act_out(x)
        return x

def normalize_coord(height, width, num_frequencies=6):
    x = torch.linspace(0, 1, width)
    y = torch.linspace(0, 1, height)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    coordinates = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
    embedded_coordinates = positional_encoding(
        coordinates, num_frequencies=num_frequencies, incl_input=True
    )
    return embedded_coordinates
