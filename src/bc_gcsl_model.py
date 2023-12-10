import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DynamicEncoder(nn.Module):
    def __init__(self, list_of_input_sizes):
        super(DynamicEncoder, self).__init__()
        for i, input_size in enumerate(list_of_input_sizes):
            setattr(self, f"task_{i}", nn.Linear(input_size, 128))
        self.common_fc = nn.Linear(128, 128)

    def forward(self, x, task_id):
        x = getattr(self, f"task_{task_id}")(x)
        return torch.relu(self.common_fc(x))

class CustomBCModel(nn.Module):
    def __init__(self, encoder, action_dim):
        super(CustomBCModel, self).__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.output = nn.Linear(1024, action_dim)

    def forward(self, x, task_id):
        x = self.encoder(x, task_id)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output(x)