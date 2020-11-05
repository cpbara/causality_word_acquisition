from torch import nn
import torch

class Attention(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Attention, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(x_dim + y_dim, x_dim),
            nn.BatchNorm1d(x_dim),
            nn.ReLU(),
            nn.Linear(x_dim , x_dim),
            nn.Sigmoid()
        )
    def forward(self, x, y):
        return torch.mul(x, self.linear(torch.cat((x, y), -1)))
    