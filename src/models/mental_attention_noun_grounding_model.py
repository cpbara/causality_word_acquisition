import torch
from torch import nn
from src.models.attention import Attention
    
class MentalAttentionModel(nn.Module):
    def __init__(self):
        super(MentalAttentionModel, self).__init__()
        self.att = Attention(4096, 768)
    def forward(self, x,y):
        return self.att(x,y)
    
class MentalAttentionNoundGroundingModel(nn.Module):
    def __init__(self):
        super(MentalAttentionNoundGroundingModel, self).__init__()

        linearLayer = lambda i, o: nn.Sequential(
            nn.Linear(i, o),
            nn.BatchNorm1d(o),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.embedder = nn.Sequential(
            linearLayer(25088, 8192),
            linearLayer( 8192, 4096)
        )
        self.linear_bbox = nn.Sequential(
            linearLayer( 4696, 2048),
            nn.Linear( 2048, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = torch.cat([self.embedder(x[0]), x[1], x[2]],-1)
        return self.linear_bbox(output)