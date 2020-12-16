from torch import nn
import torch
from src.models.attention import Attention

class ActionClassifier(nn.Module):
    def __init__(self, use_attention=False):
        super(ActionClassifier, self).__init__()
        self.use_attention = use_attention
        linearLayer = lambda i, o: nn.Sequential(
            nn.Linear(i, o),
            nn.BatchNorm1d(o),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        self.pre_linear = nn.Sequential(
            linearLayer(8192, 2048),
            linearLayer(2048,  512)
        )
        self.post_linear  = nn.Sequential(
            linearLayer(8192, 2048),
            linearLayer(2048,  512)
        )
        self.linear = nn.Sequential(
            linearLayer(1024, 512),
            nn.Linear( 512,   43),
            nn.Softmax(-1),
        )
        
    def forward(self, x):
        return self.linear(torch.cat([
            self.pre_linear(torch.cat([x[0][0],x[0][1]],axis=-1)), \
            self.post_linear(torch.cat([x[1][0],x[1][1]],axis=-1))],
            axis=-1))