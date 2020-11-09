import torch
from torch import nn
from src.models.image_model import ImageModel

class RGBVideoActionClassifier(nn.Module):
    def __init__(self):
        super(RGBVideoActionClassifier, self).__init__()

        linearLayer = lambda i, o: nn.Sequential(
            nn.Linear(i, o, bias=True),
            nn.BatchNorm1d(o),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self. img_model = ImageModel()

        self.linear = nn.Sequential(
            linearLayer( 8192, 2048),
            linearLayer( 2048, 43)
        )

    def forward(self, x):
        fun1 = lambda d: torch.mean(self.img_model(d),axis=0)
        fun2 = lambda x: torch.stack([fun1(x[i]) for i in range(len(x))])
        return self.linear(torch.cat([fun2(x[j]) for j in range(len(x))],axis=-1))