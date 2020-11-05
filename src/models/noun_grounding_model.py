from torch import nn
from src.models.attention import Attention
    
class MentalAttentionModel(nn.Module):
    def __init__(self):
        super(MentalAttentionModel, self).__init__()
        self.att = Attention(4096, 768)
    def forward(self, x,y):
        return self.att(x,y)
    
class NoundGroundingModel(nn.Module):
    def __init__(self):
        super(NoundGroundingModel, self).__init__()

        linearLayer = lambda i, o: nn.Sequential(
            nn.Linear(i, o),
            nn.BatchNorm1d(o),
            nn.ReLU(),
            # nn.Dropout(0.2),
        )

        self.mental_attention = MentalAttentionModel()
        self.att = Attention(4096,300)
        self.linear_bbox = nn.Sequential(
            linearLayer( 4096, 2048),
            nn.Linear( 2048, 4),
            nn.Sigmoid()
        )
        self.linear_importance = nn.Sequential(
            linearLayer( 4096, 2048),
            nn.Linear( 2048, 2),
            nn.Softmax(-1)
        )

    def forward(self, x):
        output = self.att(self.mental_attention(x[0], x[1]), x[2])
        return self.linear_bbox(output), self.linear_importance(output)