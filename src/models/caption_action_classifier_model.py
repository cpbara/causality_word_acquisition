import torch
from torch import nn

class CaptionActionClassifier(nn.Module):
    def __init__(self, transformer):
        super(CaptionActionClassifier, self).__init__()

        self.transformer = transformer
        linearLayer = lambda i, o: nn.Sequential(
            nn.Linear(i, o),
            # nn.BatchNorm(o),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.linear = nn.Sequential(
            linearLayer( 2*768, 1024),
            linearLayer( 1024, 512),
            linearLayer( 512, 43),
        )

    def forward(self, x):
        _, pre  = self.transformer(x[0]['input_ids'], x[0]['attention_mask'])
        _, post = self.transformer(x[1]['input_ids'], x[1]['attention_mask'])
        output = torch.cat((pre,post),axis=-1)
        output = self.linear(output)
        
        return output