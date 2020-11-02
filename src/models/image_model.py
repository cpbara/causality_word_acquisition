from torch import nn
from torchvision.models import vgg16 as VGG16

class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        
        self.conv_model = nn.Sequential(
            *list(VGG16(pretrained=True).children())[:-1],
            )
        
        self.img_linear = nn.Sequential(
            *list(list(VGG16(pretrained=True).children())[-1].children())[:-3],
        )
        
    def forward(self,x):
        return self.img_linear(self.conv_model(x).reshape(-1,512*7*7))