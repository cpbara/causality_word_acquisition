from torch import nn
from torchvision.models import vgg16 as VGG16

class ImageModel_4096(nn.Module):
    def __init__(self):
        super(ImageModel_4096, self).__init__()
        
        self.conv_model = nn.Sequential(
            *list(VGG16(pretrained=True).children())[:-1],
            )
        
        self.img_linear = nn.Sequential(
            *list(list(VGG16(pretrained=True).children())[-1].children())[:-3],
        )
        
    def forward(self,x):
        return self.img_linear(self.conv_model(x).reshape(-1,512*7*7))

class ImageModel_7x7x512(nn.Module):
    def __init__(self):
        super(ImageModel_7x7x512, self).__init__()
        
        self.conv_model = nn.Sequential(
            *list(VGG16(pretrained=True).children())[:-1],
            )
        
    def forward(self,x):
        return self.conv_model(x).reshape(-1,512*7*7)