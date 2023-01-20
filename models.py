
from torchvision import datasets, models, transforms

from torch import nn
import torchvision

class TransferModel18(nn.Module):
    """
    ResNet18 transfer learning model
    
    """
    def __init__(self, pretrained=True, freeze=True, device='cuda'):
        super(TransferModel18, self).__init__()

        if pretrained==True:
            self.model = torchvision.models.resnet18(pretrained=True).to(device)
        else:
            self.model = torchvision.models.resnet18(pretrained=False).to(device)
        
        if freeze==True:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            pass

        # Fully-connected layer
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=3, bias=True, device=device),
        ) 
        

    def forward(self, x):
        return self.model(x)

