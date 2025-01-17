import torch
import torch.nn as nn
import torchvision.models as models
from AGCA import AGCA
import torch.optim as optim

class PrunedMobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(PrunedMobileNetV2, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        self.agca = AGCA(inc=64)  
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.mobilenet_v2.features(x)
        x = self.agca(x)  
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
def prunedmodel(num_classes):
    model = PrunedMobileNetV2(num_classes)
    model.mobilenet_v2.features = nn.Sequential(*model.mobilenet_v2.features[:-9])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    return model,device,loss,optimizer
