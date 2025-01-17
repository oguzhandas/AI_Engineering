import torch.nn as nn

class AGCA(nn.Module):
    def __init__(self, inc, reduction=32):
        super(AGCA, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inc, inc // reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(inc // reduction, inc, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.global_avg_pool(x)
        attention = self.conv1(attention)
        attention = nn.ReLU()(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention