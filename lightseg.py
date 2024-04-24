import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class LiteSegDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LiteSegDecoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)

class LiteSegPP(nn.Module):
    def __init__(self, num_classes):
        super(LiteSegPP, self).__init__()
        # Define encoder
        self.encoder = models.mobilenet_v2(pretrained=True).features

        # Define decoder
        self.decoder = LiteSegDecoder(in_channels=320, num_classes=num_classes)

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Decoder
        return self.decoder(x)