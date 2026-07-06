"""
Model Architectures for Traffic Sign Recognition
Contains: ConvNeXt-Tiny, ResNet18, EfficientNet-B0 with Attention mechanisms
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention for region-based focus"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    """CBAM: Channel + Spatial Attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = SEBlock(channels, reduction)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class ConvNeXtTiny_Attention(nn.Module):
    """ConvNeXt-Tiny with SE-blocks and CBAM attention"""
    def __init__(self, num_classes=17, pretrained=False):
        super().__init__()
        self.backbone = tv_models.convnext_tiny(weights=None)
        self.backbone.features[0][0] = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        self.se1 = SEBlock(96, reduction=8)
        self.se2 = SEBlock(192, reduction=8)
        self.se3 = SEBlock(384, reduction=16)
        self.cbam = CBAM(768, reduction=16)
        num_features = 768
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(num_features),
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone.features[0](x)
        x = self.se1(x)
        x = self.backbone.features[1](x)
        x = self.backbone.features[2](x)
        x = self.se2(x)
        x = self.backbone.features[3](x)
        x = self.backbone.features[4](x)
        x = self.se3(x)
        x = self.backbone.features[5](x)
        x = self.backbone.features[6](x)
        x = self.cbam(x)
        x = self.backbone.features[7](x)
        x = self.backbone.avgpool(x)
        x = self.backbone.classifier(x)
        return x


class ResNet18_Attention(nn.Module):
    """ResNet18 with SE-blocks and CBAM attention"""
    def __init__(self, num_classes=17, pretrained=False):
        super().__init__()
        resnet = tv_models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.layer1 = resnet.layer1
        self.se1 = SEBlock(64, reduction=4)
        self.layer2 = resnet.layer2
        self.se2 = SEBlock(128, reduction=8)
        self.layer3 = resnet.layer3
        self.se3 = SEBlock(256, reduction=16)
        self.spatial3 = SpatialAttention(kernel_size=5)
        self.layer4 = resnet.layer4
        self.cbam4 = CBAM(512, reduction=16)
        self.avgpool = resnet.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.se1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)
        x = self.spatial3(x)
        x = self.layer4(x)
        x = self.cbam4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class EfficientNetB0_Attention(nn.Module):
    """EfficientNet-B0 with CBAM attention"""
    def __init__(self, num_classes=17, pretrained=False):
        super().__init__()
        self.backbone = tv_models.efficientnet_b0(weights=None)
        self.backbone.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.cbam = CBAM(1280, reduction=16)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone.features(x)
        x = self.cbam(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)
        return x
