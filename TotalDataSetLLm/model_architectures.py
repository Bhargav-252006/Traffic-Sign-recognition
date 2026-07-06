"""
Model Architectures for Traffic Sign Recognition — Full GTSRB (43 Classes)
IEEE Paper Models: ConvNeXt-Tiny, ResNet18, EfficientNet-B0
These are the same architectures used in 'only the final model and the output.ipynb'
and saved as ieee_model_*.pth / ensemble_config.pth
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


# ── 1. ConvNeXt-Tiny ──
# Best for: complex feature hierarchies, modern architecture
# Modification: stride=1 first conv for 32×32 input instead of stride=4
class ConvNeXtTinyTraffic(nn.Module):
    def __init__(self, num_classes=43, pretrained=False):
        super().__init__()
        self.backbone = tv_models.convnext_tiny(
            weights='IMAGENET1K_V1' if pretrained else None)
        # Replace stride-4 stem with stride-1 for 32×32 input
        self.backbone.features[0][0] = nn.Conv2d(3, 96, 3, stride=1, padding=1)
        in_feat = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1), nn.LayerNorm(in_feat),
            nn.Linear(in_feat, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes))

    def forward(self, x):
        return self.backbone(x)


# ── 2. Enhanced ResNet18 ──
# Best for: fast training, solid baseline, proven on traffic signs
# Modification: stride-1 conv1, no maxpool (preserves spatial info at 32×32)
class EnhancedResNet18(nn.Module):
    def __init__(self, num_classes=43, pretrained=False):
        super().__init__()
        resnet = tv_models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        resnet.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        in_feat = resnet.fc.in_features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_feat, 512), nn.ReLU(True), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(0.25),
            nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


# ── 3. EfficientNet-B0 ──
# Best for: parameter efficiency, compound scaling
# Modification: stride-1 stem conv for 32×32
class EfficientNetB0Traffic(nn.Module):
    def __init__(self, num_classes=43, pretrained=False):
        super().__init__()
        self.backbone = tv_models.efficientnet_b0(
            weights='IMAGENET1K_V1' if pretrained else None)
        self.backbone.features[0][0] = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        in_feat = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_feat, 512), nn.SiLU(True), nn.Dropout(0.2),
            nn.Linear(512, num_classes))

    def forward(self, x):
        return self.backbone(x)


# ── Factory ──
MODEL_REGISTRY = {
    'convnext_tiny':   ConvNeXtTinyTraffic,
    'resnet18':        EnhancedResNet18,
    'efficientnet_b0': EfficientNetB0Traffic,
}


def get_model(name, num_classes=43):
    """Create a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](num_classes=num_classes)


if __name__ == "__main__":
    print("IEEE Model Architectures — Full GTSRB (43 classes)")
    print("=" * 60)
    for name, cls in MODEL_REGISTRY.items():
        m = cls(num_classes=43)
        total_p = sum(x.numel() for x in m.parameters())
        print(f"  {name:<20} {total_p:>12,} params")
        del m
    print("=" * 60)
