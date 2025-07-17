import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super(ResNet18Classifier, self).__init__()
        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        self.model = resnet18(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes=8, use_pretrained=True):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if use_pretrained else None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class VGG16Classifier(nn.Module):
    def __init__(self, num_classes=8, use_pretrained=True):
        super().__init__()
        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if use_pretrained else None)
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)