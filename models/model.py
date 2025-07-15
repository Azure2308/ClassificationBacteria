import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super(ResNet18Classifier, self).__init__()
        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        self.model = resnet18(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
