# evaluate.py
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from models.model import ResNet18Classifier, EfficientNetB0Classifier, VGG16Classifier
from utils.datasets import CustomImageDataset


def evaluate():
    data_dir = 'data'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = CustomImageDataset(os.path.join(data_dir, 'val'), transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    checkpoint_path = 'best_checkpoint3.pth' if os.path.exists('best_checkpoint3.pth') else 'model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = VGG16Classifier(num_classes=len(val_dataset.classes), use_pretrained=False)
    state = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_preds, target_names=val_dataset.classes))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(val_dataset.classes))
    plt.xticks(tick_marks, val_dataset.classes, rotation=45)
    plt.yticks(tick_marks, val_dataset.classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print('Saved confusion_matrix.png')

if __name__ == "__main__":
    evaluate()