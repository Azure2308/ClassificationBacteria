# evaluate.py

import torch
from utils.datasets import get_dataloaders
from models.model import SimpleCNN
from sklearn.metrics import classification_report

def evaluate_full():
    data_dir = "data"
    batch_size = 32
    image_size = (128, 128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names = get_dataloaders(data_dir, batch_size, image_size)

    model = SimpleCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load("model.pth"))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    evaluate_full()