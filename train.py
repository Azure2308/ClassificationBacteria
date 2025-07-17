import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from models.model import ResNet18Classifier, VGG16Classifier, EfficientNetB0Classifier
from utils.datasets import CustomImageDataset


def train_model(model, train_loader, val_loader, device, start_epoch, epochs, optimizer, criterion, best_acc):
    train_losses, val_accs = [], []
    for epoch in range(start_epoch, epochs + 1):
        model.train()

        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_loss:.4f}")

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = 100 * correct / total
        val_accs.append(val_acc)
        print(f" Validation Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_acc': best_acc
            }, 'best_checkpoint.pth')
            print(f"New best model saved with acc: {best_acc:.2f}%")
    epochs_range = list(range(start_epoch, epochs + 1))
    plt.figure()
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('training_history1.png')
    print('Saved training_history1.png')
    print(f"Training complete. Best Acc: {best_acc:.2f}%")

def main():
    data_dir = "data"
    batch_size = 32
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CustomImageDataset(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = CustomImageDataset(os.path.join(data_dir, 'val'), transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ResNet18Classifier(num_classes=len(train_dataset.classes), use_pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    start_epoch = 1
    best_acc = 0.0

    if os.path.exists('best_checkpoint.pth'):
        print("Loading checkpoint...")
        checkpoint = torch.load('best_checkpoint.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resuming from epoch {checkpoint['epoch']} with best acc {best_acc:.2f}%")

    train_model(model, train_loader, val_loader, device, start_epoch, epochs, optimizer, criterion, best_acc)

if __name__ == "__main__":
    main()