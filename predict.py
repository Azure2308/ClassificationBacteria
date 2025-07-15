# predict.py

import torch
from torchvision import transforms
from PIL import Image
from models.model import SimpleCNN
from utils.datasets import get_dataloaders

def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, class_names = get_dataloaders("data", batch_size=1)

    model = SimpleCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load("model.pth"))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        predicted_class = class_names[pred.item()]
        print(f"🔎 Предсказанный класс: {predicted_class}")

if __name__ == "__main__":
    predict("some_image.jpg")