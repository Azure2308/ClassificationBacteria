import os

from PIL import Image
from torch.utils.data import Dataset


def pil_loader(path):
    # Загружает изображение и конвертирует в RGB
    return Image.open(path).convert("RGB")

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, loader=pil_loader):
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader
        self.classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    path = os.path.join(cls_dir, fname)
                    self.samples.append((path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.loader(path)
        if self.transform:
            image = self.transform(image)
        return image, label