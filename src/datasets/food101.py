import os
from torchvision.datasets import Food101
from torchvision import transforms
from torch.utils.data import DataLoader

class Food101Dataset:
    def __init__(self, root='data/food-101', split='train', batch_size=32, num_workers=2, image_size=(224, 224)):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.dataset = Food101(root=root, split=split, download=True, transform=self.transform)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=num_workers)

    def get_loader(self):
        return self.loader

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx] 