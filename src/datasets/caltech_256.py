import os
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets
from torchvision.transforms import transforms

def get_caltech256_datasets(data_path: str, train_split: float = 0.8) -> tuple[Dataset, Dataset]:
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Dataset directory not found at {data_path}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=data_path, transform=None) # Keep original images for path
    train_size = int(train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    return train_dataset, test_dataset

def get_caltech256_loaders(data_path: str, batch_size: int = 32, train_split: float = 0.8) -> tuple[DataLoader, DataLoader]:

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    full_dataset = datasets.ImageFolder(root=data_path, transform=transform)
    train_size = int(train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
