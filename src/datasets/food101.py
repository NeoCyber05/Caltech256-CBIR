import os
import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision.datasets import Food101
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import yaml
import logging
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

# Setup logger
logger = logging.getLogger(__name__)

class CBIRDataset(Dataset):
    """Base class cho CBIR datasets"""
    pass

class Food101Dataset(CBIRDataset):
    def __init__(
        self, 
        root: str = "./data/food-101", 
        split: str = 'train', 
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        config_path: str = 'config.yaml'
    ):
        self.root = root
        self.split = split
        self.image_size = image_size
        
        # Setup transforms
        if augment and split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        # Load Food101 dataset
        self.dataset = Food101(
            root=root, 
            split=split, 
            transform=self.transform,
            download=True
        )

    def __getitem__(self, index):
        image, label = self.dataset[index]
        class_name = self.dataset.classes[label]
        return image, label, class_name

    def __len__(self):
        return len(self.dataset)

    def get_class_names(self) -> List[str]:
        return self.dataset.classes

    def get_num_classes(self) -> int:
        return len(self.dataset.classes)

    @property
    def classes(self) -> List[str]:
        return self.dataset.classes


class Food101DataModule:
    """DataModule để quản lý train/test dataloaders"""
    
    def __init__(
        self,
        root: str = 'data/food-101',
        batch_size: int = 32,
        num_workers: int = 2,
        image_size: Tuple[int, int] = (224, 224),
        config_path: str = 'config.yaml'
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.config_path = config_path
        
        # Initialize datasets
        self.train_dataset = None
        self.test_dataset = None
        
    def setup(self):
        """Setup train và test datasets"""
        self.train_dataset = Food101Dataset(
            root=self.root,
            split='train',
            image_size=self.image_size,
            augment=True,
            config_path=self.config_path
        )
        
        self.test_dataset = Food101Dataset(
            root=self.root,
            split='test',
            image_size=self.image_size,
            augment=False,
            config_path=self.config_path
        )
        
        logger.info(f"Setup hoàn tất: Train={len(self.train_dataset)}, Test={len(self.test_dataset)}")
    
    def get_train_loader(self) -> DataLoader:
        """Trả về train dataloader"""
        if self.train_dataset is None:
            self.setup()
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_test_loader(self) -> DataLoader:
        """Trả về test dataloader"""
        if self.test_dataset is None:
            self.setup()
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_class_names(self) -> List[str]:
        """Trả về danh sách class names"""
        if self.train_dataset is None:
            self.setup()
        return self.train_dataset.get_class_names()
    
    def get_num_classes(self) -> int:
        """Trả về số lượng classes"""
        if self.train_dataset is None:
            self.setup()
        return self.train_dataset.get_num_classes()


def create_data_module(config_path: str = 'config.yaml') -> Food101DataModule:
    """Factory function để tạo DataModule từ config"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        dataset_config = config.get('dataset', {})
        
        return Food101DataModule(
            root=dataset_config.get('path', 'data/food-101'),
            batch_size=dataset_config.get('batch_size', 32),
            num_workers=2,
            image_size=tuple(dataset_config.get('image_size', [224, 224])),
            config_path=config_path
        )
    except Exception as e:
        logger.warning(f"Không thể đọc config: {e}. Sử dụng default.")
        return Food101DataModule()


if __name__ == "__main__":
    # Test dataset
    print("Testing Food101 Dataset...")
    
    data_module = create_data_module()
    data_module.setup()
    
    train_loader = data_module.get_train_loader()
    print(f"Train: {len(data_module.train_dataset)} samples")
    print(f"Classes: {data_module.get_num_classes()}")
    
    # Test một batch
    for batch_idx, (images, labels, class_names) in enumerate(train_loader):
        print(f"Batch {batch_idx}: {images.shape}, Labels: {labels[:3]}")
        print(f"Classes: {class_names[:3]}")
        break
    
    print("Test hoàn tất!") 