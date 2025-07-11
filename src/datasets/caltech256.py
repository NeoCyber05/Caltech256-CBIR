import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import glob
from typing import Tuple, List

class Caltech256Dataset(Dataset):
    def __init__(self, root="./data/caltech-256", split='train', image_size=(224, 224), test_size=0.2):
        self.root = root
        self.split = split
        self.image_size = image_size
        
        # Transforms - avoid lambda for multiprocessing
        transform_list = [
            transforms.Resize(image_size),
        ]
        
        if split == 'train':
            transform_list.append(transforms.RandomHorizontalFlip())
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.transform = transforms.Compose(transform_list)
        
        self._check_dataset()
        self._load_data(test_size)
    
    def _check_dataset(self):
        dataset_path = os.path.join(self.root, "256_ObjectCategories")
        if os.path.exists(dataset_path):
            return
            
        print("❌ Caltech-256 dataset not found!")
        print("📥 Please download manually:")
        print("1. Kaggle: https://www.kaggle.com/datasets/jessicali9530/caltech256")
        print("2. Official: https://data.caltech.edu/records/nyy15-4j048")
        print(f"3. Extract 256_ObjectCategories folder to: {self.root}")
        print("4. Folder structure should be:")
        print(f"   {self.root}/256_ObjectCategories/001.ak47/...")
        print(f"   {self.root}/256_ObjectCategories/002.american-flag/...")
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    def _load_data(self, test_size):
        dataset_path = os.path.join(self.root, "256_ObjectCategories")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Get classes (exclude clutter and invalid folders)
        try:
            all_dirs = [d for d in os.listdir(dataset_path) 
                       if os.path.isdir(os.path.join(dataset_path, d))]
        except Exception as e:
            raise FileNotFoundError(f"Cannot read dataset directory: {e}")
        
        # Filter valid class directories (format: xxx.class_name)
        class_dirs = []
        self.classes = []
        
        for d in sorted(all_dirs):
            # Skip clutter and invalid formats
            if d.startswith('257') or '.' not in d:
                continue
            
            parts = d.split('.', 1)
            if len(parts) == 2 and parts[0].isdigit():
                class_dirs.append(d)
                self.classes.append(parts[1])
        
        print(f"📂 Found {len(class_dirs)} valid categories")
        
        # Load all images (multiple extensions)
        all_paths, all_labels = [], []
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(dataset_path, class_dir)
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                images.extend(glob.glob(os.path.join(class_path, ext)))
            all_paths.extend(images)
            all_labels.extend([class_idx] * len(images))
        
        if len(all_paths) == 0:
            raise ValueError("No images found in dataset!")
        
        print(f"📊 Loaded {len(all_paths)} total images from {len(class_dirs)} classes")
        
        # Train/test split với error handling
        if len(set(all_labels)) < 2:
            raise ValueError("Need at least 2 classes for stratified split")
        
        try:
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                all_paths, all_labels, test_size=test_size, random_state=42, stratify=all_labels)
        except ValueError as e:
            # Fallback to non-stratified split nếu có class quá ít sample
            print(f"⚠️  Stratified split failed: {e}. Using random split.")
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                all_paths, all_labels, test_size=test_size, random_state=42)
        
        if self.split == 'train':
            self.image_paths, self.labels = train_paths, train_labels
        else:
            self.image_paths, self.labels = test_paths, test_labels
        
        print(f"📋 {self.split.capitalize()}: {len(self.image_paths)} images")
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        class_name = self.classes[label]
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        return image, label, class_name
    
    def __len__(self):
        return len(self.image_paths)


class Caltech256DataModule:
    def __init__(self, root='data/caltech-256', batch_size=32, num_workers=0, image_size=(224, 224)):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
    def prepare_data(self):
        # Check if dataset exists
        temp_dataset = Caltech256Dataset(self.root, 'train', self.image_size, test_size=0.2)
        del temp_dataset
        
    def setup(self, stage=None):
        self.train_dataset = Caltech256Dataset(self.root, 'train', self.image_size)
        self.test_dataset = Caltech256Dataset(self.root, 'test', self.image_size)
        
        # Compatibility
        self.food101_train = self.train_dataset
        self.food101_test = self.test_dataset
        
        print(f"Train: {len(self.train_dataset)}, Test: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        if not hasattr(self, 'train_dataset'):
            self.setup()
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                         shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def test_dataloader(self):
        if not hasattr(self, 'test_dataset'):
            self.setup()
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                         shuffle=False, num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    # Test
    data_module = Caltech256DataModule(batch_size=4)
    data_module.prepare_data()
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    print(f"Classes: {len(data_module.train_dataset.classes)}")
    
    # Test batch
    for images, labels, class_names in train_loader:
        print(f"Batch: {images.shape}, Labels: {labels[:3]}")
        break 