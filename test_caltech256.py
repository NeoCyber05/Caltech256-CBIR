#!/usr/bin/env python3
"""
Quick test for Caltech-256 dataset
"""

from src.datasets.caltech256 import Caltech256DataModule

def test_dataset():
    print("🧪 Testing Caltech-256 Dataset")
    
    # Test với batch nhỏ
    data_module = Caltech256DataModule(batch_size=4)
    
    try:
        print("📥 Preparing data...")
        data_module.prepare_data()
        
        print("⚙️  Setting up...")
        data_module.setup()
        
        train_loader = data_module.train_dataloader()
        test_loader = data_module.test_dataloader()
        
        print(f"✅ Success!")
        print(f"   Train: {len(data_module.train_dataset)} images")
        print(f"   Test: {len(data_module.test_dataset)} images")
        print(f"   Classes: {len(data_module.train_dataset.classes)}")
        print(f"   Sample classes: {data_module.train_dataset.classes[:5]}")
        
        # Test một batch
        print("\n🔍 Testing batch...")
        for images, labels, class_names in train_loader:
            print(f"   Batch shape: {images.shape}")
            print(f"   Labels: {labels}")
            print(f"   Classes: {class_names}")
            break
            
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset() 