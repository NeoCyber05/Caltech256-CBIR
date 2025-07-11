import os
import pickle
import gzip
import pandas as pd
import numpy as np
import torch
from time import time
from tqdm import tqdm

from src.datasets.caltech256 import Caltech256DataModule
from src.featuring.rgb_histogram import RGBHistogram
from src.storage.VectorDBStore import VectorDBStore
from src.retrieval.KNN import KNNRetrieval
from src.pipeline import CBIR
from src.metrics import average_precision, recall, hit_rate


def main():
    # Config - Sử dụng gần toàn bộ dataset (30,607 total images)
    TRAIN_SIZE = 24607  # ~80% of dataset  
    TEST_SIZE = 6000    # ~20% of dataset
    BATCH_SIZE = 64
    
    print(f"🚀 Caltech-256 CBIR Evaluation")
    print(f"📊 Full dataset: Train={TRAIN_SIZE}, Test={TEST_SIZE}")
    print(f"📝 Note: Caltech-256 không có train/test split chính thức")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    # Calculate dataset path BEFORE creating data_module  
    # Go up 3 levels: file -> evaluation -> src -> project_root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(project_root, 'data', 'caltech-256')
    print(f"📁 Dataset path: {dataset_path}")
    print(f"📂 Dataset exists: {os.path.exists(dataset_path)}")
    
    # Check if 256_ObjectCategories exists
    full_dataset_path = os.path.join(dataset_path, '256_ObjectCategories')
    print(f"📂 Full dataset path: {full_dataset_path}")
    print(f"📂 Full dataset exists: {os.path.exists(full_dataset_path)}")
    
    # Dataset
    data_module = Caltech256DataModule(root=dataset_path, batch_size=BATCH_SIZE)
    data_module.prepare_data()
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()
    
    print(f"Classes: {len(data_module.train_dataset.classes)}")
    
    # CBIR Setup - tối ưu parameters
    rgb_histogram = RGBHistogram(n_bin=2, h_type="region", n_slice=3)
    color_store = VectorDBStore(KNNRetrieval(metric="cosine"))
    cbir = CBIR(rgb_histogram, color_store)
    
    # Indexing
    print(f"Indexing {TRAIN_SIZE} images...")
    start = time()
    indexed = 0
    
    for images, labels, _ in tqdm(train_loader, desc="Indexing"):
        if indexed >= TRAIN_SIZE:
            break
            
        if device.type == "cuda":
            images = images.to(device)
        
        count = min(len(images), TRAIN_SIZE - indexed)
        images = images[:count]
        images = (images.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        
        cbir.add_images(images)
        indexed += count
    
    indexing_time = time() - start
    print(f"Indexed {indexed} images in {indexing_time:.2f}s")
    
    # Save compressed model
    model_path = 'out/caltech256_model.pkl.gz'
    with gzip.open(model_path, 'wb') as f:
        pickle.dump(cbir, f)
    
    file_size = os.path.getsize(model_path) / 1024 / 1024
    print(f"💾 Model saved: {file_size:.2f} MB")
    
    # Evaluation với multiple k values
    print("Starting evaluation...")
    start = time()
    results = []
    ground_truth = []
    tested = 0
    
    # Get dataset targets for evaluation
    dataset_targets = []
    for images, labels, _ in train_loader:
        if len(dataset_targets) >= indexed:
            break
        count = min(len(labels), indexed - len(dataset_targets))
        dataset_targets.extend(labels[:count].numpy())
    dataset_targets = np.array(dataset_targets)
    
    # Query với k=100 để có đủ data cho tất cả metrics
    MAX_K = 100
    
    for images, labels, _ in tqdm(test_loader, desc="Testing"):
        if tested >= TEST_SIZE:
            break
            
        if device.type == "cuda":
            images = images.to(device)
        
        count = min(len(images), TEST_SIZE - tested)
        images = images[:count]
        labels = labels[:count]
        
        images = (images.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        
        for image in images:
            if tested >= TEST_SIZE:
                break
            # Query với k=100 để tính được MAP@100, Recall@100
            result = cbir.query_similar_images(image, k=MAX_K)
            results.append(result)
            tested += 1
        
        ground_truth.extend(labels.numpy())
    
    retrieval_time = time() - start
    print(f"Tested {tested} images in {retrieval_time:.2f}s")
    
    # Calculate metrics cho k=5, k=10, k=100
    k_values = [5, 10, 100]
    metrics_data = {}
    
    print(f"\n📊 Calculating metrics for k={k_values}...")
    
    for k in k_values:
        map_k, recall_k, hit_k = [], [], []
        
        for r, gt in zip(results, ground_truth):
            # Lấy top-k results
            top_k_results = r[:k]
            indices = [item.index for item in top_k_results]
            preds = np.take(dataset_targets, indices)
            relevant = np.where(dataset_targets == gt)[0]
            
            map_k.append(average_precision(preds.tolist(), [gt], k))
            recall_k.append(recall(indices, relevant, k))
            hit_k.append(hit_rate(preds.tolist(), [gt], k))
        
        # Store metrics
        metrics_data[f'MAP@{k}'] = np.mean(map_k)
        metrics_data[f'Recall@{k}'] = np.mean(recall_k)
        metrics_data[f'HitRate@{k}'] = np.mean(hit_k)
        
        print(f"   📈 k={k}: MAP={np.mean(map_k):.4f}, Recall={np.mean(recall_k):.4f}, HitRate={np.mean(hit_k):.4f}")
    
    # Add timing and file size metrics
    metrics_data['indexing_time'] = indexing_time
    metrics_data['retrieval_time'] = retrieval_time
    metrics_data['file_size_mb'] = file_size
    metrics_data['indexed_images'] = indexed
    metrics_data['tested_images'] = tested
    
    # Save results
    results_df = pd.DataFrame([metrics_data])
    results_df.to_csv('out/caltech256_results.csv', index=False)
    
    print(f"\n📊 Final Results Summary:")
    print(f"=" * 50)
    for k in k_values:
        print(f"📈 k={k}:")
        print(f"   MAP@{k}: {metrics_data[f'MAP@{k}']:.4f}")
        print(f"   Recall@{k}: {metrics_data[f'Recall@{k}']:.4f}")
        print(f"   HitRate@{k}: {metrics_data[f'HitRate@{k}']:.4f}")
    
    print(f"\n⏱️  Performance:")
    print(f"   Indexing: {indexing_time:.2f}s ({indexed} images)")
    print(f"   Retrieval: {retrieval_time:.2f}s ({tested} images)")
    print(f"   File size: {file_size:.2f}MB")
    
    print("✅ Results saved to out/caltech256_results.csv")
    
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    os.makedirs('out', exist_ok=True)
    main() 