import time
import itertools
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import logging
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.metrics import average_precision, hit_rate, recall

logger = logging.getLogger(__name__)


class EvaluationFramework:
    def __init__(self, output_dir: str = "out/results"):
        self.methods = {}
        self.cache = {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def add_method(self, name: str, cbir_instance):
        self.methods[name] = cbir_instance
        self.cache[name] = {}
        
    def index_methods(self, dataloader: DataLoader):
        """Index all methods"""
        indexing_times = {}
        
        for name, method in self.methods.items():
            start = time.time()
            for images, labels, _ in tqdm(dataloader, desc=f"Indexing {name}"):
                if hasattr(images, 'numpy'):
                    images = (images.numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
                method.add_images(images)
            indexing_times[name] = time.time() - start
            
        return indexing_times
    
    def _get_retrieval_results(self, image, image_idx, k, d2s):
        """Get results with caching"""
        results = {}
        
        for name, method in self.methods.items():
            cache_key = f"{image_idx}-{k}-{d2s}"
            
            if cache_key in self.cache[name]:
                results[name] = self.cache[name][cache_key]
            else:
                result = method.query_similar_images(image, k=k, distance_transform=d2s)
                self.cache[name][cache_key] = result
                results[name] = result
                
        return results
    
    def _ensemble_results(self, method_results, weights, k=1000):
        """Combine multiple method results"""
        if len(method_results) == 1:
            return list(method_results.values())[0]
            
        if not weights:
            weights = [1.0] * len(method_results)
            
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Combine scores
        combined_scores = {}
        for i, (name, results) in enumerate(method_results.items()):
            for result in results:
                idx = result.index
                score = result.score * weights[i]
                combined_scores[idx] = combined_scores.get(idx, 0) + score
        
        # Sort and return top k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        from collections import namedtuple
        Result = namedtuple('Result', ['index', 'score'])
        return [Result(idx, score) for idx, score in sorted_results[:k]]
    
    def evaluate(self, test_dataloader, k=1000, d2s="exp", weights=None):
        """Run evaluation"""
        all_predictions = []
        all_ground_truths = []
        dataset = test_dataloader.dataset
        
        start_time = time.time()
        image_count = 0
        
        for images, labels, _ in tqdm(test_dataloader, desc="Evaluating"):
            if hasattr(images, 'numpy'):
                images = (images.numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
                
            for image, label in zip(images, labels):
                # Get results from all methods
                method_results = self._get_retrieval_results(image, image_count, k, d2s)
                
                # Ensemble if multiple methods
                final_result = self._ensemble_results(method_results, weights, k)
                
                # Extract predictions
                predictions = [r.index for r in final_result]
                all_predictions.append(predictions)
                all_ground_truths.append(label.item() if hasattr(label, 'item') else label)
                
                image_count += 1
        
        retrieval_time = (time.time() - start_time) / len(dataset)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_ground_truths, dataset)
        metrics['avg_retrieval_time'] = round(retrieval_time, 6)
        
        return metrics
    
    def _calculate_metrics(self, predictions, ground_truths, dataset):
        """Calculate all metrics"""
        # Get dataset targets
        if hasattr(dataset, 'targets'):
            targets = np.array(dataset.targets)
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
            targets = np.array(dataset.dataset.targets)
        else:
            targets = np.arange(len(dataset))
        
        metrics = {}
        k_values = [1, 5, 10, 20, 100, 1000]
        
        for k in k_values:
            if k > len(predictions[0]):
                continue
                
            ap_scores = []
            hit_scores = []
            recall_scores = []
            
            for pred, gt in zip(predictions, ground_truths):
                gt_indices = np.where(targets == gt)[0]
                
                ap_scores.append(average_precision(pred, [gt], k))
                hit_scores.append(hit_rate(pred, [gt], k))
                recall_scores.append(recall(pred, gt_indices, k))
            
            metrics[f'map@{k}'] = round(np.mean(ap_scores), 6)
            metrics[f'hit_rate@{k}'] = round(np.mean(hit_scores), 6)
            metrics[f'recall@{k}'] = round(np.mean(recall_scores), 6)
        
        return metrics
    
    def grid_search(self, train_loader, test_loader, 
                   k_values=None, d2s_values=None, weight_combinations=None,
                   experiment_name="eval"):
        """Run grid search evaluation"""
        
        # Default values
        if k_values is None:
            k_values = [len(train_loader.dataset)]
        if d2s_values is None:
            d2s_values = ["exp", "log", "logistic", "gaussian", "inverse"]
        if weight_combinations is None and len(self.methods) > 1:
            n_methods = len(self.methods)
            weight_combinations = [
                [1.0] * n_methods,  # Equal
                [0.8] + [0.2/(n_methods-1)] * (n_methods-1),  # First dominant
                [0.3] + [0.7/(n_methods-1)] * (n_methods-1),  # Others dominant
            ]
        elif len(self.methods) == 1:
            weight_combinations = [[1.0]]
        
        # Index all methods
        indexing_times = self.index_methods(train_loader)
        avg_indexing_time = np.mean(list(indexing_times.values()))
        
        # Initialize results
        results = []
        
        # Grid search
        total = len(k_values) * len(d2s_values) * len(weight_combinations)
        
        with tqdm(total=total, desc="Grid Search") as pbar:
            for k, d2s, weights in itertools.product(k_values, d2s_values, weight_combinations):
                
                # Evaluate
                metrics = self.evaluate(test_loader, k=k, d2s=d2s, weights=weights)
                
                # Store result
                result = {
                    "k": k,
                    "distance2score": d2s,
                    "weight": str(weights),
                    "avg_indexing_time": round(avg_indexing_time, 6)
                }
                result.update(metrics)
                results.append(result)
                
                # Print current result
                print(f"k={k}, d2s={d2s}, weights={weights}")
                print(f"mAP@1={metrics.get('map@1', 0):.4f}, mAP@5={metrics.get('map@5', 0):.4f}, mAP@10={metrics.get('map@10', 0):.4f}")
                
                pbar.update(1)
        
        # Save results
        df = pd.DataFrame(results)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"{experiment_name}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        
        return df
    
    def save_cache(self, filename):
        """Save cache"""
        with open(self.output_dir / f"{filename}.pkl", 'wb') as f:
            pickle.dump(self.cache, f)
    
    def load_cache(self, filename):
        """Load cache"""
        with open(self.output_dir / f"{filename}.pkl", 'rb') as f:
            self.cache = pickle.load(f)


# Example usage
def example():
    eval_framework = EvaluationFramework()
    
    # Add your CBIR methods
    # eval_framework.add_method("rgb_histogram", cbir_rgb)
    # eval_framework.add_method("edge_histogram", cbir_edge)
    
    # Run evaluation
    # results = eval_framework.grid_search(train_loader, test_loader)
    
    pass


if __name__ == "__main__":
    example() 