from __future__ import annotations

import math
from typing import Literal, List, Any

import numpy as np
import torch

from src import FeatureStore, Retrieve
from src.feature_store import ImageSearchObject
from src.utils.distance import d2s_typing, get_score


class VectorDBStore(FeatureStore):
    """
    Store and query image feature vectors using a retrieval model (KNN).
    Optimized version that doesn't store original images to reduce memory usage.
    """

    def __init__(self, retrieval_model: Retrieve) -> None:
        self.retrieval_model = retrieval_model
        self.vectors: List[Any] = []
        self.indices: List[int] = []
        # Remove self.images to save memory - don't store original images

    def add(self, images: List[Any], features: List[Any]) -> None:
        """
        Add images and features to store.
        Note: Original images are not stored to save memory.
        """
        images, features = self.validate_inputs(images, features)
        for image, feature in zip(images, features):
            # Don't store image to save memory: self.images.append(image)
            self.vectors.append(feature)
            self.indices.append(len(self.indices))
        self.retrieval_model.fit(self.vectors, self.indices)

    def search(self, feature: np.ndarray, k=5, distance_transform: d2s_typing = "exp") -> List[ImageSearchObject]:
        """
        Search for similar images.
        Returns ImageSearchObject with None for image data to save memory.
        """
        distances, indices = self.retrieval_model.predict(feature, k=k)
        d2s_func = get_score(distance_transform)
        result = []
        for idx, distance in zip(indices, distances):
            # Return None for image to save memory
            result.append(ImageSearchObject(idx, d2s_func(distance), None))
        return result
