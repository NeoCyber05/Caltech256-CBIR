from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from src.utils.distance import d2s_typing

class ImageSearchObject:
    def __init__(self, index: int, score: float, image=None) -> None:
        self.index = index
        self.score = score
        self.image = image


class FeatureStore(ABC):
    def validate_inputs(self, images: np.ndarray | list, features: np.ndarray | list):
        """Validate and normalize image and feature inputs"""
        if isinstance(images, list) and isinstance(features, list):
            assert len(images) == len(features), "Images and features must have same length"
            assert (len(images) > 0 and len(features) > 0), "Images and features cannot be empty"
        elif isinstance(images, np.ndarray) and isinstance(features, np.ndarray):
            images = [images]
            features = [features]

        return images, features

    @abstractmethod
    def add(self, images: np.ndarray | list, features: np.ndarray | list) -> None:
        """Add images and their features to the store"""
        pass

    @abstractmethod
    def search(self, feature: np.ndarray, k=5, distance_transform: d2s_typing = "exp") -> list[ImageSearchObject]:
        """Find k most similar items to query feature"""
        pass