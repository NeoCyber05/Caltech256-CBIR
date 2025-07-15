from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class ImageSearchObject:
    def __init__(self, index: str | int, score: float, image=None) -> None:
        self.index = index
        self.score = score
        self.image = image


class FeatureStore(ABC):
    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        """Add items to the store"""
        pass

    @abstractmethod
    def search(self, feature: np.ndarray, k=5, **kwargs) -> list[ImageSearchObject]:
        """Find k most similar items to query feature"""
        pass