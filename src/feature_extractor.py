from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Union, List
import cv2
import numpy as np


class FeatureExtractor(ABC):
    """Base abstract class for all feature extractors."""

    @abstractmethod
    def __call__(self, input):
        """Process input and extract features."""
        pass

    def read_image(self, input, expected_dims):
        if isinstance(input, os.PathLike):
            return cv2.imread(str(input))
        elif isinstance(input, str):
            if not os.path.exists(input):
                raise FileNotFoundError(f"Not found File: {input}")
            return cv2.imread(input)
        elif isinstance(input, np.ndarray):
            assert len(input.shape) == expected_dims, f"Image must be a {expected_dims}D array, got shape {input.shape}"
            return input
        else:
            raise TypeError(f"Invalid type for image: {type(input)}")


class SingleFeatureExtractor(FeatureExtractor):
    """Extracts features from a single image."""

    @abstractmethod
    def __call__(self, input: Union[np.ndarray, os.PathLike]):
        pass

    def read_image(self, input: Union[np.ndarray, os.PathLike, str]):
        return super().read_image(input, expected_dims=3) #(height × width × channels)


class BatchFeatureExtractor(FeatureExtractor):
    """Extracts features from multiple images."""

    @abstractmethod
    def __call__(self, input: Union[np.ndarray, List[os.PathLike]]):
        pass

    def read_image(self, input: np.ndarray):
        return super().read_image(input, expected_dims=4) #(batch_size × height × width × channels)