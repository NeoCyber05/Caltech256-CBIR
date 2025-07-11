from __future__ import annotations

import os
from typing import List, Union
import cv2
import numpy as np

from src import BatchFeatureExtractor,FeatureExtractor,FeatureStore,SingleFeatureExtractor
from src.feature_store import ImageSearchObject
from src.utils.distance import d2s_typing

class CBIR:
    """
    Connects feature extractor and feature store for indexing and querying similar images.
    """
    def __init__(self, feature_extractor: FeatureExtractor, feature_store: FeatureStore):
        self.feature_extractor = feature_extractor
        self.feature_store = feature_store

    def _prepare_images(self, images: Union[List[np.ndarray], List[os.PathLike], np.ndarray, os.PathLike]) -> List[np.ndarray]:
        """
        Convert img -> np.ndarray.
        """
        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                return [images]
            elif len(images.shape) == 4:
                return [img for img in images]
        elif isinstance(images, (str, os.PathLike)):
            if not os.path.exists(images):
                raise FileNotFoundError(f"File not found: {images}")
            return [cv2.imread(str(images))]
        elif isinstance(images, list):
            assert len(images) > 0, "Images cannot be empty"
            if isinstance(images[0], (str, os.PathLike)):
                return [cv2.imread(str(image)) for image in images]
            elif isinstance(images[0], np.ndarray):
                return images
        raise TypeError("Invalid input type for images")

    def add_images(
        self, images: Union[List[np.ndarray], List[os.PathLike], np.ndarray, os.PathLike]
    ) -> None:
        """
        Add img & feature -> feature store.
        """
        images_list = self._prepare_images(images)
        features = []
        if isinstance(self.feature_extractor, SingleFeatureExtractor):
            for image in images_list:
                features.append(self.feature_extractor(image))
        elif isinstance(self.feature_extractor, BatchFeatureExtractor):
            features = self.feature_extractor(images_list).tolist()
        self.feature_store.add(images_list, features)

    def query_similar_images(self,image: Union[np.ndarray, os.PathLike],k: int = 5,distance_transform: d2s_typing = "exp",) -> List[ImageSearchObject]:
        if isinstance(image, (str, os.PathLike)):
            if not os.path.exists(image):
                raise FileNotFoundError(f"File not found: {image}")
            image = cv2.imread(str(image))
        if isinstance(self.feature_extractor, SingleFeatureExtractor):
            feature = self.feature_extractor(image)
            return self.feature_store.search(feature, k=k, distance_transform=distance_transform)
        elif isinstance(self.feature_extractor, BatchFeatureExtractor):
            features = self.feature_extractor([image])
            result = []
            for feature in features:
                result.append(self.feature_store.search(feature, k=k, distance_transform=distance_transform))
            return result
