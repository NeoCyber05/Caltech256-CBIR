from __future__ import annotations

import math
from typing import Literal, List, Any

import numpy as np
import torch

from src import FeatureStore, Retrieve
#from src.entities.search_objects import ImageSearchObject
from src.utils.distance import d2s_typing, get_score


class VectorDBStore(FeatureStore):
    """
    Store and query image feature vectors using a retrieval model (KNN).
    """

    def __init__(self, retrieval_model: Retrieve) -> None:
        self.retrieval_model = retrieval_model
        self.vectors: List[Any] = []
        self.indices: List[int] = []
        self.images: List[Any] = []

    def add_images_with_vectors(self, images: List[Any], vectors: List[Any]) -> None:
        """
        add img & vectorDB to store
        """
        images, vectors = super().check_input_index(images, vectors)
        for image, vector in zip(images, vectors):
            self.images.append(image)
            self.vectors.append(vector)
            self.indices.append(len(self.indices))
        self.retrieval_model.fit(self.vectors, self.indices)

    def query_similar_images(self,vector: np.ndarray,top_k: int = 5,return_image: bool = False,distance_transform: d2s_typing = "exp",
    ) -> List[ImageSearchObject]:
        """
        Query top_k img
        """
        distances, indices = self.retrieval_model.predict(vector, k=top_k)
        d2s_func = get_score(distance_transform)
        result = []
        for idx, distance in zip(indices, distances):
            image = self.images[idx] if return_image else None
            result.append(ImageSearchObject(idx, d2s_func(distance), image))
        return result
