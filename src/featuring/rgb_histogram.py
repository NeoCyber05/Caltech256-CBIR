from __future__ import annotations

import os
from typing import Literal

import cv2
import numpy as np

from src.feature_extractor import SingleFeatureExtractor


class RGBHistogram(SingleFeatureExtractor):
    """RGB Histogram feature extractor that can work in global or regional mode."""

    def __init__(
        self,
        n_bin: int = 12,
        n_slice: int = 3,
        h_type: Literal["global", "region"] = "region",
        normalize: bool = True,
    ) -> None:
        """
        n_bin: Number of bins for histogram
        n_slice: Number of image slices for regional histogram
        h_type: "global" for whole image or "region" for divided regions
        normalize: Whether to normalize the histogram
        """
        self.n_bin = n_bin
        self.n_slice = n_slice
        self.h_type = h_type
        self.normalize = normalize

    def __call__(self, input: np.ndarray | os.PathLike) -> np.ndarray:
        """Extract RGB histogram features from an image."""
        img = super().read_image(input)

        if self.h_type == "global":
            hist = self._extract_global_histogram(img)
        else:  # region
            hist = self._extract_regional_histogram(img)

        if self.normalize and np.sum(hist) > 0:
            hist = hist / np.sum(hist)

        return hist.flatten()

    def _extract_global_histogram(self, img: np.ndarray) -> np.ndarray:
        """Extract histogram from whole image."""
        return self._count_hist(img, self.n_bin)

    def _extract_regional_histogram(self, img: np.ndarray) -> np.ndarray:
        """Extract histograms from image regions."""
        height, width, _ = img.shape
        hist = np.zeros((self.n_slice, self.n_slice, self.n_bin**3))

        h_slices = np.linspace(0, height, self.n_slice + 1, endpoint=True).astype(int)
        w_slices = np.linspace(0, width, self.n_slice + 1, endpoint=True).astype(int)

        for h_idx in range(len(h_slices) - 1):
            for w_idx in range(len(w_slices) - 1):
                img_region = img[
                    h_slices[h_idx]:h_slices[h_idx + 1],
                    w_slices[w_idx]:w_slices[w_idx + 1]
                ]
                hist[h_idx][w_idx] = self._count_hist(img_region, self.n_bin)

        return hist

    def _count_hist(self, input: np.ndarray, n_bin: int, channel: int = 3) -> np.ndarray:
        """Calculate color histogram for an image."""
        return cv2.calcHist(
            [input],
            np.arange(channel),
            None,
            [n_bin, n_bin, n_bin],
            [0, 256, 0, 256, 0, 256],
        ).flatten()