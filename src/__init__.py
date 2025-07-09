"""
CBIR (Content-Based Image Retrieval) Package
===========================================

Package chính cho hệ thống tìm kiếm ảnh dựa trên nội dung với dataset Food 101.

Modules:
    - datasets: Xử lý và load dữ liệu
    - featuring: Trích xuất đặc trưng từ ảnh
    - retrieval: Các thuật toán tìm kiếm và so sánh
    - storage: Lưu trữ và quản lý đặc trưng
    - utils: Các tiện ích hỗ trợ
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from __future__ import annotations

from src.feature_extractor import FeatureExtractor, SingleFeatureExtractor, BatchFeatureExtractor
from src.feature_store import FeatureStore
from src.retrieve import Retrieve
from src.storage import *
from src.featuring import *
from src.retrieval import *
from src.metrics import *