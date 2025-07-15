import argparse
import os
import random
import uuid
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from src.datasets.caltech_256 import get_caltech256_datasets
from src.featuring.resnet50 import get_feature_extractor, get_image_transform, extract_features
from src.storage.ChromaDBStore import ChromaDBStore

# --- CONFIGURATION ---
DATASET_PATH = "data/caltech-256/256_ObjectCategories"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "caltech256_resnet50"

def index_images(collection_store, model, transform):
    """
    Duyệt qua dataset, trích xuất feature và lưu vào ChromaDB.
    """
    print(f"\n--- Bắt đầu quá trình Indexing ---")

    if not os.path.exists(DATASET_PATH):
        print(f"Lỗi: Thư mục dataset không tồn tại tại '{DATASET_PATH}'.")
        return

    if collection_store.count() > 0:
        user_input = input(f"Collection '{COLLECTION_NAME}' đã có {collection_store.count()} ảnh. Xóa và index lại? (y/n): ")
        if user_input.lower() != 'y':
            print("Bỏ qua indexing.")
            return
        print(f"Xóa collection cũ '{COLLECTION_NAME}'...")
        collection_store.client.delete_collection(name=COLLECTION_NAME)
        collection_store.collection = collection_store.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    
    train_dataset, _ = get_caltech256_datasets(DATASET_PATH)

    batch_size = 100
    features_batch, metadatas_batch, ids_batch = [], [], []

    total_images = len(train_dataset)
    for i in tqdm(range(total_images), desc="🖼️  Đang trích xuất features"):
        image_path, _ = train_dataset.dataset.samples[train_dataset.indices[i]]
        feature_vector = extract_features(image_path, model, transform)
        
        if feature_vector is not None:
            features_batch.append(feature_vector)
            metadatas_batch.append({"filepath": image_path})
            ids_batch.append(str(uuid.uuid4()))

        if len(features_batch) >= batch_size or i == total_images - 1:
            if features_batch:
                collection_store.add(metadatas=metadatas_batch, features=features_batch, ids=ids_batch)
                features_batch, metadatas_batch, ids_batch = [], [], []

    print(f"✅ Indexing hoàn tất! Collection hiện có {collection_store.count()} ảnh.")

def search_similar_images(query_image_path, collection_store, model, transform, n_results=5):
    if not os.path.exists(query_image_path):
        print(f"Lỗi: File ảnh truy vấn không tồn tại tại '{query_image_path}'")
        return

    print(f"\n--- Tìm kiếm {n_results} ảnh tương đồng với: {os.path.basename(query_image_path)} ---")
    
    query_vector = extract_features(query_image_path, model, transform)
    if query_vector is None:
        return

    results = collection_store.search(feature=query_vector, k=n_results)
    
    print("\n--- 🏆 Kết quả tìm kiếm (dạng văn bản) ---")
    if not results:
        print("Không tìm thấy kết quả nào.")
        return
        
    for i, result in enumerate(results):
        print(f"  - Hạng {i+1}: {result.image} (Khoảng cách: {result.score:.4f})")

    # --- Hiển thị kết quả bằng Matplotlib ---
    try:
        query_img = Image.open(query_image_path).convert("RGB")

        fig, axs = plt.subplots(1, n_results + 1, figsize=(20, 5))
        fig.suptitle('Kết quả tìm kiếm', fontsize=20)

        # Hiển thị ảnh query
        axs[0].imshow(query_img)
        axs[0].set_title("Ảnh truy vấn")
        axs[0].axis('off')

        # Hiển thị các ảnh kết quả
        for i, result in enumerate(results):
            result_img_path = result.image
            if os.path.exists(result_img_path):
                result_img = Image.open(result_img_path).convert("RGB")
                axs[i + 1].imshow(result_img)
                axs[i + 1].set_title(f"Hạng {i+1}\nDist: {result.score:.4f}")
                axs[i + 1].axis('off')
            else:
                print(f"Warning: Không tìm thấy file ảnh kết quả tại {result_img_path}")
                axs[i + 1].text(0.5, 0.5, 'Image not found', ha='center', va='center')
                axs[i + 1].axis('off')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"\n⚠️ Đã xảy ra lỗi khi cố gắng hiển thị ảnh: {e}")


def main():
    parser = argparse.ArgumentParser(description="Hệ thống CBIR dùng ResNet50 và ChromaDB.")
    parser.add_argument("--mode", type=str, required=True, choices=['index', 'search'], help="Chế độ chạy: 'index' để tạo và lưu features, 'search' để tìm ảnh.")
    parser.add_argument("--query_image", type=str, help="Đường dẫn đến ảnh truy vấn (bắt buộc ở chế độ 'search').")
    parser.add_argument("--n_results", type=int, default=5, help="Số lượng kết quả tìm kiếm (chỉ dùng ở chế độ 'search').")
    args = parser.parse_args()

    print("Tải mô hình và image transform...")
    feature_extractor_model = get_feature_extractor()
    image_transform = get_image_transform()
    
    print("Kết nối tới ChromaDB...")
    db_store = ChromaDBStore(db_path=DB_PATH, collection_name=COLLECTION_NAME)

    if args.mode == 'index':
        index_images(db_store, feature_extractor_model, image_transform)
    elif args.mode == 'search':
        if not args.query_image:
            print("Lỗi: Đối số --query_image là bắt buộc khi chạy ở chế độ 'search'.")
            parser.print_help()
            return

        if db_store.count() == 0:
            print("Lỗi: Cơ sở dữ liệu trống. Vui lòng chạy chế độ 'index' trước.")
            return
        search_similar_images(args.query_image, db_store, feature_extractor_model, image_transform, args.n_results)

if __name__ == "__main__":
    main() 