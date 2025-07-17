import os
import uuid
from tqdm import tqdm

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


def main():
    print("Tải mô hình và image transform...")
    feature_extractor_model = get_feature_extractor()
    image_transform = get_image_transform()
    
    print("Kết nối tới ChromaDB...")
    db_store = ChromaDBStore(db_path=DB_PATH, collection_name=COLLECTION_NAME)

    index_images(db_store, feature_extractor_model, image_transform)

if __name__ == "__main__":
    main() 