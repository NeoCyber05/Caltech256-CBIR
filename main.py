import os
import uuid
from tqdm import tqdm
from PIL import Image
import io

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import torch

from src.datasets.caltech_256 import get_caltech256_datasets
from src.featuring.resnet50 import get_feature_extractor, get_image_transform, extract_features
from src.storage.ChromaDBStore import ChromaDBStore

# --- CONFIGURATION ---
DATASET_PATH = "data/caltech-256/256_ObjectCategories"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "caltech256_resnet50"


app = FastAPI()

# --- MOUNT STATIC DIRECTORIES ---
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")

feature_extractor_model = None
image_transform = None
db_store = None

def _index_dataset_if_needed():
    """
    Hàm nội bộ để kiểm tra và thực hiện indexing nếu DB trống.
    Sẽ block cho đến khi hoàn thành.
    """
    global db_store, feature_extractor_model, image_transform
    
    if db_store.count() > 0:
        print(f"--> Database already contains {db_store.count()} images. Skipping indexing.")
        return

    print("\n---> Database is empty. Starting one-time indexing process. This may take a while...")

    if not os.path.exists(DATASET_PATH):
        print(f"FATAL: Dataset directory not found at '{DATASET_PATH}'. Cannot perform indexing.")
        # Hoặc raise một exception để dừng server
        raise RuntimeError(f"Dataset directory not found: {DATASET_PATH}")

    try:
        train_dataset, _ = get_caltech256_datasets(DATASET_PATH)
        batch_size = 100
        features_batch, metadatas_batch, ids_batch = [], [], []

        total_images = len(train_dataset)
        for i in tqdm(range(total_images), desc="🖼️  Extracting features for indexing"):
            image_path, _ = train_dataset.dataset.samples[train_dataset.indices[i]]
            feature_vector = extract_features(image_path, feature_extractor_model, image_transform)

            if feature_vector is not None:
                features_batch.append(feature_vector)
                metadatas_batch.append({"filepath": image_path})
                ids_batch.append(str(uuid.uuid4()))

            if len(features_batch) >= batch_size or i == total_images - 1:
                if features_batch:
                    db_store.add(metadatas=metadatas_batch, features=features_batch, ids=ids_batch)
                    features_batch, metadatas_batch, ids_batch = [], [], []
        
        count = db_store.count()
        print(f"✅ Indexing complete! Collection now has {count} images.")

    except Exception as e:
        print(f"FATAL: An error occurred during automatic indexing: {e}")
        raise e

# --- EVENTS ---
@app.on_event("startup")
def startup_event():
    """
    Tải tài nguyên và tự động index khi server khởi động.
    Lưu ý: Đây là sync function, sẽ block server cho đến khi hoàn tất.
    """
    global feature_extractor_model, image_transform, db_store
    print("--> Loading model and image transform...")
    feature_extractor_model = get_feature_extractor()
    image_transform = get_image_transform()

    print("--> Connecting to ChromaDB...")
    db_store = ChromaDBStore(db_path=DB_PATH, collection_name=COLLECTION_NAME)

    # Tự động index nếu cần
    _index_dataset_if_needed()

    print("--> Startup complete. API is ready.")

# --- API ENDPOINTS ---
@app.get("/", tags=["Demo"])
async def demo():
    """
    Phục vụ trang demo HTML.
    """
    return FileResponse('static/index.html')

@app.post("/search", tags=["Search"])
async def search_similar_images_endpoint(
    query_image: UploadFile = File(...),
    n_results: int = Form(5)
):
    """
    Tìm kiếm các ảnh tương tự với ảnh được tải lên.
    """
    global db_store, feature_extractor_model, image_transform
    
    if db_store.count() == 0:
        raise HTTPException(status_code=503, detail="Database is empty.")

    try:
        # Đọc nội dung file ảnh upload
        image_bytes = await query_image.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Trích xuất feature từ ảnh
        img_tensor = image_transform(image).unsqueeze(0)

        device = next(feature_extractor_model.parameters()).device
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            features = feature_extractor_model(img_tensor)


        query_vector = features.squeeze().cpu().numpy().tolist()
        results = db_store.search(feature=query_vector, k=n_results)

        if not results:
            return JSONResponse(content={"message": "No similar images found.", "results": []})

        search_results = [
            {"image_path": result.image, "distance": result.score}
            for result in results
        ]

        return JSONResponse(content={"message": "Search successful.", "results": search_results})
    
    except Exception as e:
        print(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during search: {str(e)}")

