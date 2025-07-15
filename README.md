# Content-Based Image Retrieval with Caltech-256 Dataset

A Content-Based Image Retrieval (CBIR) system using ResNet-50 and ChromaDB on the Caltech-256 dataset.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory-name>
    ```

2.  **Install dependencies:**
    It is recommended to create a virtual environment first.
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Setup

1.  Download the Caltech-256 dataset from the official source: [256_ObjectCategories.tar](http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar).

2.  Create a `data` directory in the project root.

3.  Extract the downloaded archive. The final path to the image folders should be:
    `./data/caltech-256/256_ObjectCategories/`

## Usage

The primary script `run_resnet50_cbir.py` operates in two modes: `index` and `search`.

### 1. Indexing

Before searching, you must index the image dataset. This process extracts features from all images and populates the ChromaDB vector store.

```bash
python run_resnet50_cbir.py --mode index
```

This command creates a `chroma_db` directory in the project root, which stores the feature vectors.

### 2. Searching

After indexing is complete, you can perform a similarity search using a specific query image. The `--query_image` argument is required for this mode.

-   **Example:**
    ```bash
    python run_resnet50_cbir.py --mode search --query_image "/path/to/your/image.jpg"
    ```

-   **To specify the number of results:**
    Use the `--n_results` argument to control how many similar images are returned.
    ```bash
    python run_resnet50_cbir.py --mode search --query_image "/path/to/your/image.jpg" --n_results 10
    ```

Search results, including the query image and its closest matches, will be displayed in a Matplotlib window.

