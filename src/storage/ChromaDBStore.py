import chromadb
import numpy as np
from src.feature_store import FeatureStore, ImageSearchObject

class ChromaDBStore(FeatureStore):
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "image_features"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine is good for CNN features
        )

    def add(self, metadatas: list[dict], features: list[np.ndarray], ids: list[str]) -> None:
        """
        Add features and their metadatas to the ChromaDB collection.
        IDs are expected to be unique for each item.
        """
        if not features:
            print("No features to add.")
            return

        # ChromaDB expects a list of lists/np.ndarrays for embeddings
        embeddings = [f.tolist() for f in features]

        self.collection.upsert(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, feature: np.ndarray, k: int = 5) -> list[ImageSearchObject]:
        """
        Find k most similar items to query feature in ChromaDB.
        """
        if feature is None:
            print("Cannot search with a null feature vector.")
            return []


        query_embedding = feature.tolist() if isinstance(feature, np.ndarray) else feature

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["metadatas", "distances"]
        )

        search_results = []
        if not results['ids'][0]:
            return []

        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]

            search_results.append(ImageSearchObject(index=doc_id, score=distance, image=metadata.get('filepath')))
            
        return search_results

    def count(self) -> int:
        """Returns the number of items in the collection."""
        return self.collection.count() 