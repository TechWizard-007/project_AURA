
import json
import os

from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

import chromadb

from processing.videoChunker import OUTPUT_DIR


# Load environment variables
load_dotenv()


# Configuration
VECTOR_DB_DIR = Path(
    os.getenv("AURA_VECTOR_DB_DIR", "data/vector_db")
)

COLLECTION_NAME = os.getenv(
    "AURA_VISUAL_COLLECTION",
    "aura_visual"
)

EMBEDDING_DIMENSIONS = 1024


class VectorStore:

    def __init__(self):

        print("\nInitializing VectorStore...")
        print(f"Database location: {VECTOR_DB_DIR}")
        print(f"Collection: {COLLECTION_NAME}")

        # Create persistent ChromaDB client
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR)
        )

        # Create or retrieve visual collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            metadata={
                "hnsw:space": "cosine"
            }
        )

        print("ChromaDB connection successful!")
        print(f"Existing records: {self.collection.count()}")

    def process_chunk(self, embedding_json_path: Path):

        # Load generated embeddings
        with open(embedding_json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        chunk_id = data["chunk_id"]
        embeddings_data = data.get("individual_embeddings", [])

        if not embeddings_data:
            print(f"No embeddings found: {embedding_json_path}")
            return 0

        # Prepare ChromaDB records
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for item in embeddings_data:

            embedding = item["embedding"]

            if len(embedding) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"Invalid embedding dimensions in "
                    f"{chunk_id}/{item['frame_id']}: "
                    f"{len(embedding)}"
                )

            # Unique ID for every frame
            record_id = (
                f"{OUTPUT_DIR.name}_{chunk_id}_{item['frame_id']}"
            )

            ids.append(record_id)

            embeddings.append(embedding)

            documents.append(
                item["description"]
            )

            metadatas.append({
                "chunk_id": chunk_id,
                "frame_id": item["frame_id"],
                "frame_path": item["frame_path"],
                "local_timestamp": float(item["local_timestamp"]),
                "global_timestamp": float(item["global_timestamp"]),
                "embedding_model": data["model"]
            })

        # Insert or update records
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        return len(ids)

    def process_all_chunks(self):

        # Locate all generated embedding files
        embedding_files = sorted(
            (OUTPUT_DIR / "extracted").glob(
                "chunk_*/visual_embeddings.json"
            )
        )

        if not embedding_files:
            raise FileNotFoundError(
                "No visual_embeddings.json files found."
            )

        print(f"\nFound {len(embedding_files)} chunks.")

        total_inserted = 0

        for embedding_file in tqdm(
            embedding_files,
            desc="Storing visual embeddings",
            unit="chunk"
        ):

            inserted = self.process_chunk(embedding_file)

            total_inserted += inserted

            tqdm.write(
                f"Stored {inserted} embeddings from "
                f"{embedding_file.parent.name}"
            )

        print("\nVisual storage completed!")
        print(f"Embeddings processed: {total_inserted}")
        print(f"Total records in collection: {self.collection.count()}")


def main():

    vector_store = VectorStore()
    vector_store.process_all_chunks()


if __name__ == "__main__":
    main()
