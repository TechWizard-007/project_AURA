
import json
import os

from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

import chromadb

from videoChunker import OUTPUT_DIR


# Load environment variables
load_dotenv()


# Configuration
VECTOR_DB_DIR = Path(
    os.getenv("AURA_VECTOR_DB_DIR", "data/vector_db")
)

COLLECTION_NAME = os.getenv(
    "AURA_AUDIO_COLLECTION",
    "aura_audio"
)

EMBEDDING_DIMENSIONS = int(
    os.getenv("AURA_EMBEDDING_DIMENSIONS", "1024")
)


class AudioVectorStore:

    def __init__(self):

        print("\nInitializing AudioVectorStore...")
        print(f"Database location: {VECTOR_DB_DIR}")
        print(f"Collection: {COLLECTION_NAME}")

        # Create persistent ChromaDB client
        VECTOR_DB_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR)
        )

        # Create or retrieve audio collection
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
        with open(
            embedding_json_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        chunk_id = data["chunk_id"]

        embeddings_data = data.get(
            "segment_embeddings",
            []
        )

        if not embeddings_data:
            print(
                f"No audio embeddings found: "
                f"{embedding_json_path}"
            )
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
                    f"{chunk_id}/{item['segment_id']}: "
                    f"{len(embedding)}"
                )

            # Unique ID for every audio segment
            record_id = (
                f"{OUTPUT_DIR.name}_{chunk_id}_"
                f"{item['segment_id']}"
            )

            ids.append(record_id)

            embeddings.append(embedding)

            documents.append(item["text"])

            metadatas.append({

                "chunk_id": chunk_id,

                "segment_id": item["segment_id"],

                "start": float(item["start"]),
                "end": float(item["end"]),

                "global_start": float(item["global_start"]),
                "global_end": float(item["global_end"]),

                "language": data["language"],

                "embedding_model": data["embedding_model"]

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

        # Locate all generated audio embedding files
        embedding_files = sorted(
            (OUTPUT_DIR / "extracted").glob(
                "chunk_*/audio_embeddings.json"
            )
        )

        if not embedding_files:
            raise FileNotFoundError(
                "No audio_embeddings.json files found."
            )

        print(f"\nFound {len(embedding_files)} chunks.")

        total_inserted = 0

        for embedding_file in tqdm(
            embedding_files,
            desc="Storing audio embeddings",
            unit="chunk"
        ):

            inserted = self.process_chunk(embedding_file)

            total_inserted += inserted

            tqdm.write(
                f"Stored {inserted} embeddings from "
                f"{embedding_file.parent.name}"
            )

        print("\nAudio storage completed!")
        print(f"Embeddings processed: {total_inserted}")
        print(
            f"Total records in collection: "
            f"{self.collection.count()}"
        )


def main():

    vector_store = AudioVectorStore()

    vector_store.process_all_chunks()


if __name__ == "__main__":

    main()
