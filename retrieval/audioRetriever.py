
import os

from pathlib import Path
from dotenv import load_dotenv

import chromadb

from embeddings.embeddingGenerator import EmbeddingGenerator


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


class AudioRetriever:

    def __init__(self):

        print("\nInitializing AURA Audio Retriever...")

        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()

        # Connect to persistent ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR)
        )

        # Access existing audio collection
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=None
        )

        print("Audio Retriever initialized successfully!")
        print(f"Available audio records: {self.collection.count()}")

    def search(self, query: str, n_results: int = 5):

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        total_records = self.collection.count()

        if total_records == 0:
            print("The audio collection is empty.")
            return []

        # Prevent requesting more results than available
        n_results = min(n_results, total_records)

        # Generate query embedding using BGE-M3
        query_embedding = self.embedding_generator.generate_embeddings(
            [query]
        )[0]

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        # Extract and organize results
        retrieved_results = []

        for index in range(len(results["ids"][0])):

            distance = results["distances"][0][index]

            result = {
                "id": results["ids"][0][index],
                "text": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "distance": distance,
                "similarity": 1 - distance
            }

            retrieved_results.append(result)

        return retrieved_results

    def display_results(self, results):

        if not results:
            print("\nNo matching audio segments found.")
            return

        print(f"\nFound {len(results)} matching audio segments:")
        print("=" * 65)

        for rank, result in enumerate(results, start=1):

            metadata = result["metadata"]

            print(f"\nResult #{rank}")

            print(f"Chunk: {metadata['chunk_id']}")
            print(f"Segment: {metadata['segment_id']}")

            print(
                f"Timestamp: "
                f"{metadata['global_start']:.2f}s - "
                f"{metadata['global_end']:.2f}s"
            )

            print(f"Language: {metadata['language']}")

            print(
                f"Cosine similarity: "
                f"{result['similarity']:.4f}"
            )

            print(f"Transcript: {result['text']}")

            print("-" * 65)


def main():

    retriever = AudioRetriever()

    print("\nAURA Audio Search")
    print("Type 'exit' to quit.")

    while True:

        query = input("\nWhat are you looking for? ").strip()

        if query.lower() in {"exit", "quit"}:
            print("\nClosing AURA Audio Search.")
            break

        if not query:
            continue

        results = retriever.search(
            query=query,
            n_results=5
        )

        retriever.display_results(results)


if __name__ == "__main__":

    main()
