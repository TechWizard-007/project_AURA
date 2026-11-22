
import os

from pathlib import Path
from dotenv import load_dotenv

import chromadb

from embeddingGenerator import EmbeddingGenerator
from temporalGrouper import TemporalGrouper


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


class Retriever:

    def __init__(self):

        print("\nInitializing AURA Retriever...")

        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()

        # Connect to persistent ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR)
        )

        # Access existing collection
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=None
        )

        # Initialize temporal grouping
        self.temporal_grouper = TemporalGrouper(
            max_gap_seconds=2.0
        )

        print("Retriever initialized successfully!")
        print(f"Available visual records: {self.collection.count()}")

    def search_moments(
        self,
        query: str,
        n_candidates: int = 20,
        top_moments: int = 5
    ):

        # Retrieve a wider set of relevant frames
        frames = self.search(
            query=query,
            n_results=n_candidates
        )

        # Group frames into temporal moments
        moments = self.temporal_grouper.group_frames(frames)

        # Return the highest-ranked moments
        return moments[:top_moments]

    def search(self, query: str, n_results: int = 5):

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        total_records = self.collection.count()

        if total_records == 0:
            print("The visual collection is empty.")
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
                "description": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "distance": distance,
                "similarity": 1 - distance
            }

            retrieved_results.append(result)

        return retrieved_results

    def display_results(self, results):

        if not results:
            print("\nNo matching results found.")
            return

        print(f"\nFound {len(results)} matching frames:")
        print("=" * 60)

        for rank, result in enumerate(results, start=1):

            metadata = result["metadata"]

            print(f"\nResult #{rank}")
            print(f"Chunk: {metadata['chunk_id']}")
            print(f"Frame: {metadata['frame_id']}")

            print(
                f"Timestamp: "
                f"{metadata['global_timestamp']:.2f} seconds"
            )

            print(
                f"Cosine similarity: "
                f"{result['similarity']:.4f}"
            )

            print(f"Description: {result['description']}")

            print("-" * 60)


def main():

    retriever = Retriever()

    print("\nAURA Visual Search")
    print("Type 'exit' to quit.")

    while True:

        query = input("\nWhat are you looking for? ").strip()

        if query.lower() in {"exit", "quit"}:
            print("\nClosing AURA Visual Search.")
            break

        if not query:
            continue

        # Perform temporal-aware search
        moments = retriever.search_moments(
            query=query,
            n_candidates=20,
            top_moments=5
        )

        # Display grouped moments
        retriever.temporal_grouper.display_moments(moments)


if __name__ == "__main__":
    main()
