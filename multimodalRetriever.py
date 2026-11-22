
from retriever import Retriever
from audioRetriever import AudioRetriever


# Reciprocal Rank Fusion configuration
RRF_K = 60


class MultimodalRetriever:

    def __init__(self):

        print("\nInitializing AURA Multimodal Retriever...")

        # Initialize independent retrievers
        self.visual_retriever = Retriever()
        self.audio_retriever = AudioRetriever()

        print("\nMultimodal retriever initialized successfully!")

    def fuse_results(
        self,
        visual_results: list[dict],
        audio_results: list[dict],
        top_k: int = 10
    ) -> list[dict]:

        fused_results = []

        # Process visual retrieval results
        for rank, result in enumerate(visual_results, start=1):

            rrf_score = 1 / (RRF_K + rank)

            fused_results.append({
                "id": result["id"],
                "modality": "visual",
                "content": result["description"],
                "metadata": result["metadata"],
                "similarity": result["similarity"],
                "source_rank": rank,
                "rrf_score": rrf_score
            })

        # Process audio retrieval results
        for rank, result in enumerate(audio_results, start=1):

            rrf_score = 1 / (RRF_K + rank)

            fused_results.append({
                "id": result["id"],
                "modality": "audio",
                "content": result["text"],
                "metadata": result["metadata"],
                "similarity": result["similarity"],
                "source_rank": rank,
                "rrf_score": rrf_score
            })

        # Sort all results by their RRF scores
        fused_results.sort(
            key=lambda result: result["rrf_score"],
            reverse=True
        )

        return fused_results[:top_k]

    def search(
        self,
        query: str,
        n_results: int = 5,
        top_k: int = 10
    ) -> list[dict]:

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        print("\nSearching visual collection...")
        visual_results = self.visual_retriever.search(
            query=query,
            n_results=n_results
        )

        print("\nSearching audio collection...")
        audio_results = self.audio_retriever.search(
            query=query,
            n_results=n_results
        )

        # Combine both ranked result lists
        fused_results = self.fuse_results(
            visual_results=visual_results,
            audio_results=audio_results,
            top_k=top_k
        )

        return fused_results

    @staticmethod
    def display_results(results: list[dict]):

        if not results:
            print("\nNo matching results found.")
            return

        print(f"\nFound {len(results)} multimodal results:")
        print("=" * 70)

        for rank, result in enumerate(results, start=1):

            metadata = result["metadata"]

            print(f"\nResult #{rank}")
            print(f"Modality: {result['modality'].upper()}")
            print(f"Chunk: {metadata['chunk_id']}")

            # Display modality-specific timestamps
            if result["modality"] == "visual":

                timestamp = metadata["global_timestamp"]

                print(f"Timestamp: {timestamp:.2f}s")
                print(f"Frame: {metadata['frame_id']}")

            else:

                start = metadata["global_start"]
                end = metadata["global_end"]

                print(f"Timestamp: {start:.2f}s - {end:.2f}s")
                print(f"Segment: {metadata['segment_id']}")

            print(f"Source rank: {result['source_rank']}")

            print(
                f"Original similarity: "
                f"{result['similarity']:.4f}"
            )

            print(
                f"RRF score: "
                f"{result['rrf_score']:.6f}"
            )

            print(f"Content: {result['content']}")

            print("-" * 70)


def main():

    retriever = MultimodalRetriever()

    print("\nAURA Multimodal Search")
    print("Type 'exit' to quit.")

    while True:

        query = input("\nWhat are you looking for? ").strip()

        if query.lower() in {"exit", "quit"}:
            print("\nClosing AURA Multimodal Search.")
            break

        if not query:
            continue

        results = retriever.search(
            query=query,
            n_results=5,
            top_k=10
        )

        retriever.display_results(results)


if __name__ == "__main__":

    main()
