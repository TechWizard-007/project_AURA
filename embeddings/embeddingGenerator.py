
import json
import os
import requests

from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from processing.videoChunker import OUTPUT_DIR


# Load environment variables
load_dotenv()


# Configuration
OLLAMA_BASE_URL = os.getenv(
    "AURA_OLLAMA_BASE_URL",
    "http://localhost:11434"
)

EMBEDDING_MODEL = os.getenv(
    "AURA_EMBEDDING_MODEL",
    "bge-m3:latest"
)

EMBEDDING_DIMENSIONS = int(
    os.getenv("AURA_EMBEDDING_DIMENSIONS", "1024")
)

EMBEDDING_TIMEOUT = int(
    os.getenv("AURA_EMBEDDING_TIMEOUT", "300")
)


class EmbeddingGenerator:

    def __init__(self):

        self.session = requests.Session()

        self.endpoint = (
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed"
        )

        print("\nInitializing EmbeddingGenerator...")
        print(f"Model: {EMBEDDING_MODEL}")
        print(f"Endpoint: {self.endpoint}")

        self._check_connection()

    def _check_connection(self):

        try:
            response = self.session.get(
                f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags",
                timeout=10
            )

            response.raise_for_status()

        except requests.RequestException as error:
            raise ConnectionError(
                f"Could not connect to Ollama: {error}"
            ) from error

        print("Ollama connection successful!")

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:

        if not texts:
            return []

        payload = {
            "model": EMBEDDING_MODEL,
            "input": texts,
            "keep_alive": "10m"
        }

        try:
            response = self.session.post(
                self.endpoint,
                json=payload,
                timeout=EMBEDDING_TIMEOUT
            )

            response.raise_for_status()

        except requests.RequestException as error:
            raise RuntimeError(
                f"Embedding generation failed: {error}"
            ) from error

        result = response.json()
        embeddings = result.get("embeddings")

        if not embeddings:
            raise ValueError(
                "Ollama returned an empty embedding response."
            )

        if len(embeddings) != len(texts):
            raise ValueError(
                "Mismatch between input texts and generated embeddings."
            )

        for index, embedding in enumerate(embeddings):

            if len(embedding) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"Unexpected embedding dimension at index {index}: "
                    f"{len(embedding)}"
                )

        return embeddings

    def process_chunk(self, visual_json_path: Path):

        # Load visual analysis
        with open(visual_json_path, "r", encoding="utf-8") as file:
            visual_data = json.load(file)

        individual_segments = visual_data.get(
            "individual_segments", []
        )

        if not individual_segments:
            print(f"No individual descriptions found: {visual_json_path}")
            return

        # Extract descriptions
        descriptions = [
            segment["description"]
            for segment in individual_segments
        ]

        # Generate embeddings in one batch
        embeddings = self.generate_embeddings(descriptions)

        # Preserve frame information alongside each vector
        embedded_segments = []

        for segment, embedding in zip(
            individual_segments,
            embeddings
        ):

            embedded_segments.append({
                "frame_id": segment["frame_id"],
                "frame_path": segment["frame_path"],
                "local_timestamp": segment["local_timestamp"],
                "global_timestamp": segment["global_timestamp"],
                "description": segment["description"],
                "embedding": embedding
            })

        # Prepare output
        output_data = {
            "chunk_id": visual_data["chunk_id"],
            "model": EMBEDDING_MODEL,
            "dimensions": EMBEDDING_DIMENSIONS,
            "total_embeddings": len(embedded_segments),
            "individual_embeddings": embedded_segments
        }

        # Save embeddings beside the source JSON
        output_path = (
            visual_json_path.parent / "visual_embeddings.json"
        )

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(
                output_data,
                file,
                indent=4
            )

        return output_path

    def process_all_chunks(self):

        # Find all visual analysis files
        visual_files = sorted(
            (OUTPUT_DIR / "extracted").glob(
                "chunk_*/visual_analysis.json"
            )
        )

        if not visual_files:
            raise FileNotFoundError(
                "No visual_analysis.json files found."
            )

        print(f"\nFound {len(visual_files)} chunks.")

        for visual_file in tqdm(
            visual_files,
            desc="Generating embeddings",
            unit="chunk"
        ):

            output_path = self.process_chunk(visual_file)

            if output_path:
                tqdm.write(
                    f"Saved embeddings: {output_path}"
                )

        print("\nVisual embedding generation completed!")


def main():

    generator = EmbeddingGenerator()
    generator.process_all_chunks()


if __name__ == "__main__":
    main()
