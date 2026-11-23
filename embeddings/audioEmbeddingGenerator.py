
import json

from pathlib import Path
from tqdm import tqdm

from processing.videoChunker import OUTPUT_DIR
from embeddings.embeddingGenerator import EmbeddingGenerator


class AudioEmbeddingGenerator:

    def __init__(self):

        print("\nInitializing AudioEmbeddingGenerator...")

        # Reuse the existing BGE-M3 embedding engine
        self.embedding_generator = EmbeddingGenerator()

    def process_chunk(self, transcription_json_path: Path):

        # Load transcription data
        with open(
            transcription_json_path,
            "r",
            encoding="utf-8"
        ) as file:

            transcription_data = json.load(file)

        segments = transcription_data.get("segments", [])

        if not segments:
            print(
                f"No transcription segments found: "
                f"{transcription_json_path}"
            )
            return None

        # Extract segment text
        texts = [
            segment["text"].strip()
            for segment in segments
        ]

        # Generate embeddings in one batch
        embeddings = self.embedding_generator.generate_embeddings(
            texts
        )

        # Preserve segment information alongside embeddings
        embedded_segments = []

        for index, (segment, embedding) in enumerate(
            zip(segments, embeddings)
        ):

            embedded_segments.append({

                "segment_id": f"segment_{index + 1:03d}",

                "start": segment["start"],
                "end": segment["end"],

                "global_start": segment["global_start"],
                "global_end": segment["global_end"],

                "text": segment["text"],

                "embedding": embedding

            })

        # Prepare output
        output_data = {

            "chunk_id": transcription_data["chunk_id"],

            "audio_path": transcription_data["audio_path"],

            "language": transcription_data["language"],

            "model": self.embedding_generator.__class__.__name__,

            "embedding_model": "bge-m3:latest",

            "dimensions": 1024,

            "total_embeddings": len(embedded_segments),

            "segment_embeddings": embedded_segments

        }

        # Save embeddings beside transcription JSON
        output_path = (
            transcription_json_path.parent
            / "audio_embeddings.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                output_data,
                file,
                indent=4,
                ensure_ascii=False
            )

        return output_path

    def process_all_chunks(self):

        # Locate all transcription files
        transcription_files = sorted(
            (OUTPUT_DIR / "extracted").glob(
                "chunk_*/transcription.json"
            )
        )

        if not transcription_files:
            raise FileNotFoundError(
                "No transcription.json files found."
            )

        print(
            f"\nFound {len(transcription_files)} chunks."
        )

        for transcription_file in tqdm(
            transcription_files,
            desc="Generating audio embeddings",
            unit="chunk"
        ):

            output_path = self.process_chunk(
                transcription_file
            )

            if output_path:

                tqdm.write(
                    f"Saved embeddings: {output_path}"
                )

        print("\nAudio embedding generation completed!")


def main():

    generator = AudioEmbeddingGenerator()

    generator.process_all_chunks()


if __name__ == "__main__":

    main()
