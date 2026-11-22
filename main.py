
from pathlib import Path
from tqdm import tqdm

from videoChunker import (
    chunk_video,
    OUTPUT_DIR,
    CHUNK_DURATION
)

from audioExtractor import extract_audio
from frameExtractor import extract_frames


# Extraction configuration
FRAME_FPS = 1.0


def process_chunks():

    # Discover all generated video chunks
    chunks = sorted(OUTPUT_DIR.glob("chunk_*.mp4"))

    if not chunks:
        raise FileNotFoundError(
            f"No video chunks found in {OUTPUT_DIR}"
        )

    print(f"\nFound {len(chunks)} video chunks.\n")

    for chunk in tqdm(
        chunks,
        desc="Extracting media",
        unit="chunk"
    ):

        chunk_number = int(
            chunk.stem.split("_")[-1]
        )

        # Calculate original video timestamp
        chunk_start = (
            (chunk_number - 1) * CHUNK_DURATION
        )

        # Create output directory
        chunk_output_dir = (
            OUTPUT_DIR / "extracted" / chunk.stem
        )

        chunk_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Extract audio
        audio_path = chunk_output_dir / "audio.wav"

        extract_audio(
            video_path=chunk,
            output_path=audio_path
        )

        # Extract representative frames
        frames_dir = chunk_output_dir / "frames"

        frames = extract_frames(
            video_path=chunk,
            output_dir=frames_dir,
            fps=FRAME_FPS,
            chunk_start_seconds=chunk_start
        )

        tqdm.write(
            f"{chunk.name}: "
            f"Audio extracted | "
            f"{len(frames)} frames extracted"
        )

    print("\nAURA preprocessing completed!")


def main():

    print("\n========== PROJECT AURA ==========")

    # Step 1: Chunk the original video
    chunk_video()

    # Step 2: Process every generated chunk
    process_chunks()


if __name__ == "__main__":
    main()
