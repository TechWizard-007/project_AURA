
from pathlib import Path
from tqdm import tqdm

from videoChunker import (
    chunk_video,
    OUTPUT_DIR,
    CHUNK_DURATION
)

from audioExtractor import extract_audio
from audioTranscriber import AudioTranscriber
from frameExtractor import extract_frames
from visualAnalyzer import VisualAnalyzer

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

    # Initialize transcription model once
    transcriber = AudioTranscriber()
    visual_analyzer = VisualAnalyzer()

    for chunk in tqdm(
        chunks,
        desc="Processing chunks",
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

        # Step 1: Extract audio
        audio_path = chunk_output_dir / "audio.wav"

        extract_audio(
            video_path=chunk,
            output_path=audio_path
        )

        # Step 2: Transcribe audio
        transcription_path = (
            chunk_output_dir / "transcription.json"
        )

        transcription = transcriber.transcribe(
            audio_path=audio_path,
            output_path=transcription_path,
            chunk_start_seconds=chunk_start
        )

        # Step 3: Extract representative frames
        frames_dir = chunk_output_dir / "frames"

        frames = extract_frames(
            video_path=chunk,
            output_dir=frames_dir,
            fps=FRAME_FPS,
            chunk_start_seconds=chunk_start
        )
        visual_output_path = chunk_output_dir / "visual_analysis.json"

        visual_analyzer.analyze_frames(
            frames=frames,
            output_path=visual_output_path,
            chunk_id=chunk.stem
        )

        tqdm.write(
            f"{chunk.name}: "
            f"Audio extracted | "
            f"{len(transcription['segments'])} speech segments | "
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
