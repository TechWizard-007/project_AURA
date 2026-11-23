
import os
import math
import subprocess

from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm


# Load environment variables
load_dotenv()


# Configuration
INPUT_VIDEO = Path(os.getenv("AURA_INPUT_VIDEO"))
OUTPUT_DIR = Path(os.getenv("AURA_OUTPUT_DIR"))
CHUNK_DURATION = float(os.getenv("AURA_CHUNK_DURATION"))


def get_video_duration(video_path: Path) -> float:
    """Retrieve video duration using FFprobe."""

    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    return float(result.stdout.strip())


def chunk_video():

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"Video not found: {INPUT_VIDEO}")

    if CHUNK_DURATION <= 0:
        raise ValueError("Chunk duration must be greater than zero.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_duration = get_video_duration(INPUT_VIDEO)

    total_chunks = math.ceil(total_duration / CHUNK_DURATION)

    print(f"\nInput video: {INPUT_VIDEO}")
    print(f"Video duration: {total_duration:.3f} seconds")
    print(f"Chunk duration: {CHUNK_DURATION} seconds")
    print(f"Total chunks: {total_chunks}\n")

    for i in tqdm(range(total_chunks), desc="Processing video", unit="chunk"):

        start = i * CHUNK_DURATION
        duration = min(CHUNK_DURATION, total_duration - start)

        output_file = OUTPUT_DIR / f"chunk_{i + 1:03d}.mp4"

        command = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", str(INPUT_VIDEO),
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_file)
        ]

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL
        )

        tqdm.write(
            f"Created: {output_file} ({duration:.3f}s)"
        )


if __name__ == "__main__":
    chunk_video()
