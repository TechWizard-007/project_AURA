import math
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm


# Load environment variables from the project root's .env file.
load_dotenv()


# Configuration
INPUT_VIDEO = os.getenv("AURA_INPUT_VIDEO")
OUTPUT_DIR = Path(os.getenv("AURA_OUTPUT_DIR", "data/processed/v1")).expanduser()
CHUNK_DURATION = float(os.getenv("AURA_CHUNK_DURATION", "10"))

SUPPORTED_VIDEO_FORMATS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".avi",
    ".m4v",
}


def resolve_input_video(video_path: str | Path | None = None) -> Path:
    """
    Resolve and validate the input video.

    Priority:
    1. A path explicitly supplied to the function / --video.
    2. AURA_INPUT_VIDEO from the .env file.
    """

    selected_path = video_path if video_path is not None else INPUT_VIDEO

    if not selected_path:
        raise ValueError(
            "No input video was provided. Supply one with "
            "'--video <path>' or set AURA_INPUT_VIDEO in your .env file.\n"
            'Example: uv run python main.py --video "data/raw_videos/my_video.mp4"'
        )

    path = Path(selected_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Input video not found: {path}\n"
            "Check the path, or pass the correct file using --video."
        )

    if not path.is_file():
        raise ValueError(f"The input path is not a file: {path}")

    if path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_FORMATS))
        raise ValueError(
            f"Unsupported video format: '{path.suffix or '[no extension]'}'. "
            f"Supported formats: {supported}"
        )

    return path.resolve()


def get_video_duration(video_path: Path) -> float:
    """Retrieve video duration using FFprobe."""

    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFprobe was not found. Install FFmpeg and make sure ffprobe "
            "is available on your system PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or "FFprobe could not read this file."
        raise RuntimeError(f"Could not inspect the input video: {details}") from exc

    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"FFprobe returned an invalid duration for: {video_path}"
        ) from exc

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"The video has an invalid duration: {duration}")

    return duration


def chunk_video(video_path: str | Path | None = None) -> None:
    """Split the selected input video into fixed-duration MP4 chunks."""

    input_video = resolve_input_video(video_path)

    if CHUNK_DURATION <= 0:
        raise ValueError("AURA_CHUNK_DURATION must be greater than zero.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_duration = get_video_duration(input_video)
    total_chunks = math.ceil(total_duration / CHUNK_DURATION)

    print(f"\nInput video: {input_video}")
    print(f"Output directory: {OUTPUT_DIR}")
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
            "-i", str(input_video),
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_file),
        ]

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "FFmpeg was not found. Install FFmpeg and make sure ffmpeg "
                "is available on your system PATH."
            ) from exc

        tqdm.write(f"Created: {output_file} ({duration:.3f}s)")


if __name__ == "__main__":
    chunk_video()
