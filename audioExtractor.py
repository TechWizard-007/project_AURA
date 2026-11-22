
import subprocess
from pathlib import Path


def extract_audio(
    video_path: Path,
    output_path: Path
) -> Path:
    """
    Extract audio from a video chunk.

    Output specifications:
    - WAV format
    - 16 kHz sampling rate
    - Mono channel
    - PCM 16-bit encoding
    """

    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video chunk not found: {video_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),

        # Audio extraction
        "-map", "0:a:0",
        "-vn",

        # Audio specifications
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",

        str(output_path)
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL
    )

    return output_path


if __name__ == "__main__":

    test_chunk = Path(
        "data/processed/v1/chunk_001.mp4"
    )

    output_audio = Path(
        "data/processed/v1/extracted/chunk_001/audio.wav"
    )

    result = extract_audio(
        test_chunk,
        output_audio
    )

    print(f"Audio extracted successfully: {result}")
