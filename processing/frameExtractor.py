
import subprocess
from pathlib import Path


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 1.0,
    chunk_start_seconds: float = 0.0
) -> list[dict]:
    """
    Extract representative frames from a video chunk.

    Args:
        video_path: Input video chunk.
        output_dir: Directory to store extracted frames.
        fps: Number of frames extracted per second.
        chunk_start_seconds: Global starting timestamp of the chunk.

    Returns:
        List containing frame paths and global timestamps.
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video chunk not found: {video_path}"
        )

    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_pattern = output_dir / "frame_%03d.jpg"

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),

        # Extract frames at the specified FPS
        "-vf", f"fps={fps}",

        # JPEG quality
        "-q:v", "2",

        # Sequential filenames starting from zero
        "-start_number", "0",

        str(output_pattern)
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    frames = sorted(output_dir.glob("frame_*.jpg"))

    if not frames:
        raise RuntimeError(
            f"No frames extracted from: {video_path}"
        )

    frame_metadata = []

    for index, frame_path in enumerate(frames):

        local_timestamp = index / fps

        global_timestamp = (
            chunk_start_seconds + local_timestamp
        )

        frame_metadata.append({
            "frame_path": frame_path,
            "local_timestamp": round(local_timestamp, 3),
            "global_timestamp": round(global_timestamp, 3)
        })

    return frame_metadata


if __name__ == "__main__":

    test_chunk = Path(
        "data/processed/v1/chunk_001.mp4"
    )

    output_frames = Path(
        "data/processed/v1/extracted/chunk_001/frames"
    )

    frames = extract_frames(
        video_path=test_chunk,
        output_dir=output_frames,
        fps=1.0,
        chunk_start_seconds=0.0
    )

    print(f"\nExtracted {len(frames)} frames:\n")

    for frame in frames:
        print(
            f"{frame['frame_path']} | "
            f"{frame['global_timestamp']} seconds"
        )
