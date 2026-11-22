
from pathlib import Path

from visualAnalyzer import VisualAnalyzer


# Configuration
CHUNK_ID = "chunk_001"
FPS = 1.0

FRAMES_DIR = Path(
    "data/processed/v1/extracted/chunk_001/frames"
)

OUTPUT_PATH = Path(
    "data/processed/v1/extracted/chunk_001/visual_analysis.json"
)


def main():

    frame_paths = sorted(
        FRAMES_DIR.glob("frame_*.jpg")
    )

    if not frame_paths:
        raise FileNotFoundError(
            f"No frames found in {FRAMES_DIR}"
        )

    frames = []

    for index, frame_path in enumerate(frame_paths):

        timestamp = index / FPS

        frames.append({
            "frame_path": frame_path,
            "local_timestamp": round(timestamp, 3),
            "global_timestamp": round(timestamp, 3)
        })

    analyzer = VisualAnalyzer()

    print("\nTesting temporal analysis only...")

    result = analyzer.analyze_temporal_group(
        frames[5:10]
    )

    print("\nTemporal description:")
    print(result["description"])

    print("\nPerformance metrics:")
    print(result["ollama_metrics"])


if __name__ == "__main__":
    main()
