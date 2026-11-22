import json
from pathlib import Path

from visualAnalyzer import VisualAnalyzer

# First chunk's frame directory
FRAME_DIR = Path(
    "data/processed/v1/extracted/chunk_001/frames"
)

OUTPUT_PATH = Path(
    "data/processed/v1/extracted/chunk_001/visual_analysis.json"
)


def main():

    frames = []

    for image_path in sorted(FRAME_DIR.glob("*.jpg")):

        frame_number = int(image_path.stem.split("_")[1])

        frames.append({
            "frame_path": image_path,
            "local_timestamp": float(frame_number),
            "global_timestamp": float(frame_number)
        })

    if not frames:
        raise FileNotFoundError("No frames found!")

    analyzer = VisualAnalyzer()

    result = analyzer.analyze_frames(
        frames=frames,
        output_path=OUTPUT_PATH,
        chunk_id="chunk_001"
    )

    print("\nTest completed!")
    print(f"Frames analyzed: {len(result['individual_segments'])}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()