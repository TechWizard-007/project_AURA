
import json
import os

from pathlib import Path

import torch
import whisper

from dotenv import load_dotenv


# Load environment configuration
load_dotenv()


# Transcription configuration
MODEL_PATH = Path(
    os.getenv(
        "AURA_WHISPER_MODEL_PATH",
        str(Path.home() / ".cache" / "whisper" / "large-v3-turbo.pt")
    )
)

DEVICE = os.getenv("AURA_WHISPER_DEVICE", "cuda").lower()
BEAM_SIZE = int(os.getenv("AURA_WHISPER_BEAM_SIZE", "5"))

LANGUAGE = (
    os.getenv("AURA_WHISPER_LANGUAGE", "").strip() or None
)


class AudioTranscriber:

    def __init__(self):

        print(f"\nLoading Whisper model: {MODEL_PATH.name}")

        # Validate GPU availability
        if DEVICE.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Cannot load Whisper on GPU."
            )

        # Validate model checkpoint
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Whisper checkpoint not found: {MODEL_PATH}"
            )

        # Load model once
        self.model = whisper.load_model(
            str(MODEL_PATH),
            device=DEVICE
        )

        self.fp16 = DEVICE.startswith("cuda")

        print(f"Whisper model loaded successfully on {DEVICE}!")

    def transcribe(
        self,
        audio_path: Path,
        output_path: Path,
        chunk_start_seconds: float = 0.0
    ) -> dict:

        """
        Transcribe an audio chunk and generate timestamps.

        Args:
            audio_path: Path to the extracted WAV file.
            output_path: Path to save the transcription JSON.
            chunk_start_seconds: Global starting timestamp of the chunk.

        Returns:
            Dictionary containing transcription and metadata.
        """

        audio_path = Path(audio_path)
        output_path = Path(output_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        if chunk_start_seconds < 0:
            raise ValueError(
                "Chunk start timestamp cannot be negative."
            )

        # Run speech recognition
        transcription = self.model.transcribe(
            str(audio_path),
            language=LANGUAGE,
            task="transcribe",
            beam_size=BEAM_SIZE,
            temperature=0,
            fp16=self.fp16,
            verbose=False
        )

        transcription_segments = []

        # Process transcription segments
        for segment in transcription["segments"]:

            local_start = round(segment["start"], 3)
            local_end = round(segment["end"], 3)

            transcription_segments.append({
                "start": local_start,
                "end": local_end,
                "global_start": round(
                    chunk_start_seconds + local_start, 3
                ),
                "global_end": round(
                    chunk_start_seconds + local_end, 3
                ),
                "text": segment["text"].strip()
            })

        # Build structured output
        result = {
            "chunk_id": audio_path.parent.name,
            "audio_path": str(audio_path),
            "model": MODEL_PATH.name,
            "device": DEVICE,
            "language": transcription["language"],
            "text": transcription["text"].strip(),
            "segments": transcription_segments
        }

        # Create output directory if necessary
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save transcription
        with output_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                indent=4,
                ensure_ascii=False
            )

        return result


if __name__ == "__main__":

    transcriber = AudioTranscriber()

    test_audio = Path(
    "data/processed/v1/extracted/chunk_002/audio.wav"
    )

    output_json = Path(
        "data/processed/v1/extracted/chunk_002/transcription.json"
    )

    result = transcriber.transcribe(
        audio_path=test_audio,
        output_path=output_json,
        chunk_start_seconds=10.0
    )

    print("\nTranscription completed!")
    print(f"Detected language: {result['language']}")
    print(f"Segments generated: {len(result['segments'])}")
    print(f"Saved to: {output_json}")
