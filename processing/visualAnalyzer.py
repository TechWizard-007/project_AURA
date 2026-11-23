
import os
import time
import base64
import json
import requests

from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()


# Ollama configuration
OLLAMA_BASE_URL = os.getenv(
    "AURA_OLLAMA_BASE_URL",
    "http://localhost:11434"
).rstrip("/")

VISION_MODEL = os.getenv(
    "AURA_VISION_MODEL",
    "qwen3-vl:8b-instruct"
)

REQUEST_TIMEOUT = int(
    os.getenv("AURA_VISION_TIMEOUT", "300")
)

# Temporal analysis configuration
FRAME_GROUP_SIZE = int(
    os.getenv("AURA_FRAME_GROUP_SIZE", "5")
)

ENABLE_TEMPORAL_ANALYSIS = (
    os.getenv(
        "AURA_ENABLE_TEMPORAL_ANALYSIS",
        "true"
    ).lower() == "true"
)


class VisualAnalyzer:

    def __init__(self):

        self.model = VISION_MODEL

        self.generate_url = (
            f"{OLLAMA_BASE_URL}/api/generate"
        )

        print("\nInitializing visual analyzer...")
        print(f"Model: {self.model}")

        try:
            response = requests.get(
                f"{OLLAMA_BASE_URL}/api/tags",
                timeout=10
            )

            response.raise_for_status()

        except requests.RequestException as error:
            raise ConnectionError(
                f"Unable to connect to Ollama: {error}"
            ) from error

        available_models = [
            model["name"]
            for model in response.json().get("models", [])
        ]

        if self.model not in available_models:
            raise RuntimeError(
                f"Model '{self.model}' is not available.\n"
                f"Available models: {available_models}"
            )

        print("Ollama connection established!")
        print("Visual analyzer initialized successfully!")

    @staticmethod
    def encode_image(image_path: Path) -> str:

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with image_path.open("rb") as image_file:
            return base64.b64encode(
                image_file.read()
            ).decode("utf-8")

    @staticmethod
    def get_metrics(result: dict) -> dict:

        prompt_eval_duration = result.get(
            "prompt_eval_duration", 0
        )

        eval_duration = result.get(
            "eval_duration", 0
        )

        prompt_eval_count = result.get(
            "prompt_eval_count", 0
        )

        eval_count = result.get(
            "eval_count", 0
        )

        return {
            "total_duration_seconds": round(
                result.get("total_duration", 0) / 1e9,
                3
            ),

            "load_duration_seconds": round(
                result.get("load_duration", 0) / 1e9,
                3
            ),

            "prompt_eval_duration_seconds": round(
                prompt_eval_duration / 1e9,
                3
            ),

            "eval_duration_seconds": round(
                eval_duration / 1e9,
                3
            ),

            "prompt_eval_count": prompt_eval_count,

            "eval_count": eval_count,

            "prompt_tokens_per_second": round(
                prompt_eval_count / (prompt_eval_duration / 1e9),
                2
            ) if prompt_eval_duration > 0 else 0,

            "tokens_per_second": round(
                eval_count / (eval_duration / 1e9),
                2
            ) if eval_duration > 0 else 0
        }

    


    def generate(
        self,
        prompt: str,
        images: list[str]
    ) -> dict:

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": images,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": 16384
            }
        }

        start_time = time.perf_counter()

        response = requests.post(
            self.generate_url,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        if not response.ok:
            print("\nOllama API Error!")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")

            response.raise_for_status()

        # This must be OUTSIDE the error-checking block
        result = response.json()

        wall_time = time.perf_counter() - start_time

        description = result.get(
            "response", ""
        ).strip()

        if not description:
            raise RuntimeError(
                "Ollama returned an empty description."
            )

        return {
            "description": description,

            "inference_time_seconds": round(
                wall_time, 3
            ),

            "ollama_metrics": self.get_metrics(result)
        }



    # -----------------------------------------
    # INDIVIDUAL FRAME ANALYSIS
    # -----------------------------------------

    @staticmethod
    def create_individual_prompt(timestamp: float) -> str:

        return f"""
    You are the visual understanding component of Project AURA.

    Analyze this video frame at timestamp {timestamp} seconds.

    Describe the visually important information:
    - Environment and setting.
    - Main objects and subjects.
    - Observable actions and spatial relationships.
    - Important readable text.

    RULES:
    - Describe only what is visibly supported.
    - Do not infer identities, intentions or emotions.
    - Avoid repeating obvious visual details.
    - Prioritize distinctive information useful for semantic retrieval.
    - Use concise, factual language.

    Return a single description in approximately 50-80 words.
    """

    def analyze_individual_frame(
        self,
        frame: dict
    ) -> dict:

        image = self.encode_image(
            frame["frame_path"]
        )

        prompt = self.create_individual_prompt(
            frame["global_timestamp"]
        )

        return self.generate(
            prompt=prompt,
            images=[image]
        )

    def analyze_individual_frames(
        self,
        frames: list[dict]
    ) -> list[dict]:

        results = []

        print("\nStarting individual frame analysis...")

        for frame in tqdm(
            frames,
            desc="Individual frames",
            unit="frame"
        ):

            analysis = self.analyze_individual_frame(
                frame
            )

            results.append({

                "frame_id": Path(
                    frame["frame_path"]
                ).stem,

                "frame_path": str(
                    frame["frame_path"]
                ),

                "local_timestamp": frame[
                    "local_timestamp"
                ],

                "global_timestamp": frame[
                    "global_timestamp"
                ],

                "description": analysis[
                    "description"
                ],

                "inference_time_seconds": analysis[
                    "inference_time_seconds"
                ],

                "ollama_metrics": analysis[
                    "ollama_metrics"
                ]

            })

        return results

    # -----------------------------------------
    # TEMPORAL GROUP ANALYSIS
    # -----------------------------------------

    @staticmethod
    def create_temporal_prompt(
        frames: list[dict]
    ) -> str:

        timeline = "\n".join(
            f"- Image {index + 1}: "
            f"{frame['global_timestamp']} seconds"
            for index, frame in enumerate(frames)
        )

        return f"""
    Analyze these {len(frames)} consecutive video frames
    in chronological order.

    FRAME TIMELINE:
    {timeline}

    Identify the main observable event or action
    across the sequence.

    Focus on:
    - What actions occur?
    - How do they change or progress?
    - Which subjects or objects remain involved?
    - What is the overall sequence of events?

    RULES:
    - Describe the event as a unified sequence.
    - Combine repeated or continuous actions.
    - Avoid describing each frame individually.
    - Avoid repeating static background details.
    - Do not infer emotions, intentions or identities.
    - Do not invent events between frames.
    - If no meaningful action occurs, state that
    the scene remains largely unchanged.

    Use concise, factual language.
    Return ONE unified description in approximately
    50-80 words.
    """
    def analyze_temporal_group(
        self,
        frames: list[dict]
    ) -> dict:

        images = [
            self.encode_image(frame["frame_path"])
            for frame in frames
        ]

        prompt = self.create_temporal_prompt(
            frames
        )

        return self.generate(
            prompt=prompt,
            images=images
        )

    def analyze_temporal_frames(
        self,
        frames: list[dict]
    ) -> list[dict]:

        frame_groups = [
            frames[index:index + FRAME_GROUP_SIZE]
            for index in range(
                0,
                len(frames),
                FRAME_GROUP_SIZE
            )
        ]

        results = []

        print("\nStarting temporal group analysis...")

        for index, group in enumerate(
            tqdm(
                frame_groups,
                desc="Temporal groups",
                unit="group"
            ),
            start=1
        ):

            analysis = self.analyze_temporal_group(
                group
            )

            results.append({

                "group_id": f"group_{index:03d}",

                "frame_ids": [
                    Path(frame["frame_path"]).stem
                    for frame in group
                ],

                "start_timestamp": group[0][
                    "global_timestamp"
                ],

                "end_timestamp": group[-1][
                    "global_timestamp"
                ],

                "description": analysis[
                    "description"
                ],

                "inference_time_seconds": analysis[
                    "inference_time_seconds"
                ],

                "ollama_metrics": analysis[
                    "ollama_metrics"
                ]

            })

        return results

    # -----------------------------------------
    # COMPLETE VISUAL ANALYSIS
    # -----------------------------------------

    
    def analyze_frames(
        self,
        frames: list[dict],
        output_path: Path,
        chunk_id: str
    ) -> dict:

        if not frames:
            raise ValueError("No frames provided.")

        if FRAME_GROUP_SIZE <= 0:
            raise ValueError("Frame group size must be greater than zero.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_start = time.perf_counter()

        result = {
            "chunk_id": chunk_id,
            "model": self.model,
            "status": "in_progress",
            "total_frames": len(frames),
            "frame_group_size": FRAME_GROUP_SIZE,
            "temporal_analysis_enabled": ENABLE_TEMPORAL_ANALYSIS,

            "total_temporal_groups": (
                (len(frames) + FRAME_GROUP_SIZE - 1)
                // FRAME_GROUP_SIZE
                if ENABLE_TEMPORAL_ANALYSIS else 0
            ),
            
            "individual_segments": [],
            "temporal_segments": []
        }

        def save_progress():

            result["total_elapsed_time_seconds"] = round(
                time.perf_counter() - total_start, 3
            )

            with output_path.open("w", encoding="utf-8") as file:
                json.dump(
                    result,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

        try:

            # Pass 1: Individual frame analysis
            print("\nStarting individual frame analysis...")

            for frame in tqdm(
                frames,
                desc="Individual frames",
                unit="frame"
            ):

                analysis = self.analyze_individual_frame(frame)

                result["individual_segments"].append({
                    "frame_id": Path(frame["frame_path"]).stem,
                    "frame_path": str(frame["frame_path"]),
                    "local_timestamp": frame["local_timestamp"],
                    "global_timestamp": frame["global_timestamp"],
                    "description": analysis["description"],
                    "inference_time_seconds": analysis["inference_time_seconds"],
                    "ollama_metrics": analysis["ollama_metrics"]
                })

                save_progress()

            print("\nIndividual analysis completed!")

            
            # Pass 2: Temporal group analysis 

            if ENABLE_TEMPORAL_ANALYSIS:

                frame_groups = [
                    frames[index:index + FRAME_GROUP_SIZE]
                    for index in range(0, len(frames), FRAME_GROUP_SIZE)
                ]

                print("\nStarting temporal group analysis...")

                for index, group in enumerate(
                    tqdm(frame_groups, desc="Temporal groups", unit="group"),
                    start=1
                ):

                    analysis = self.analyze_temporal_group(group)

                    result["temporal_segments"].append({
                        "group_id": f"group_{index:03d}",
                        "frame_ids": [
                            Path(frame["frame_path"]).stem
                            for frame in group
                        ],
                        "start_timestamp": group[0]["global_timestamp"],
                        "end_timestamp": group[-1]["global_timestamp"],
                        "description": analysis["description"],
                        "inference_time_seconds": analysis["inference_time_seconds"],
                        "ollama_metrics": analysis["ollama_metrics"]
                    })

                    save_progress()

                print("\nTemporal analysis completed!")

            else:

                print("\nTemporal analysis skipped (disabled in .env).")

                save_progress()

            # Final statistics
            result["individual_analysis_time_seconds"] = round(
                sum(
                    segment["inference_time_seconds"]
                    for segment in result["individual_segments"]
                ), 3
            )

            result["temporal_analysis_time_seconds"] = round(
                sum(
                    segment["inference_time_seconds"]
                    for segment in result["temporal_segments"]
                ), 3
            )

            result["status"] = "completed"

            save_progress()

            print("\nVisual analysis completed!")
            print(f"Individual descriptions: {len(result['individual_segments'])}")
            print(f"Temporal descriptions: {len(result['temporal_segments'])}")
            print(f"Total elapsed time: {result['total_elapsed_time_seconds']:.2f}s")
            print(f"Saved to: {output_path}")

            return result

        except Exception:

            result["status"] = "partial"
            save_progress()

            print("\nAnalysis interrupted. Completed results preserved.")

            raise
