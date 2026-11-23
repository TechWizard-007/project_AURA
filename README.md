# Project AURA

**Local-first audio-visual understanding and semantic video retrieval.**

Project AURA turns a video into timestamped visual descriptions and
speech transcripts, embeds those records locally, and makes them
searchable with natural-language queries.

The current MVP uses a modular pipeline built around FFmpeg, Whisper,
Ollama-hosted vision and embedding models, and ChromaDB. Video
understanding and retrieval run locally; no paid inference API is
required.

> **Status:** Functional research / portfolio MVP. The ingestion and
> retrieval stages work independently. This is not yet a production
> video-search service.

## What it does

-   Splits an input video into configurable MP4 chunks.
-   Extracts mono, 16 kHz WAV audio and transcribes it with Whisper.
-   Samples video frames at a configurable rate (currently 1 FPS).
-   Generates concise per-frame visual descriptions with Qwen3-VL
    through Ollama.
-   Optionally analyses groups of consecutive frames to produce temporal
    descriptions.
-   Generates 1024-dimensional BGE-M3 embeddings for individual visual
    descriptions and transcript segments.
-   Stores visual and audio records in separate persistent ChromaDB
    collections.
-   Retrieves relevant frames, groups nearby visual matches into
    candidate moments, and searches transcript segments independently.
-   Provides an experimental multimodal result merger.

## Architecture

``` mermaid
flowchart TD
    A[Input video] --> B[FFmpeg video chunker]
    B --> C[MP4 chunks]
    C --> D[Audio extraction]
    C --> E[Frame extraction]
    D --> F[16 kHz mono WAV]
    F --> G[Whisper transcription]
    G --> H[Timestamped transcript JSON]
    E --> I[Sampled image frames]
    I --> J[Qwen3-VL via Ollama]
    J --> K[Per-frame descriptions]
    J --> L[Optional temporal group descriptions]
    K --> M[BGE-M3 embeddings]
    H --> N[BGE-M3 embeddings]
    M --> O[visual_embeddings.json]
    N --> P[audio_embeddings.json]
    O --> Q[(ChromaDB: aura_visual)]
    P --> R[(ChromaDB: aura_audio)]
    S[Natural-language query] --> T[BGE-M3 query embedding]
    T --> U[Visual retriever]
    T --> V[Audio retriever]
    Q --> U
    R --> V
    U --> W[Temporal grouping]
    W --> X[Timestamped visual moments]
    V --> Y[Timestamped transcript matches]
    X --> Z[Experimental multimodal merger]
    Y --> Z
```

### Retrieval design

**Visual retrieval**

1.  Embed the user's query with BGE-M3.
2.  Search the `aura_visual` ChromaDB collection.
3.  Retrieve a wider set of candidate frames.
4.  Group nearby frames using timestamp proximity.
5.  Rank candidate moments using a combination of maximum and average
    frame similarity.

The current temporal grouping defaults are a maximum 2-second gap
between consecutive frames and a maximum 4-second group duration.

**Audio retrieval**

1.  Embed the user's query with the same BGE-M3 model.
2.  Search the `aura_audio` collection.
3.  Return matching transcript segments with their global start and end
    timestamps.

**Multimodal retrieval**

The current `MultimodalRetriever` independently retrieves visual and
audio results, assigns each result a rank-based score using
`1 / (60 + rank)`, and merges the two lists.

This is an experimental ranked interleaving strategy, **not full
cross-modal evidence fusion**: visual frames and transcript segments
have different record IDs, so the current implementation does not
combine evidence for the same underlying moment.

## Models and core components

  -----------------------------------------------------------------------
  Component               Current implementation  Purpose
  ----------------------- ----------------------- -----------------------
  Video processing        FFmpeg / FFprobe        Chunking, audio
                                                  extraction and frame
                                                  sampling

  Speech recognition      OpenAI Whisper          Timestamped speech
                          `large-v3-turbo`        transcription

  Visual understanding    Qwen3-VL 8B via Ollama  Frame descriptions and
                                                  temporal group
                                                  descriptions

  Text embeddings         BGE-M3 via Ollama       Embeddings for visual
                                                  descriptions,
                                                  transcripts and queries

  Vector database         ChromaDB                Persistent vector
                                                  storage and
                                                  cosine-distance
                                                  retrieval

  Orchestration           Python                  Pipeline and retrieval
                                                  modules
  -----------------------------------------------------------------------

The models run locally through Ollama or the local Whisper installation.
Model weights must be downloaded before first use.

## Repository layout

The following is the intended source-code layout. Runtime data and model
files are deliberately excluded from version control.

``` text
project_AURA/
├── data/
│   ├── raw_videos/                 # Input videos (local only)
│   ├── processed/                  # Chunks and extracted artifacts (local only)
│   └── vector_db/                  # Persistent ChromaDB data (local only)
├── .env                            # Local configuration (not committed)
├── .env.example                    # Configuration template
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
│
├── videoChunker.py
├── audioExtractor.py
├── audioTranscriber.py
├── frameExtractor.py
├── visualAnalyzer.py
├── main.py
│
├── embeddingGenerator.py
├── audioEmbeddingGenerator.py
├── vectorStore.py
├── audioVectorStore.py
│
├── retriever.py
├── audioRetriever.py
├── temporalGrouper.py
└── multimodalRetriever.py
```

## Requirements

-   Windows, Linux or macOS (the current project has been developed and
    tested on Windows).
-   Python **3.12**.
-   [`uv`](https://docs.astral.sh/uv/) for environment and dependency
    management.
-   [FFmpeg](https://ffmpeg.org/) with both `ffmpeg` and `ffprobe`
    available on `PATH`.
-   [Ollama](https://ollama.com/) running locally.
-   A CUDA-capable NVIDIA GPU is recommended for practical Whisper and
    vision-model performance. The Whisper device is configurable; CPU
    execution is possible but will be slower.

### Ollama models

Pull the models used by the current configuration:

``` powershell
ollama pull qwen3-vl:8b-instruct
ollama pull bge-m3:latest
```

Make sure the Ollama service is running before starting AURA.

### Whisper checkpoint

The current transcription module loads the `large-v3-turbo` checkpoint
from a local file. By default, it expects:

``` text
~/.cache/whisper/large-v3-turbo.pt
```

If the checkpoint is not present, download it once with the Whisper
package, or set `AURA_WHISPER_MODEL_PATH` in your local `.env` to its
actual location. The configured path must point to the checkpoint file.

## Setup

Clone the repository, then open a terminal in its root directory.

``` powershell
# Create the environment and install the locked dependencies
uv sync

# Copy the configuration template
Copy-Item .env.example .env
```

Edit `.env` and set `AURA_INPUT_VIDEO` to the path of a video you want
to process. Create the input directory if needed:

``` powershell
New-Item -ItemType Directory -Force data/raw_videos
```

Place the input video in that directory, or set `AURA_INPUT_VIDEO` to
another valid path.

## Configuration

AURA reads configuration from `.env`. See [`.env.example`](.env.example)
for the full template.

  -----------------------------------------------------------------------------------------
  Variable                          Purpose                 Typical value
  --------------------------------- ----------------------- -------------------------------
  `AURA_INPUT_VIDEO`                Source video path       `data/raw_videos/example.mp4`

  `AURA_OUTPUT_DIR`                 Chunk and extraction    `data/processed/v1`
                                    output root             

  `AURA_CHUNK_DURATION`             Chunk length in seconds `10`

  `AURA_WHISPER_MODEL_PATH`         Local Whisper           Default cache path if omitted
                                    checkpoint path         

  `AURA_WHISPER_DEVICE`             Whisper execution       `cuda`
                                    device                  

  `AURA_WHISPER_BEAM_SIZE`          Whisper beam size       `5`

  `AURA_WHISPER_LANGUAGE`           Optional forced         blank
                                    language; blank enables 
                                    detection               

  `AURA_OLLAMA_BASE_URL`            Ollama server URL       `http://localhost:11434`

  `AURA_VISION_MODEL`               Ollama vision model     `qwen3-vl:8b-instruct`

  `AURA_VISION_TIMEOUT`             Vision request timeout  `300`
                                    in seconds              

  `AURA_FRAME_GROUP_SIZE`           Frames per              `5`
                                    temporal-analysis group 

  `AURA_EMBEDDING_MODEL`            Ollama embedding model  `bge-m3:latest`

  `AURA_EMBEDDING_DIMENSIONS`       Expected embedding      `1024`
                                    dimensions              

  `AURA_EMBEDDING_TIMEOUT`          Embedding request       `300`
                                    timeout in seconds      

  `AURA_VECTOR_DB_DIR`              Persistent ChromaDB     `data/vector_db`
                                    directory               

  `AURA_VISUAL_COLLECTION`          Visual collection name  `aura_visual`

  `AURA_AUDIO_COLLECTION`           Audio collection name   `aura_audio`

  `AURA_ENABLE_TEMPORAL_ANALYSIS`   Enable Qwen temporal    `true`
                                    group descriptions      
  -----------------------------------------------------------------------------------------

## Run the pipeline

The current workflow is intentionally split into separate stages.
`main.py` performs video chunking and preprocessing; embedding
generation and ChromaDB ingestion are separate commands.

### 1. Chunk and preprocess the video

``` powershell
uv run python main.py
```

This runs:

1.  Video chunking and MP4 conversion.
2.  Audio extraction for each chunk.
3.  Whisper transcription with local and global timestamps.
4.  Frame extraction at the configured sampling rate.
5.  Qwen3-VL analysis of individual frames.
6.  Optional Qwen3-VL temporal analysis over groups of frames.

### 2. Generate visual embeddings

``` powershell
uv run python embeddingGenerator.py
```

This reads each `visual_analysis.json` and embeds its
`individual_segments`. Temporal group descriptions are currently saved
in the analysis JSON but are not included in this embedding-generation
step.

### 3. Generate audio embeddings

``` powershell
uv run python audioEmbeddingGenerator.py
```

This embeds the transcript segments produced by Whisper. Empty
transcript files are skipped.

### 4. Ingest vectors into ChromaDB

``` powershell
uv run python vectorStore.py
uv run python audioVectorStore.py
```

The two scripts populate separate collections:

-   `aura_visual`
-   `aura_audio`

The stores use ChromaDB `upsert`, so re-running ingestion with the same
record IDs updates existing records rather than creating duplicate
copies. It does not automatically delete stale records that are no
longer present in the source data.

### 5. Search

Run any retriever after the corresponding collection has been populated:

``` powershell
# Timestamp-aware visual search
uv run python retriever.py

# Transcript search
uv run python audioRetriever.py

# Experimental combined result list
uv run python multimodalRetriever.py
```

Enter a natural-language query at the prompt. Type `exit` or `quit` to
close the search loop.

## Example queries

Queries used during the MVP smoke test included:

``` text
a scene showing character alone
A close-up of an important object
Someone speaking in Korean
a guy hugging a girl
```

The visual retriever returned relevant moments such as a handwritten
card, a bouquet, a hand interacting with a door, and a couple embracing.
The multimodal retriever surfaced the embrace around 49--52 seconds.

These are qualitative examples from one test video, not a formal
accuracy benchmark.

## MVP benchmark

**Test input:** one 73.381-second short film, 1920 × 802 at 25 FPS, with
VP9 video and Opus audio.

  Measurement                       Observed result
  ------------------------------- -----------------
  Input duration                     73.381 seconds
  Chunk duration                         10 seconds
  Video chunks                                    8
  Sampled frames                                 73
  Temporal descriptions                          15
  Whisper transcript segments                    21
  Visual vector records                          73
  Audio vector records                           21
  Video preprocessing wall time       46 min 49 sec
  Visual embedding generation           about 4 sec
  Audio embedding generation            about 7 sec

The preprocessing wall time is dominated by local Qwen3-VL visual
inference, including per-frame and temporal-group analysis. The
embedding and ChromaDB ingestion stages were much faster in this run.

These numbers describe one run on one video and one local machine. They
are not general throughput guarantees. Query latency was perceived as
negligible during manual testing but was not formally benchmarked.

## Known limitations

-   **Visual ingestion is slow.** Qwen3-VL is called for every sampled
    frame and every temporal group. The current implementation
    prioritizes descriptive coverage over throughput.
-   **Audio transcription can hallucinate.** Whisper may produce text
    for music, noise or silent passages. Review transcript output before
    treating it as ground truth.
-   **Temporal descriptions are not yet embedded.** The current visual
    vector collection contains individual-frame descriptions; temporal
    descriptions are retained in JSON.
-   **Multimodal fusion is experimental.** The current rank-based merger
    interleaves independent audio and visual results; it does not align
    or jointly score evidence for a shared moment.
-   **No formal retrieval evaluation yet.** There is no labelled query
    set or measured precision, recall, mAP or nDCG.
-   **The current workflow is script-driven.** Preprocessing, embedding,
    ingestion and retrieval are launched as separate commands. There is
    no API, UI, job queue or production deployment layer yet.
-   **Re-ingestion does not prune stale records.** If source records are
    removed or renamed, old ChromaDB IDs may remain until explicitly
    cleaned.

## Current scope and possible next steps

The MVP establishes the end-to-end local pipeline and proves that
timestamped natural-language retrieval works on a sample video.

Potential future work:

-   Adaptive or scene-aware frame sampling.
-   Faster or batched visual inference where supported.
-   Speech validation or voice activity detection before indexing
    transcripts.
-   Embedding temporal descriptions.
-   Time-aware audio-visual evidence alignment and genuine cross-modal
    fusion.
-   A reproducible evaluation set and retrieval metrics.
-   A FastAPI service or lightweight user interface.
-   Docker packaging and deployment.

## Privacy and data handling

Input videos, extracted media, JSON artifacts, model checkpoints and the
local ChromaDB database can contain private or copyrighted material.
Keep those assets out of public commits unless you have permission to
share them. The supplied `.gitignore` excludes the standard local data
and model locations.

## License

No license has been selected for this repository yet. Add a license file
before inviting others to reuse or redistribute the project.
