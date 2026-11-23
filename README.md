**# Project AURA**



**\*\*Local-first audio-visual understanding and semantic video retrieval.\*\***



Project AURA turns a video into timestamped visual descriptions and

speech transcripts, embeds those records locally, and makes them

searchable with natural-language queries.



The current MVP uses a modular pipeline built around FFmpeg, Whisper,

Ollama-hosted vision and embedding models, and ChromaDB. Video

understanding and retrieval run locally; no paid inference API is

required.



\> **\*\*Status:\*\*** Functional research / portfolio MVP. The ingestion and

\> retrieval stages work independently. This is not yet a production

\> video-search service.



**## What it does**



\-   Splits an input video into configurable MP4 chunks.

\-   Extracts mono, 16 kHz WAV audio and transcribes it with Whisper.

\-   Samples video frames at a configurable rate (currently 1 FPS).

\-   Generates concise per-frame visual descriptions with Qwen3-VL

    through Ollama.

\-   Optionally analyses groups of consecutive frames to produce temporal

    descriptions.

\-   Generates 1024-dimensional BGE-M3 embeddings for individual visual

    descriptions and transcript segments.

\-   Stores visual and audio records in separate persistent ChromaDB

    collections.

\-   Retrieves relevant frames, groups nearby visual matches into

    candidate moments, and searches transcript segments independently.

\-   Provides an experimental multimodal result merger.



**## Architecture**



\`\`\` mermaid

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

\`\`\`



**### Retrieval design**



**\*\*Visual retrieval\*\***



1\.  Embed the user's query with BGE-M3.

2\.  Search the \`aura_visual\` ChromaDB collection.

3\.  Retrieve a wider set of candidate frames.

4\.  Group nearby frames using timestamp proximity.

5\.  Rank candidate moments using a combination of maximum and average

    frame similarity.



The current temporal grouping defaults are a maximum 2-second gap

between consecutive frames and a maximum 4-second group duration.



**\*\*Audio retrieval\*\***



1\.  Embed the user's query with the same BGE-M3 model.

2\.  Search the \`aura_audio\` collection.

3\.  Return matching transcript segments with their global start and end

    timestamps.



**\*\*Multimodal retrieval\*\***



The current \`MultimodalRetriever\` independently retrieves visual and

audio results, assigns each result a rank-based score using

\`1 / (60 + rank)\`, and merges the two lists.



This is an experimental ranked interleaving strategy, \*\*not full

cross-modal evidence fusion\*\*: visual frames and transcript segments

have different record IDs, so the current implementation does not

combine evidence for the same underlying moment.



**## Models and core components**



**&#xA0; -----------------------------------------------------------------------**

  Component               Current implementation  Purpose

**&#xA0; ----------------------- ----------------------- -----------------------**

  Video processing        FFmpeg / FFprobe        Chunking, audio

                                                  extraction and frame

                                                  sampling



  Speech recognition      OpenAI Whisper          Timestamped speech

                          \`large-v3-turbo\`        transcription



  Visual understanding    Qwen3-VL 8B via Ollama  Frame descriptions and

                                                  temporal group

                                                  descriptions



  Text embeddings         BGE-M3 via Ollama       Embeddings for visual

                                                  descriptions,

                                                  transcripts and queries



  Vector database         ChromaDB                Persistent vector

                                                  storage and

                                                  cosine-distance

                                                  retrieval



  Orchestration           Python                  Pipeline and retrieval

                                                  modules

  -----------------------------------------------------------------------



The models run locally through Ollama or the local Whisper installation.

Model weights must be downloaded before first use.



**## Repository layout**



The source code is organised into Python packages. Runtime data and

model files are deliberately excluded from version control.



\`\`\` text

project_AURA/

├── data/

│   ├── raw_videos/                 # Input videos (local only)

│   ├── processed/                  # Chunks and extracted artifacts (local only)

│   └── vector_db/                  # Persistent ChromaDB data (local only)

│

├── processing/

│   ├── \_\_init\_\_.py

│   ├── videoChunker.py

│   ├── audioExtractor.py

│   ├── audioTranscriber.py

│   ├── frameExtractor.py

│   └── visualAnalyzer.py

│

├── embeddings/

│   ├── \_\_init\_\_.py

│   ├── embeddingGenerator.py

│   └── audioEmbeddingGenerator.py

│

├── storage/

│   ├── \_\_init\_\_.py

│   ├── vectorStore.py

│   └── audioVectorStore.py

│

├── retrieval/

│   ├── \_\_init\_\_.py

│   ├── retriever.py

│   ├── audioRetriever.py

│   ├── multimodalRetriever.py

│   └── temporalGrouper.py

│

├── tests/

│   ├── \_\_init\_\_.py

│   └── test_visual.py

│

├── main.py

├── .env                            # Local configuration (not committed)

├── .env.example                    # Configuration template

├── .gitignore

├── .python-version

├── pyproject.toml

├── uv.lock

└── README.md

\`\`\`



Run package modules from the repository root with \`python -m\`. This

keeps the project root on the Python import path and allows package

imports to resolve consistently.



**## Requirements**



\-   Windows, Linux or macOS (the current project has been developed and

    tested on Windows).

\-   Python **\*\*3.12\*\***.

\-   [\`uv\`]\(https\://docs.astral.sh/uv/) for environment and dependency

    management.

\-   [FFmpeg]\(https\://ffmpeg.org/) with both \`ffmpeg\` and \`ffprobe\`

    available on \`PATH\`.

\-   [Ollama]\(https\://ollama.com/) running locally.

\-   A CUDA-capable NVIDIA GPU is recommended for practical Whisper and

    vision-model performance. The Whisper device is configurable; CPU

    execution is possible but will be slower.



**### Ollama models**



Pull the models used by the current configuration:



\`\`\` powershell

ollama pull qwen3-vl:8b-instruct

ollama pull bge-m3:latest

\`\`\`



Make sure the Ollama service is running before starting AURA.



**### Whisper checkpoint**



The current transcription module loads the \`large-v3-turbo\` checkpoint

from a local file. By default, it expects:



\`\`\` text

\~/.cache/whisper/large-v3-turbo.pt

\`\`\`



If the checkpoint is not present, download it once with the Whisper

package, or set \`AURA_WHISPER_MODEL_PATH\` in your local \`.env\` to its

actual location. The configured path must point to the checkpoint file.



**## Setup

Follow these steps if you are setting up AURA on a fresh computer.

### 1. Install the prerequisites

Install the following tools before cloning the project:

- **Git:** https://git-scm.com/downloads
- **Python 3.12:** https://www.python.org/downloads/
- **uv:** https://docs.astral.sh/uv/getting-started/installation/
- **FFmpeg:** https://ffmpeg.org/download.html  
  Make sure both `ffmpeg` and `ffprobe` are available on your system `PATH`.
- **Ollama:** https://ollama.com/download

After installation, open a new PowerShell terminal and check that the commands are available:

```powershell
git --version
python --version
uv --version
ffmpeg -version
ffprobe -version
ollama --version
```

### 2. Clone the AURA repository

Open PowerShell in the folder where you want to keep the project. Copy the repository's HTTPS URL from GitHub using **Code → HTTPS**, then run:

```powershell
git clone https://github.com/YOUR-USERNAME/Project-AURA.git
```

Replace the example URL with the actual HTTPS URL of this repository.

Move into the newly cloned project folder:

```powershell
cd Project-AURA
```

You should now be inside the repository root, where `pyproject.toml`, `uv.lock`, and `main.py` are located.

### 3. Install the Python dependencies

Run this from the repository root:

```powershell
uv sync
```

This creates the project's virtual environment and installs the dependencies declared in `pyproject.toml`, using the versions recorded in `uv.lock`.

### 4. Create your local configuration file

Copy the example configuration:

```powershell
Copy-Item .env.example .env
```

Open `.env` in your editor. Set `AURA_INPUT_VIDEO` to the path of your own video file. Do not commit your `.env` file; it contains local configuration.

### 5. Add a video to process

Create the input directory:

```powershell
New-Item -ItemType Directory -Force data/raw_videos
```

Place your video inside that folder, then set `AURA_INPUT_VIDEO` in `.env` to its path. You can also use a different valid path if you prefer.

### 6. Download the required local models

Pull the Ollama models used by the current configuration:

```powershell
ollama pull qwen3-vl:8b-instruct
ollama pull bge-m3:latest
```

Make sure the Ollama service is running before starting AURA.

AURA's Whisper transcription module also needs the `large-v3-turbo` checkpoint. See the **Whisper checkpoint** section above for its expected location and configuration.

Once these steps are complete, continue to [Run the pipeline](#run-the-pipeline).

**## Configuration**



AURA reads configuration from \`.env\`. See [\`.env.example\`]\(.env.example)

for the full template.



**&#xA0; -------------------------------------------------------------------------------------**

  Variable                          Purpose             Typical value

**&#xA0; --------------------------------- ------------------- -------------------------------**

  \`AURA_INPUT_VIDEO\`                Source video path   \`data/raw_videos/example.mp4\`



  \`AURA_OUTPUT_DIR\`                 Chunk and           \`data/processed/v1\`

                                    extraction output  &#x20;

                                    root                



  \`AURA_CHUNK_DURATION\`             Chunk length in     \`10\`

                                    seconds            &#x20;



  \`AURA_WHISPER_MODEL_PATH\`         Local Whisper       Default cache path if omitted

                                    checkpoint path    &#x20;



  \`AURA_WHISPER_DEVICE\`             Whisper execution   \`cuda\`

                                    device              



  \`AURA_WHISPER_BEAM_SIZE\`          Whisper beam size   \`5\`



  \`AURA_WHISPER_LANGUAGE\`           Optional forced     blank

                                    language; blank    &#x20;

                                    enables detection  &#x20;



  \`AURA_OLLAMA_BASE_URL\`            Ollama server URL   \`http\://localhost:11434\`



  \`AURA_VISION_MODEL\`               Ollama vision model \`qwen3-vl:8b-instruct\`



  \`AURA_VISION_TIMEOUT\`             Vision request      \`300\`

                                    timeout in seconds  



  \`AURA_FRAME_GROUP_SIZE\`           Frames per          \`5\`

                                    temporal-analysis  &#x20;

                                    group              &#x20;



  \`AURA_EMBEDDING_MODEL\`            Ollama embedding    \`bge-m3:latest\`

                                    model              &#x20;



  \`AURA_EMBEDDING_DIMENSIONS\`       Expected embedding  \`1024\`

                                    dimensions          



  \`AURA_EMBEDDING_TIMEOUT\`          Embedding request   \`300\`

                                    timeout in seconds  



  \`AURA_VECTOR_DB_DIR\`              Persistent ChromaDB \`data/vector_db\`

                                    directory          &#x20;



  \`AURA_VISUAL_COLLECTION\`          Visual collection   \`aura_visual\`

                                    name                



  \`AURA_AUDIO_COLLECTION\`           Audio collection    \`aura_audio\`

                                    name                



  \`AURA_ENABLE_TEMPORAL_ANALYSIS\`   Enable Qwen         \`true\`

                                    temporal group      

                                    descriptions        

  -------------------------------------------------------------------------------------



**## Run the pipeline**



The current workflow is intentionally split into separate stages.

\`main.py\` performs video chunking and preprocessing; embedding

generation and ChromaDB ingestion are separate commands.



**### 1. Chunk and preprocess the video**



\`\`\` powershell

uv run python main.py

\`\`\`



This runs:



1\.  Video chunking and MP4 conversion.

2\.  Audio extraction for each chunk.

3\.  Whisper transcription with local and global timestamps.

4\.  Frame extraction at the configured sampling rate.

5\.  Qwen3-VL analysis of individual frames.

6\.  Optional Qwen3-VL temporal analysis over groups of frames.



**### 2. Generate visual embeddings**



\`\`\` powershell

uv run python -m embeddings.embeddingGenerator

\`\`\`



This reads each \`visual_analysis.json\` and embeds its

\`individual_segments\`. Temporal group descriptions are currently saved

in the analysis JSON but are not included in this embedding-generation

step.



**### 3. Generate audio embeddings**



\`\`\` powershell

uv run python -m embeddings.audioEmbeddingGenerator

\`\`\`



This embeds the transcript segments produced by Whisper. Empty

transcript files are skipped.



**### 4. Ingest vectors into ChromaDB**



\`\`\` powershell

uv run python -m storage.vectorStore

uv run python -m storage.audioVectorStore

\`\`\`



The two scripts populate separate collections:



\-   \`aura_visual\`

\-   \`aura_audio\`



The stores use ChromaDB \`upsert\`, so re-running ingestion with the same

record IDs updates existing records rather than creating duplicate

copies. It does not automatically delete stale records that are no

longer present in the source data.



**### 5. Search**



Run any retriever after the corresponding collection has been populated:



\`\`\` powershell

\# Timestamp-aware visual search

uv run python -m retrieval.retriever



\# Transcript search

uv run python -m retrieval.audioRetriever



\# Experimental combined result list

uv run python -m retrieval.multimodalRetriever

\`\`\`



Enter a natural-language query at the prompt. Type \`exit\` or \`quit\` to

close the search loop.



**### 6. Run the visual-analysis test script**



The test script currently points to the first processed chunk in the

example \`data/processed/v1\` directory.



\`\`\` powershell

uv run python -m tests.test_visual

\`\`\`



Adjust \`FRAME_DIR\` and \`OUTPUT_PATH\` in \`tests/test_visual.py\` if your

local data is stored elsewhere.



**## Example queries**



Queries used during the MVP smoke test included:



\`\`\` text

a scene showing character alone

A close-up of an important object

Someone speaking in Korean

a guy hugging a girl

\`\`\`



The visual retriever returned relevant moments such as a handwritten

card, a bouquet, a hand interacting with a door, and a couple embracing.

The multimodal retriever surfaced the embrace around 49--52 seconds.



These are qualitative examples from one test video, not a formal

accuracy benchmark.



**## Demo and test media**



The following short films are suitable sample inputs for demonstrating

AURA's video-preprocessing and retrieval pipeline. They are third-party

works and are **\*\*not\*\*** part of the AURA source code or project ownership.



**&#xA0; -------------------------------------------------------------------------------------------------------------------------------------------------------**

  Film                                                                     Duration Published licence                                    Source

**&#xA0; ----------------------------------------------------------- --------------------- ---------------------------------------------------- ----------------**

  [The Daily                                                       About 50 seconds [CC BY                                               Blender Studio

  Dweebs]\(https\://studio.blender.org/projects/dailydweebs/)                         3.0]\(https\://creativecommons.org/licenses/by/3.0/)  &#x20;



  [Biscuit: From Stray Dog to                                            60 seconds [CC BY                                               The Studio

  Superhero]\(https\://www\.thestudio.io/osvideo/)                                     4.0]\(https\://creativecommons.org/licenses/by/4.0/)  &#x20;

  -------------------------------------------------------------------------------------------------------------------------------------------------------



**### Attribution**



If you use either film in a public AURA demonstration, retain the

corresponding attribution:



\-   **\*\*The Daily Dweebs:\*\*** © Blender Foundation, [Blender

    Studio]\(https\://studio.blender.org/projects/dailydweebs/). Licensed

    under [Creative Commons Attribution

    3.0]\(https\://creativecommons.org/licenses/by/3.0/).

\-   **\*\*Biscuit:\*\*** "Biscuit by The Studio (thestudio.io), CC BY 4.0".

    Licence: [Creative Commons Attribution

    4.0]\(https\://creativecommons.org/licenses/by/4.0/).



These are Creative Commons-licensed works, **\*\*not public-domain works\*\***.

The licences allow reuse subject to their attribution terms. Check the

original source page and any asset-specific notices before

redistributing a copy or modified version.



Use the films as local test inputs. Do not add downloaded source videos

to the AURA repository; the \`.gitignore\` excludes \`data/raw_videos/\` and

other generated data directories.



**## MVP benchmark**



**\*\*Test input:\*\*** one 73.381-second short film, 1920 × 802 at 25 FPS, with

VP9 video and Opus audio.



  Measurement                       Observed result

**&#xA0; ------------------------------- -----------------**

  Input duration                     73.381 seconds

  Chunk duration                         10 seconds

  Video chunks                                    8

  Sampled frames                                 73

  Temporal descriptions                          15

  Whisper transcript segments                    21

  Visual vector records                          73

  Audio vector records                           21

  Video preprocessing wall time       46 min 49 sec

  Visual embedding generation           about 4 sec

  Audio embedding generation            about 7 sec



The preprocessing wall time is dominated by local Qwen3-VL visual

inference, including per-frame and temporal-group analysis. The

embedding and ChromaDB ingestion stages were much faster in this run.



These numbers describe one run on one video and one local machine. They

are not general throughput guarantees. Query latency was perceived as

negligible during manual testing but was not formally benchmarked.



**## Known limitations**



\-   **\*\*Visual ingestion is slow.\*\*** Qwen3-VL is called for every sampled

    frame and every temporal group. The current implementation

    prioritizes descriptive coverage over throughput.

\-   **\*\*Audio transcription can hallucinate.\*\*** Whisper may produce text

    for music, noise or silent passages. Review transcript output before

    treating it as ground truth.

\-   **\*\*Temporal descriptions are not yet embedded.\*\*** The current visual

    vector collection contains individual-frame descriptions; temporal

    descriptions are retained in JSON.

\-   **\*\*Multimodal fusion is experimental.\*\*** The current rank-based merger

    interleaves independent audio and visual results; it does not align

    or jointly score evidence for a shared moment.

\-   **\*\*No formal retrieval evaluation yet.\*\*** There is no labelled query

    set or measured precision, recall, mAP or nDCG.

\-   **\*\*The current workflow is script-driven.\*\*** Preprocessing, embedding,

    ingestion and retrieval are launched as separate commands. There is

    no API, UI, job queue or production deployment layer yet.

\-   **\*\*Re-ingestion does not prune stale records.\*\*** If source records are

    removed or renamed, old ChromaDB IDs may remain until explicitly

    cleaned.



**## Current scope and possible next steps**



The MVP establishes the end-to-end local pipeline and proves that

timestamped natural-language retrieval works on a sample video.



Potential future work:



\-   Adaptive or scene-aware frame sampling.

\-   Faster or batched visual inference where supported.

\-   Speech validation or voice activity detection before indexing

    transcripts.

\-   Embedding temporal descriptions.

\-   Time-aware audio-visual evidence alignment and genuine cross-modal

    fusion.

\-   A reproducible evaluation set and retrieval metrics.

\-   A FastAPI service or lightweight user interface.

\-   Docker packaging and deployment.



**## Privacy and data handling**



Input videos, extracted media, JSON artifacts, model checkpoints and the

local ChromaDB database can contain private or copyrighted material.

Keep those assets out of public commits unless you have permission to

share them. The supplied \`.gitignore\` excludes the standard local data

and model locations.



**## License**



No license has been selected for this repository yet. Add a license file

before inviting others to reuse or redistribute the project.
