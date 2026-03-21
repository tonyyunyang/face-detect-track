# Movie-Like Shots

Minimal face detection + tracking pipeline for video clips.

Scope of this project stage:
- face detection
- face tracking
- stable per-frame IDs
- JSON export

Out of scope:
- face generation
- face swapping
- anonymization
- UI

## Layout

- `BoT-FaceSORT-VEED/`: third-party tracking repo, kept separate and tracked from the root repo as a git pointer
- `insightface-VEED/`: third-party face-analysis repo, kept separate and tracked from the root repo as a git pointer
- `src/movie_like_shots/`: project-owned wrapper code
- `models/`: downloaded detector and ReID weights
- `outputs/`: JSON exports and optional preview videos
- `kids-party-trim1-720.mp4`, `kids-party-trim1-1080.mp4`: sample inputs left in place intentionally to avoid moving large files recklessly

If you clone the root repo fresh, initialize the vendor repos with:

```bash
git submodule update --init --recursive
```

## Local Dev With `conda` + `uv`

Use `conda` only for local isolation, and `uv` to manage/install the project.

```bash
conda activate veed
uv pip install -e .
movie-like-shots kids-party-trim1-720.mp4 --write-preview
```

If a crowded scene is producing too many transient IDs, enable the optional export filter:

```bash
conda activate veed
uv pip install -e .
movie-like-shots kids-party-trim2-720.mp4 \
  --write-preview \
  --filter-tracks \
  --min-track-length 10 \
  --min-track-median-area 2500
```

If you want to drop low-confidence tracked detections from the exported JSON and preview, enable the optional confidence filter:

```bash
conda activate veed
uv pip install -e .
movie-like-shots kids-party-trim2-720.mp4 \
  --write-preview \
  --filter-confidence \
  --min-confidence 0.6
```

Or without installing a console script:

```bash
conda activate veed
uv pip install -e .
python -m movie_like_shots.cli kids-party-trim1-720.mp4 --write-preview
```

The default tracker is `ocsort`, which keeps the minimal pipeline free of a hard PyTorch dependency. If you want the heavier appearance-based tracker later, install the optional extra and provide an AdaFace checkpoint:

```bash
conda activate veed
uv pip install -e ".[botfacesort]"
movie-like-shots kids-party-trim1-720.mp4 --tracker botfacesort --reid-model models/adaface/resnet101_ir_webface12m.ckpt
```

## Final `uv`-Only Usage

```bash
uv sync
uv run movie-like-shots kids-party-trim1-720.mp4 --write-preview
```

On Apple Silicon, the default `--device auto` path uses CoreML for the SCRFD detector when `--det-size 640` is kept unchanged.

Track filtering is off by default. When `--filter-tracks` is enabled, the JSON export and preview keep only tracks that meet both thresholds:
- `--min-track-length`
- `--min-track-median-area`

Confidence filtering is also off by default. When `--filter-confidence` is enabled, the JSON export and preview keep only tracked detections whose `confidence` is at least:
- `--min-confidence`

`--det-thresh` is still the detector/runtime threshold. `--filter-confidence` is a separate export-stage filter on tracked detections after tracking has already run.

When any export filter is enabled, preview rendering runs in a second pass after tracking so the final MP4 matches the filtered JSON exactly.

Run the 1080p sample:

```bash
uv run movie-like-shots kids-party-trim1-1080.mp4 --write-preview
```

## Output

Each run writes into:

```text
outputs/<video-stem>/
  <video-stem>.tracks.json
  <video-stem>.preview.mp4      # only if --write-preview is used
```

The JSON includes:
- `video_metadata`
- `track_summary`
- `frames`

Each frame entry includes:
- `bbox`
- `center_point`
- `confidence`
- `id`
- `label`

`confidence` is the score carried through the tracker output for that tracked detection. In the current wrapper this is detector-derived confidence attached to the active track update, not a separate long-horizon tracking-confidence metric.

Labels are generated as readable aliases like `Person A`, `Person B`, in first-appearance order for each tracked ID.
