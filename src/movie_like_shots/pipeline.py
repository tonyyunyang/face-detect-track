from __future__ import annotations

import importlib.util
import json
import re
import string
import sys
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import gdown
import numpy as np

SCRFD_10G_GNKPS_URL = "https://drive.google.com/uc?id=1v9nhtPWMLSedueeL6c3nJEoIFlSNSCvh"
TRACKERS_REQUIRING_REID = {"botfacesort"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def bot_root() -> Path:
    return repo_root() / "BoT-FaceSORT-VEED"


def insightface_root() -> Path:
    return repo_root() / "insightface-VEED"


def default_detector_model() -> Path:
    return repo_root() / "models" / "scrfd" / "scrfd_10g_gnkps.onnx"


def default_reid_model() -> Path:
    return repo_root() / "models" / "adaface" / "resnet101_ir_webface12m.ckpt"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_detector_weights(model_path: Path) -> None:
    if model_path.exists():
        return
    ensure_parent_dir(model_path)
    gdown.download(SCRFD_10G_GNKPS_URL, str(model_path), quiet=False, use_cookies=False)


def ensure_bot_repo_on_path() -> None:
    bot_repo = bot_root()
    if str(bot_repo) not in sys.path:
        sys.path.insert(0, str(bot_repo))


@lru_cache(maxsize=1)
def load_insightface_scrfd_class() -> type:
    module_path = insightface_root() / "detection" / "scrfd" / "tools" / "scrfd.py"
    spec = importlib.util.spec_from_file_location(
        "_movie_like_shots_insightface_scrfd", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load InsightFace SCRFD module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SCRFD


def tracker_requires_reid(tracker_type: str) -> bool:
    return tracker_type in TRACKERS_REQUIRING_REID


@lru_cache(maxsize=1)
def available_onnx_providers() -> tuple[str, ...]:
    import onnxruntime as ort

    return tuple(ort.get_available_providers())


def resolve_detector_execution(device: str, detector_size: int) -> tuple[list[str], int]:
    available = set(available_onnx_providers())

    if device == "cpu":
        return ["CPUExecutionProvider"], -1
    if device == "cuda":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
        return ["CPUExecutionProvider"], -1

    # The current SCRFD 10G ONNX path is reliable with CoreML at 640x640 on Apple Silicon.
    if (
        sys.platform == "darwin"
        and detector_size == 640
        and "CoreMLExecutionProvider" in available
    ):
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"], 0
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
    return ["CPUExecutionProvider"], -1


def resolve_tracker_device(tracker_type: str, device: str) -> Any:
    if not tracker_requires_reid(tracker_type):
        return "cpu"

    import torch

    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def label_for_index(index: int) -> str:
    letters: list[str] = []
    current = index
    while True:
        current, remainder = divmod(current, 26)
        letters.append(string.ascii_uppercase[remainder])
        if current == 0:
            break
        current -= 1
    return f"Person {''.join(reversed(letters))}"


def round_and_clip_bbox(bbox: np.ndarray | list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def center_point_from_bbox(bbox: list[int]) -> list[int]:
    x1, y1, x2, y2 = bbox
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))
    return [cx, cy]


def bbox_area(bbox: list[int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def confidence_from_tracker_row(row: np.ndarray | list[float]) -> float:
    if len(row) <= 5:
        return 0.0
    return round(float(row[5]), 6)


def slugify_label(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug or "face"


def draw_preview_frame(frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    preview = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['label']} ({det['id']})"
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            preview,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return preview


def ordered_track_ids_for_ranges(track_ranges: dict[int, dict[str, int]]) -> list[int]:
    return sorted(
        track_ranges.keys(),
        key=lambda track_id: (track_ranges[track_id]["first_frame"], track_id),
    )


def select_track_ids_for_export(
    track_ranges: dict[int, dict[str, int]],
    track_stats: dict[int, dict[str, Any]],
    filter_tracks: bool,
    min_track_length: int,
    min_track_median_area: float,
) -> list[int]:
    ordered_track_ids = ordered_track_ids_for_ranges(track_ranges)
    if not filter_tracks:
        return ordered_track_ids

    min_track_length = max(0, min_track_length)
    min_track_median_area = max(0.0, float(min_track_median_area))
    selected_track_ids: list[int] = []
    for track_id in ordered_track_ids:
        stats = track_stats.get(track_id, {"frame_count": 0, "areas": []})
        frame_count = int(stats["frame_count"])
        median_area = (
            float(np.median(np.asarray(stats["areas"], dtype=np.float32)))
            if stats["areas"]
            else 0.0
        )
        if frame_count < min_track_length:
            continue
        if median_area < min_track_median_area:
            continue
        selected_track_ids.append(track_id)
    return selected_track_ids


def build_export_labels(selected_track_ids: list[int]) -> dict[int, str]:
    return {
        track_id: label_for_index(index)
        for index, track_id in enumerate(selected_track_ids)
    }


def build_track_ranges_from_frames(
    frames: dict[str, list[dict[str, Any]]],
) -> dict[int, dict[str, int]]:
    track_ranges: dict[int, dict[str, int]] = {}
    for frame_key, frame_entries in frames.items():
        frame_index = int(frame_key)
        for entry in frame_entries:
            track_id = int(entry["id"])
            if track_id not in track_ranges:
                track_ranges[track_id] = {
                    "first_frame": frame_index,
                    "last_frame": frame_index,
                }
            else:
                track_ranges[track_id]["last_frame"] = frame_index
    return track_ranges


def filter_frame_entries(
    frames: dict[str, list[dict[str, Any]]],
    selected_track_ids: list[int],
    filter_confidence: bool,
    min_confidence: float,
) -> dict[str, list[dict[str, Any]]]:
    selected_track_ids_set = set(selected_track_ids)
    min_confidence = max(0.0, float(min_confidence))
    filtered_frames: dict[str, list[dict[str, Any]]] = {}
    for frame_key, frame_entries in frames.items():
        filtered_entries: list[dict[str, Any]] = []
        for entry in frame_entries:
            track_id = int(entry["id"])
            if track_id not in selected_track_ids_set:
                continue
            if filter_confidence and float(entry["confidence"]) < min_confidence:
                continue
            filtered_entries.append(
                {
                    "bbox": list(entry["bbox"]),
                    "center_point": list(entry["center_point"]),
                    "confidence": float(entry["confidence"]),
                    "id": track_id,
                }
            )
        filtered_frames[frame_key] = filtered_entries
    return filtered_frames


def apply_export_labels(
    frames: dict[str, list[dict[str, Any]]],
    export_labels: dict[int, str],
) -> dict[str, list[dict[str, Any]]]:
    labeled_frames: dict[str, list[dict[str, Any]]] = {}
    for frame_key, frame_entries in frames.items():
        labeled_entries: list[dict[str, Any]] = []
        for entry in frame_entries:
            track_id = int(entry["id"])
            labeled_entries.append(
                {
                    "bbox": list(entry["bbox"]),
                    "center_point": list(entry["center_point"]),
                    "confidence": float(entry["confidence"]),
                    "id": track_id,
                    "label": export_labels[track_id],
                }
            )
        labeled_frames[frame_key] = labeled_entries
    return labeled_frames


def build_face_clip_segment(
    track_id: int,
    label: str,
    segment_index: int,
    segment_entries: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
    padding: float,
) -> dict[str, Any]:
    padding = max(0.0, float(padding))
    max_bbox_width = max(
        max(2, int(entry["bbox"][2]) - int(entry["bbox"][0])) for entry in segment_entries
    )
    max_bbox_height = max(
        max(2, int(entry["bbox"][3]) - int(entry["bbox"][1])) for entry in segment_entries
    )
    crop_width = min(
        frame_width,
        max(2, int(round(max_bbox_width * (1.0 + (2.0 * padding))))),
    )
    crop_height = min(
        frame_height,
        max(2, int(round(max_bbox_height * (1.0 + (2.0 * padding))))),
    )
    confidences = [float(entry["confidence"]) for entry in segment_entries]
    label_slug = slugify_label(label)
    track_stem = f"track_{track_id:04d}_{label_slug}"
    start_frame = int(segment_entries[0]["frame_index"])
    end_frame = int(segment_entries[-1]["frame_index"])
    return {
        "track_id": track_id,
        "label": label,
        "label_slug": label_slug,
        "track_stem": track_stem,
        "segment_index": segment_index,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "num_frames": len(segment_entries),
        "crop_width": crop_width,
        "crop_height": crop_height,
        "frame_entries": segment_entries,
        "frame_entry_map": {
            int(entry["frame_index"]): entry for entry in segment_entries
        },
        "confidence_min": round(min(confidences), 6),
        "confidence_max": round(max(confidences), 6),
        "confidence_mean": round(sum(confidences) / len(confidences), 6),
    }


def build_face_clip_segments(
    frames: dict[str, list[dict[str, Any]]],
    frame_width: int,
    frame_height: int,
    padding: float,
) -> list[dict[str, Any]]:
    track_entries: dict[int, dict[str, Any]] = {}
    for frame_key, frame_entries in frames.items():
        frame_index = int(frame_key)
        for entry in frame_entries:
            track_id = int(entry["id"])
            if track_id not in track_entries:
                track_entries[track_id] = {
                    "label": str(entry["label"]),
                    "entries": [],
                }
            track_entries[track_id]["entries"].append(
                {
                    "frame_index": frame_index,
                    "bbox": list(entry["bbox"]),
                    "center_point": list(entry["center_point"]),
                    "confidence": float(entry["confidence"]),
                }
            )

    segments: list[dict[str, Any]] = []
    for track_id in sorted(track_entries.keys()):
        label = str(track_entries[track_id]["label"])
        detections = sorted(
            track_entries[track_id]["entries"],
            key=lambda entry: int(entry["frame_index"]),
        )
        segment_entries: list[dict[str, Any]] = []
        segment_index = 0
        for detection in detections:
            if (
                segment_entries
                and int(detection["frame_index"])
                != int(segment_entries[-1]["frame_index"]) + 1
            ):
                segments.append(
                    build_face_clip_segment(
                        track_id=track_id,
                        label=label,
                        segment_index=segment_index,
                        segment_entries=segment_entries,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        padding=padding,
                    )
                )
                segment_entries = []
                segment_index += 1
            segment_entries.append(detection)

        if segment_entries:
            segments.append(
                build_face_clip_segment(
                    track_id=track_id,
                    label=label,
                    segment_index=segment_index,
                    segment_entries=segment_entries,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    padding=padding,
                )
            )

    return segments


def crop_bounds_from_center(
    center_point: list[int],
    crop_width: int,
    crop_height: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    crop_width = min(frame_width, max(2, crop_width))
    crop_height = min(frame_height, max(2, crop_height))
    cx, cy = int(center_point[0]), int(center_point[1])
    max_x1 = max(frame_width - crop_width, 0)
    max_y1 = max(frame_height - crop_height, 0)
    x1 = max(0, min(max_x1, int(round(cx - (crop_width / 2.0)))))
    y1 = max(0, min(max_y1, int(round(cy - (crop_height / 2.0)))))
    x2 = x1 + crop_width
    y2 = y1 + crop_height
    return x1, y1, x2, y2


def export_face_clips(
    video_path: Path,
    output_dir: Path,
    fps: float,
    width: int,
    height: int,
    frames: dict[str, list[dict[str, Any]]],
    padding: float,
    show_progress: bool,
) -> dict[str, Any]:
    face_clips_dir = output_dir / "face_clips"
    face_clips_dir.mkdir(parents=True, exist_ok=True)
    face_clips_index_path = output_dir / f"{video_path.stem}.face-clips.json"

    segments = build_face_clip_segments(
        frames=frames,
        frame_width=width,
        frame_height=height,
        padding=padding,
    )

    for segment in segments:
        track_dir = face_clips_dir / segment["track_stem"]
        track_dir.mkdir(parents=True, exist_ok=True)
        clip_file_name = (
            f"{segment['track_stem']}_seg_{int(segment['segment_index']):03d}"
            f"_frames_{int(segment['start_frame']):06d}-{int(segment['end_frame']):06d}.mp4"
        )
        clip_path = track_dir / clip_file_name
        segment["clip_path"] = clip_path
        segment["relative_clip_path"] = str(clip_path.relative_to(output_dir))

    if segments:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to reopen video for face clip export: {video_path}")

        active_writers: dict[int, cv2.VideoWriter] = {}
        segments_by_start: dict[int, list[dict[str, Any]]] = {}
        segments_by_end: dict[int, list[dict[str, Any]]] = {}
        for segment_id, segment in enumerate(segments):
            segment["segment_id"] = segment_id
            segments_by_start.setdefault(int(segment["start_frame"]), []).append(segment)
            segments_by_end.setdefault(int(segment["end_frame"]), []).append(segment)

        face_clip_total_frames = max(int(segment["end_frame"]) for segment in segments) + 1
        face_clip_start_time = perf_counter()
        last_progress_time = face_clip_start_time

        try:
            frame_index = 0
            while frame_index < face_clip_total_frames:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(
                        f"Unable to read frame {frame_index} while exporting face clips."
                    )

                for segment in segments_by_start.get(frame_index, []):
                    writer = cv2.VideoWriter(
                        str(segment["clip_path"]),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps if fps > 0 else 30.0,
                        (int(segment["crop_width"]), int(segment["crop_height"])),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Unable to open face clip writer: {segment['clip_path']}"
                        )
                    active_writers[int(segment["segment_id"])] = writer

                for segment_id, writer in active_writers.items():
                    segment = segments[segment_id]
                    frame_entry = segment["frame_entry_map"].get(frame_index)
                    if frame_entry is None:
                        continue
                    x1, y1, x2, y2 = crop_bounds_from_center(
                        center_point=list(frame_entry["center_point"]),
                        crop_width=int(segment["crop_width"]),
                        crop_height=int(segment["crop_height"]),
                        frame_width=width,
                        frame_height=height,
                    )
                    crop = frame[y1:y2, x1:x2]
                    writer.write(crop)

                for segment in segments_by_end.get(frame_index, []):
                    writer = active_writers.pop(int(segment["segment_id"]), None)
                    if writer is not None:
                        writer.release()

                frame_index += 1
                if show_progress:
                    now = perf_counter()
                    if (
                        frame_index == 1
                        or frame_index % 100 == 0
                        or now - last_progress_time >= 10
                    ):
                        elapsed = now - face_clip_start_time
                        export_fps = frame_index / elapsed if elapsed > 0 else 0.0
                        remaining_frames = max(face_clip_total_frames - frame_index, 0)
                        eta_seconds = (
                            remaining_frames / export_fps if export_fps > 0 else 0.0
                        )
                        print(
                            f"[movie-like-shots] face clips {frame_index}/{face_clip_total_frames} "
                            f"({export_fps:.2f} fps, eta {eta_seconds/60:.1f} min)"
                        )
                        last_progress_time = now
        finally:
            cap.release()
            for writer in active_writers.values():
                writer.release()

    face_clip_index = {
        "video_metadata": {
            "fps": round(fps, 3),
            "width": width,
            "height": height,
        },
        "export_settings": {
            "padding": round(max(0.0, float(padding)), 3),
        },
        "num_clips_total": len(segments),
        "clips": [
            {
                "track_id": int(segment["track_id"]),
                "label": str(segment["label"]),
                "segment_index": int(segment["segment_index"]),
                "start_frame": int(segment["start_frame"]),
                "end_frame": int(segment["end_frame"]),
                "num_frames": int(segment["num_frames"]),
                "clip_path": str(segment["relative_clip_path"]),
                "crop_size": {
                    "width": int(segment["crop_width"]),
                    "height": int(segment["crop_height"]),
                },
                "confidence": {
                    "min": float(segment["confidence_min"]),
                    "max": float(segment["confidence_max"]),
                    "mean": float(segment["confidence_mean"]),
                },
            }
            for segment in segments
        ],
    }

    with face_clips_index_path.open("w", encoding="utf-8") as f:
        json.dump(face_clip_index, f, indent=2)

    return {
        "face_clips_dir": str(face_clips_dir),
        "face_clips_index_path": str(face_clips_index_path),
        "num_clips_total": len(segments),
    }


def write_filtered_preview(
    video_path: Path,
    preview_path: Path,
    fps: float,
    width: int,
    height: int,
    frames: dict[str, list[dict[str, Any]]],
    show_progress: bool,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to reopen video for filtered preview: {video_path}")

    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open preview writer: {preview_path}")

    preview_total_frames = len(frames)
    preview_start_time = perf_counter()
    last_progress_time = preview_start_time

    try:
        for preview_index, frame_entries in enumerate(frames.values(), start=1):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(
                    f"Unable to read frame {preview_index - 1} while rendering filtered preview."
                )
            writer.write(draw_preview_frame(frame, frame_entries))

            if show_progress:
                now = perf_counter()
                if (
                    preview_index == 1
                    or preview_index % 100 == 0
                    or now - last_progress_time >= 10
                ):
                    elapsed = now - preview_start_time
                    preview_fps = preview_index / elapsed if elapsed > 0 else 0.0
                    remaining_frames = max(preview_total_frames - preview_index, 0)
                    eta_seconds = (
                        remaining_frames / preview_fps if preview_fps > 0 else 0.0
                    )
                    print(
                        f"[movie-like-shots] preview {preview_index}/{preview_total_frames} "
                        f"({preview_fps:.2f} fps, eta {eta_seconds/60:.1f} min)"
                    )
                    last_progress_time = now
    finally:
        cap.release()
        writer.release()


def run_pipeline(
    video_path: Path,
    output_dir: Path | None = None,
    detector_model: Path | None = None,
    reid_model: Path | None = None,
    tracker_type: str = "ocsort",
    detector_size: int = 640,
    det_thresh: float = 0.35,
    nms_thresh: float = 0.4,
    num_bins: int = 64,
    shot_change_threshold: float = 0.4,
    use_shot_change: bool = True,
    use_shared_memory: bool = True,
    device: str = "auto",
    max_frames: int | None = None,
    write_preview: bool = False,
    filter_tracks: bool = False,
    min_track_length: int = 10,
    min_track_median_area: float = 2500.0,
    filter_confidence: bool = False,
    min_confidence: float = 0.5,
    write_face_clips: bool = False,
    face_clip_padding: float = 0.15,
    show_progress: bool = True,
) -> dict[str, str | None]:
    video_path = video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    detector_model = detector_model.resolve() if detector_model else default_detector_model()

    ensure_detector_weights(detector_model)
    if reid_model is not None:
        reid_model = reid_model.resolve()
    elif tracker_requires_reid(tracker_type):
        reid_model = default_reid_model()

    if tracker_requires_reid(tracker_type):
        if reid_model is None or not reid_model.exists():
            raise FileNotFoundError(
                "A ReID checkpoint is required for the selected tracker. "
                f"Expected: {reid_model}"
            )

    ensure_bot_repo_on_path()
    from tracker.tracker_zoo import create_tracker

    calculate_histogram = None
    chi_square_distance = None
    if tracker_type == "botfacesort" and use_shot_change:
        from utils.histogram import calculate_histogram, chi_square_distance

    device_obj = resolve_tracker_device(tracker_type, device)
    SCRFD = load_insightface_scrfd_class()

    detector = SCRFD(model_file=str(detector_model))
    detector_providers, detector_ctx_id = resolve_detector_execution(
        device, detector_size
    )
    detector.session.set_providers(detector_providers)
    detector.prepare(
        ctx_id=detector_ctx_id,
        det_thresh=det_thresh,
        nms_thresh=nms_thresh,
        input_size=(detector_size, detector_size),
    )

    tracker_config = bot_root() / "tracker" / "configs" / f"{tracker_type}.yaml"
    tracker = create_tracker(
        tracker_type,
        tracker_config,
        reid_model,
        device_obj,
        False,
        sc=use_shot_change if tracker_type == "botfacesort" else False,
        sm=use_shared_memory if tracker_type == "botfacesort" else False,
    )

    if output_dir is None:
        output_dir = repo_root() / "outputs" / video_path.stem
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{video_path.stem}.tracks.json"
    preview_path = output_dir / f"{video_path.stem}.preview.mp4" if write_preview else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = None
    requires_filtered_preview = filter_tracks or filter_confidence
    if preview_path is not None and not requires_filtered_preview:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(preview_path),
            fourcc,
            fps if fps > 0 else 30.0,
            (width, height),
        )

    frames: dict[str, list[dict[str, Any]]] = {}
    track_ranges: dict[int, dict[str, int]] = {}
    track_stats: dict[int, dict[str, Any]] = {}
    labels: dict[int, str] = {}
    prev_frame: np.ndarray | None = None
    frame_index = 0
    start_time = perf_counter()
    last_progress_time = start_time

    while True:
        if max_frames is not None and frame_index >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        raw_dets, raw_landmarks = detector.detect(
            frame, input_size=(detector_size, detector_size)
        )

        if raw_dets is None or len(raw_dets) == 0:
            face_dets = np.empty((0, 6), dtype=np.float32)
            landmarks = None
        else:
            face_dets = np.insert(raw_dets.astype(np.float32), 5, 1.0, axis=1)
            landmarks = raw_landmarks if raw_landmarks is not None and len(raw_landmarks) else None

        shot_changed = False
        if tracker_type == "botfacesort" and use_shot_change and prev_frame is not None:
            curr_hist = calculate_histogram(frame, num_bins)
            prev_hist = calculate_histogram(prev_frame, num_bins)
            shot_changed = (
                chi_square_distance(curr_hist, prev_hist) > shot_change_threshold
            )

        if tracker_type == "botfacesort" and use_shot_change:
            tracker_outputs = tracker.update(
                face_dets,
                frame,
                shot_changed=shot_changed,
                landmarks=landmarks,
            )
        elif tracker_type == "botfacesort":
            tracker_outputs = tracker.update(face_dets, frame, landmarks=landmarks)
        else:
            tracker_outputs = tracker.update(face_dets, frame)

        frame_entries: list[dict[str, Any]] = []
        if tracker_outputs is not None and len(tracker_outputs) > 0:
            new_track_ids = sorted(
                {int(row[4]) for row in tracker_outputs if int(row[4]) not in labels}
            )
            for track_id in new_track_ids:
                labels[track_id] = label_for_index(len(labels))

            for row in tracker_outputs:
                track_id = int(row[4])
                bbox = round_and_clip_bbox(row[:4], width, height)
                center_point = center_point_from_bbox(bbox)
                confidence = confidence_from_tracker_row(row)

                frame_entries.append(
                    {
                        "bbox": bbox,
                        "center_point": center_point,
                        "confidence": confidence,
                        "id": track_id,
                        "label": labels[track_id],
                    }
                )

                if track_id not in track_ranges:
                    track_ranges[track_id] = {
                        "first_frame": frame_index,
                        "last_frame": frame_index,
                    }
                else:
                    track_ranges[track_id]["last_frame"] = frame_index

                if track_id not in track_stats:
                    track_stats[track_id] = {"frame_count": 1, "areas": [bbox_area(bbox)]}
                else:
                    track_stats[track_id]["frame_count"] += 1
                    track_stats[track_id]["areas"].append(bbox_area(bbox))

        frames[str(frame_index)] = frame_entries
        if writer is not None:
            writer.write(draw_preview_frame(frame, frame_entries))
        if tracker_type == "botfacesort" and use_shot_change:
            prev_frame = frame.copy()
        frame_index += 1

        if show_progress and frame_index > 0:
            now = perf_counter()
            if frame_index == 1 or frame_index % 100 == 0 or now - last_progress_time >= 10:
                elapsed = now - start_time
                fps_processed = frame_index / elapsed if elapsed > 0 else 0.0
                if total_frames > 0:
                    remaining_frames = max(total_frames - frame_index, 0)
                    eta_seconds = remaining_frames / fps_processed if fps_processed > 0 else 0.0
                    print(
                        f"[movie-like-shots] frame {frame_index}/{total_frames} "
                        f"({fps_processed:.2f} fps, eta {eta_seconds/60:.1f} min)"
                    )
                else:
                    print(
                        f"[movie-like-shots] frame {frame_index} "
                        f"({fps_processed:.2f} fps)"
                    )
                last_progress_time = now

    cap.release()
    if writer is not None:
        writer.release()

    selected_track_ids = select_track_ids_for_export(
        track_ranges=track_ranges,
        track_stats=track_stats,
        filter_tracks=filter_tracks,
        min_track_length=min_track_length,
        min_track_median_area=min_track_median_area,
    )
    filtered_frames = filter_frame_entries(
        frames=frames,
        selected_track_ids=selected_track_ids,
        filter_confidence=filter_confidence,
        min_confidence=min_confidence,
    )
    export_track_ranges = build_track_ranges_from_frames(filtered_frames)
    export_track_ids = ordered_track_ids_for_ranges(export_track_ranges)
    export_labels = build_export_labels(export_track_ids)
    export_frames = apply_export_labels(filtered_frames, export_labels)
    face_clip_result: dict[str, Any] | None = None

    if preview_path is not None and requires_filtered_preview:
        if show_progress:
            filter_parts: list[str] = []
            if filter_tracks:
                filter_parts.append(
                    "min_track_length="
                    f"{max(0, min_track_length)}, min_track_median_area="
                    f"{max(0.0, float(min_track_median_area)):.0f}"
                )
            if filter_confidence:
                filter_parts.append(
                    f"min_confidence={max(0.0, float(min_confidence)):.3f}"
                )
            print(
                "[movie-like-shots] rendering filtered preview "
                f"({', '.join(filter_parts)})"
            )
        write_filtered_preview(
            video_path=video_path,
            preview_path=preview_path,
            fps=fps,
            width=width,
            height=height,
            frames=export_frames,
            show_progress=show_progress,
        )

    if write_face_clips:
        if show_progress:
            print(
                "[movie-like-shots] exporting face clips "
                f"(padding={max(0.0, float(face_clip_padding)):.2f})"
            )
        face_clip_result = export_face_clips(
            video_path=video_path,
            output_dir=output_dir,
            fps=fps,
            width=width,
            height=height,
            frames=frames,
            padding=face_clip_padding,
            show_progress=show_progress,
        )

    export = {
        "video_metadata": {
            "fps": round(fps, 3),
            "width": width,
            "height": height,
        },
        "track_summary": {
            "num_ids_total": len(export_track_ids),
            "ids": [
                {
                    "id": track_id,
                    "label": export_labels[track_id],
                    "first_frame": export_track_ranges[track_id]["first_frame"],
                    "last_frame": export_track_ranges[track_id]["last_frame"],
                }
                for track_id in export_track_ids
            ],
        },
        "frames": export_frames,
    }
    if face_clip_result is not None:
        export["face_clips"] = {
            "clips_dir": str(Path(face_clip_result["face_clips_dir"]).name),
            "index_path": str(Path(face_clip_result["face_clips_index_path"]).name),
            "num_clips_total": int(face_clip_result["num_clips_total"]),
            "source": "raw_unfiltered_tracks",
        }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)

    return {
        "json_path": str(json_path),
        "preview_path": str(preview_path) if preview_path is not None else None,
        "face_clips_dir": (
            str(face_clip_result["face_clips_dir"]) if face_clip_result is not None else None
        ),
        "face_clips_index_path": (
            str(face_clip_result["face_clips_index_path"])
            if face_clip_result is not None
            else None
        ),
    }
