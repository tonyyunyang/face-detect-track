#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from movie_like_shots.pipeline import (
    available_onnx_providers,
    bot_root,
    center_point_from_bbox,
    default_detector_model,
    default_reid_model,
    draw_preview_frame,
    ensure_bot_repo_on_path,
    ensure_detector_weights,
    label_for_index,
    load_insightface_scrfd_class,
    resolve_detector_execution,
    resolve_tracker_device,
    round_and_clip_bbox,
    run_pipeline,
    tracker_requires_reid,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile the movie-like-shots pipeline and verify detector providers."
    )
    parser.add_argument("video_path", type=Path, help="Path to the input video clip.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for the full export run.",
    )
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=None,
        help="Path to the SCRFD ONNX detector.",
    )
    parser.add_argument(
        "--reid-model",
        type=Path,
        default=None,
        help="Path to the ReID checkpoint for appearance-based trackers.",
    )
    parser.add_argument(
        "--tracker",
        choices=["ocsort", "bytetrack", "botfacesort"],
        default="ocsort",
        help="Tracker implementation to profile. Default: ocsort",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=640,
        help="SCRFD detector input size. Default: 640",
    )
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=0.35,
        help="Face detection confidence threshold.",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.4,
        help="Face detector NMS threshold.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device. auto should use CoreML on Apple Silicon with det-size 640.",
    )
    parser.add_argument(
        "--profile-frames",
        type=int,
        default=120,
        help="Number of frames to use for the stage breakdown profile. Default: 120",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Run the full export after the stage profile.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame cap for the full export run.",
    )
    parser.add_argument(
        "--write-preview",
        action="store_true",
        help="Include preview video writing in the stage profile and full export.",
    )
    parser.add_argument(
        "--filter-tracks",
        action="store_true",
        help="Filter short or tiny tracks out of the final JSON and preview export.",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=10,
        help="Minimum detected-frame count required when --filter-tracks is enabled.",
    )
    parser.add_argument(
        "--min-track-median-area",
        type=float,
        default=2500.0,
        help="Minimum median bbox area required when --filter-tracks is enabled.",
    )
    parser.add_argument(
        "--filter-confidence",
        action="store_true",
        help="Filter low-confidence tracked detections out of the final JSON and preview export.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum tracked-detection confidence required when --filter-confidence is enabled.",
    )
    parser.add_argument(
        "--write-face-clips",
        action="store_true",
        help="Export cropped face video clips for every raw tracked face segment during the full run.",
    )
    parser.add_argument(
        "--face-clip-padding",
        type=float,
        default=0.15,
        help="Fractional padding applied to each side of the face clip crop window.",
    )
    parser.add_argument(
        "--disable-shot-change",
        action="store_true",
        help="Disable shot-change-aware tracking for botfacesort.",
    )
    parser.add_argument(
        "--disable-shared-memory",
        action="store_true",
        help="Disable shared-memory matching for botfacesort.",
    )
    parser.add_argument(
        "--require-coreml",
        action="store_true",
        help="Fail if CoreMLExecutionProvider is not active in the detector session.",
    )
    return parser


def resolve_models(args: argparse.Namespace) -> tuple[Path, Path | None]:
    detector_model = (
        args.detector_model.resolve()
        if args.detector_model is not None
        else default_detector_model()
    )
    ensure_detector_weights(detector_model)

    reid_model: Path | None
    if args.reid_model is not None:
        reid_model = args.reid_model.resolve()
    elif tracker_requires_reid(args.tracker):
        reid_model = default_reid_model()
    else:
        reid_model = None

    if tracker_requires_reid(args.tracker):
        if reid_model is None or not reid_model.exists():
            raise FileNotFoundError(
                "A ReID checkpoint is required for the selected tracker. "
                f"Expected: {reid_model}"
            )
    return detector_model, reid_model


def build_detector(
    detector_model: Path,
    device: str,
    detector_size: int,
    det_thresh: float,
    nms_thresh: float,
):
    SCRFD = load_insightface_scrfd_class()
    detector = SCRFD(model_file=str(detector_model))
    selected_providers, ctx_id = resolve_detector_execution(device, detector_size)
    detector.session.set_providers(selected_providers)
    detector.prepare(
        ctx_id=ctx_id,
        det_thresh=det_thresh,
        nms_thresh=nms_thresh,
        input_size=(detector_size, detector_size),
    )
    active_providers = detector.session.get_providers()
    return detector, selected_providers, active_providers


def build_tracker(
    tracker_type: str,
    reid_model: Path | None,
    device: str,
    use_shot_change: bool,
    use_shared_memory: bool,
):
    ensure_bot_repo_on_path()
    from tracker.tracker_zoo import create_tracker

    tracker_config = bot_root() / "tracker" / "configs" / f"{tracker_type}.yaml"
    device_obj = resolve_tracker_device(tracker_type, device)
    return create_tracker(
        tracker_type,
        tracker_config,
        reid_model,
        device_obj,
        False,
        sc=use_shot_change if tracker_type == "botfacesort" else False,
        sm=use_shared_memory if tracker_type == "botfacesort" else False,
    )


def print_header(
    video_path: Path,
    detector_model: Path,
    selected_providers: list[str],
    active_providers: list[str],
    args: argparse.Namespace,
) -> None:
    print(f"[profile] repo: {REPO_ROOT}")
    print(f"[profile] python: {sys.executable}")
    print(f"[profile] platform: {platform.platform()}")
    print(f"[profile] video: {video_path.resolve()}")
    print(f"[profile] detector model: {detector_model}")
    print(f"[profile] available onnxruntime providers: {list(available_onnx_providers())}")
    print(f"[profile] selected detector providers: {selected_providers}")
    print(f"[profile] active detector session providers: {active_providers}")
    print(
        "[profile] export filter: "
        f"enabled={args.filter_tracks}, "
        f"min_track_length={args.min_track_length}, "
        f"min_track_median_area={args.min_track_median_area}"
    )
    print(
        "[profile] confidence filter: "
        f"enabled={args.filter_confidence}, "
        f"min_confidence={args.min_confidence}"
    )
    print(
        "[profile] face clips: "
        f"enabled={args.write_face_clips}, "
        f"padding={args.face_clip_padding}"
    )


def profile_stages(
    args: argparse.Namespace,
    detector_model: Path,
    reid_model: Path | None,
) -> None:
    detector, selected_providers, active_providers = build_detector(
        detector_model,
        args.device,
        args.det_size,
        args.det_thresh,
        args.nms_thresh,
    )
    print_header(
        args.video_path,
        detector_model,
        selected_providers,
        active_providers,
        args,
    )

    coreml_active = "CoreMLExecutionProvider" in active_providers
    print(f"[profile] coreml active: {coreml_active}")
    if args.require_coreml and not coreml_active:
        raise SystemExit("CoreMLExecutionProvider is not active in the detector session.")

    tracker = build_tracker(
        args.tracker,
        reid_model,
        args.device,
        use_shot_change=not args.disable_shot_change,
        use_shared_memory=not args.disable_shared_memory,
    )

    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(
        f"[profile] clip metadata: fps={fps:.3f}, frames={total_frames}, size={width}x{height}"
    )

    writer = None
    preview_path = None
    if args.write_preview:
        preview_path = (
            Path(tempfile.gettempdir()) / "movie_like_shots_profile_preview.mp4"
        )
        writer = cv2.VideoWriter(
            str(preview_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else 30.0,
            (width, height),
        )

    read_t = 0.0
    detect_t = 0.0
    track_t = 0.0
    preview_t = 0.0
    total_t = 0.0
    frames = 0
    labels: dict[int, str] = {}
    prev_frame = None

    calculate_histogram = None
    chi_square_distance = None
    if args.tracker == "botfacesort" and not args.disable_shot_change:
        from utils.histogram import calculate_histogram, chi_square_distance

    while frames < args.profile_frames:
        total_start = perf_counter()
        read_start = perf_counter()
        ret, frame = cap.read()
        read_t += perf_counter() - read_start
        if not ret:
            break

        detect_start = perf_counter()
        raw_dets, raw_landmarks = detector.detect(
            frame, input_size=(args.det_size, args.det_size)
        )
        detect_t += perf_counter() - detect_start

        if raw_dets is None or len(raw_dets) == 0:
            face_dets = np.empty((0, 6), dtype=np.float32)
            landmarks = None
        else:
            face_dets = np.insert(raw_dets.astype(np.float32), 5, 1.0, axis=1)
            landmarks = (
                raw_landmarks
                if raw_landmarks is not None and len(raw_landmarks)
                else None
            )

        shot_changed = False
        if (
            args.tracker == "botfacesort"
            and not args.disable_shot_change
            and prev_frame is not None
        ):
            curr_hist = calculate_histogram(frame, 64)
            prev_hist = calculate_histogram(prev_frame, 64)
            shot_changed = chi_square_distance(curr_hist, prev_hist) > 0.4

        track_start = perf_counter()
        if args.tracker == "botfacesort" and not args.disable_shot_change:
            tracker_outputs = tracker.update(
                face_dets,
                frame,
                shot_changed=shot_changed,
                landmarks=landmarks,
            )
        elif args.tracker == "botfacesort":
            tracker_outputs = tracker.update(face_dets, frame, landmarks=landmarks)
        else:
            tracker_outputs = tracker.update(face_dets, frame)
        track_t += perf_counter() - track_start

        if writer is not None:
            frame_entries = []
            if tracker_outputs is not None and len(tracker_outputs) > 0:
                new_track_ids = sorted(
                    {int(row[4]) for row in tracker_outputs if int(row[4]) not in labels}
                )
                for track_id in new_track_ids:
                    labels[track_id] = label_for_index(len(labels))
                for row in tracker_outputs:
                    track_id = int(row[4])
                    bbox = round_and_clip_bbox(row[:4], width, height)
                    frame_entries.append(
                        {
                            "bbox": bbox,
                            "center_point": center_point_from_bbox(bbox),
                            "id": track_id,
                            "label": labels[track_id],
                        }
                    )

            preview_start = perf_counter()
            writer.write(draw_preview_frame(frame, frame_entries))
            preview_t += perf_counter() - preview_start

        if args.tracker == "botfacesort" and not args.disable_shot_change:
            prev_frame = frame.copy()

        total_t += perf_counter() - total_start
        frames += 1

    cap.release()
    if writer is not None:
        writer.release()
        if preview_path is not None and preview_path.exists():
            preview_path.unlink()

    if frames == 0:
        raise RuntimeError("No frames were processed during the stage profile.")

    print(f"[profile] stage profile frames: {frames}")
    print(f"[profile] read_ms_per_frame: {read_t * 1000 / frames:.2f}")
    print(f"[profile] detect_ms_per_frame: {detect_t * 1000 / frames:.2f}")
    print(f"[profile] track_ms_per_frame: {track_t * 1000 / frames:.2f}")
    if args.write_preview:
        print(f"[profile] preview_ms_per_frame: {preview_t * 1000 / frames:.2f}")
    print(f"[profile] total_ms_per_frame: {total_t * 1000 / frames:.2f}")
    if args.filter_tracks and args.write_preview:
        print(
            "[profile] note: filtered preview is rendered in a second pass during the full run."
        )
    if args.filter_confidence and args.write_preview:
        print(
            "[profile] note: confidence-filtered preview is rendered in a second pass during the full run."
        )
    if args.write_face_clips:
        print(
            "[profile] note: face clip export runs in a separate pass during the full run."
        )


def maybe_run_full_export(
    args: argparse.Namespace,
    detector_model: Path,
    reid_model: Path | None,
) -> None:
    if not args.full_run:
        return

    print("[profile] starting full export run")
    start = perf_counter()
    result = run_pipeline(
        video_path=args.video_path,
        output_dir=args.output_dir,
        detector_model=detector_model,
        reid_model=reid_model,
        tracker_type=args.tracker,
        detector_size=args.det_size,
        det_thresh=args.det_thresh,
        nms_thresh=args.nms_thresh,
        use_shot_change=not args.disable_shot_change,
        use_shared_memory=not args.disable_shared_memory,
        device=args.device,
        max_frames=args.max_frames,
        write_preview=args.write_preview,
        filter_tracks=args.filter_tracks,
        min_track_length=args.min_track_length,
        min_track_median_area=args.min_track_median_area,
        filter_confidence=args.filter_confidence,
        min_confidence=args.min_confidence,
        write_face_clips=args.write_face_clips,
        face_clip_padding=args.face_clip_padding,
        show_progress=True,
    )
    elapsed = perf_counter() - start

    json_path = Path(result["json_path"])
    data = json.loads(json_path.read_text())
    processed_frames = len(data["frames"])
    processed_fps = processed_frames / elapsed if elapsed > 0 else 0.0

    print(f"[profile] full_run_elapsed_s: {elapsed:.2f}")
    print(f"[profile] full_run_processed_frames: {processed_frames}")
    print(f"[profile] full_run_processed_fps: {processed_fps:.2f}")
    print(f"[profile] json_path: {result['json_path']}")
    if result["preview_path"] is not None:
        print(f"[profile] preview_path: {result['preview_path']}")
    if result["face_clips_index_path"] is not None:
        print(f"[profile] face_clips_index_path: {result['face_clips_index_path']}")
    if result["face_clips_dir"] is not None:
        print(f"[profile] face_clips_dir: {result['face_clips_dir']}")


def main() -> None:
    args = build_parser().parse_args()
    detector_model, reid_model = resolve_models(args)
    profile_stages(args, detector_model, reid_model)
    maybe_run_full_export(args, detector_model, reid_model)


if __name__ == "__main__":
    main()
