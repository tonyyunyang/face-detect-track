from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run face detection + tracking on a video clip and export JSON."
    )
    parser.add_argument("video_path", type=Path, help="Path to the input video clip.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root output directory. Defaults to ./outputs/<video-stem>/",
    )
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=None,
        help="Path to the SCRFD ONNX detector. Defaults to ./models/scrfd/scrfd_10g_gnkps.onnx",
    )
    parser.add_argument(
        "--reid-model",
        type=Path,
        default=None,
        help="Path to the AdaFace ReID checkpoint. Only needed for appearance-based trackers such as botfacesort.",
    )
    parser.add_argument(
        "--tracker",
        choices=["ocsort", "bytetrack", "botfacesort"],
        default="ocsort",
        help="Tracker implementation from BoT-FaceSORT-VEED. Default: ocsort",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=640,
        help="SCRFD detector input size.",
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
        "--num-bins",
        type=int,
        default=64,
        help="Histogram bins for shot-change detection.",
    )
    parser.add_argument(
        "--shot-change-threshold",
        type=float,
        default=0.4,
        help="Shot-change threshold for histogram distance.",
    )
    parser.add_argument(
        "--disable-shot-change",
        action="store_true",
        help="Disable shot-change-aware tracking.",
    )
    parser.add_argument(
        "--disable-shared-memory",
        action="store_true",
        help="Disable shared-memory matching in BoT-FaceSORT.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device. auto uses CoreML on Apple Silicon at det-size 640, CUDA on supported NVIDIA setups, otherwise CPU.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for smoke tests.",
    )
    parser.add_argument(
        "--write-preview",
        action="store_true",
        help="Write a preview MP4 with IDs drawn on frames.",
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
        help="Export cropped face video clips for every raw tracked face segment.",
    )
    parser.add_argument(
        "--face-clip-padding",
        type=float,
        default=0.15,
        help="Fractional padding applied to each side of the face clip crop window.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_pipeline(
        video_path=args.video_path,
        output_dir=args.output_dir,
        detector_model=args.detector_model,
        reid_model=args.reid_model,
        tracker_type=args.tracker,
        detector_size=args.det_size,
        det_thresh=args.det_thresh,
        nms_thresh=args.nms_thresh,
        num_bins=args.num_bins,
        shot_change_threshold=args.shot_change_threshold,
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
    )

    print(f"JSON: {result['json_path']}")
    if result["preview_path"] is not None:
        print(f"Preview: {result['preview_path']}")
    if result["face_clips_index_path"] is not None:
        print(f"Face clips index: {result['face_clips_index_path']}")
    if result["face_clips_dir"] is not None:
        print(f"Face clips dir: {result['face_clips_dir']}")


if __name__ == "__main__":
    main()
