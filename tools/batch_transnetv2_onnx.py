#!/usr/bin/env python3
"""Batch TransNetV2 ONNX transition detection for local video files."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from multiprocessing import Lock, Value
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
FRAME_HEIGHT = 27
FRAME_WIDTH = 48
FRAME_CHANNELS = 3
WINDOW_SIZE = 100
WINDOW_STRIDE = 50
WINDOW_KEEP_START = 25
WINDOW_KEEP_END = 75

SESSION: ort.InferenceSession | None = None
INPUT_NAME: str | None = None
ASSIGNED_DEVICE_ID: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TransNetV2 ONNX over a directory of videos.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing videos.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for JSONL summary outputs.")
    parser.add_argument("--model", required=True, type=Path, help="Path to transnetv2.onnx.")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--provider", default="CPUExecutionProvider", help="ONNX Runtime provider.")
    parser.add_argument("--device-ids", default=None, help="Comma-separated GPU device ids for CUDAExecutionProvider.")
    parser.add_argument("--limit", type=int, default=None, help="Optional video limit for smoke tests.")
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively.")
    parser.add_argument("--resume", action="store_true", help="Skip videos already present in results.jsonl.")
    parser.add_argument("--save-predictions", action="store_true", help="Write per-frame predictions as .npz files.")
    return parser.parse_args()


def parse_device_ids(value: str | None) -> list[int]:
    if not value:
        return []
    device_ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        device_ids.append(int(part))
    return device_ids


def init_worker(model_path: str, provider: str, device_ids: list[int], device_counter: Any, device_lock: Any) -> None:
    global SESSION, INPUT_NAME, ASSIGNED_DEVICE_ID
    if provider == "CUDAExecutionProvider" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    available = ort.get_available_providers()
    provider_config: list[Any]
    if provider == "CUDAExecutionProvider" and provider in available:
        if device_ids:
            with device_lock:
                device_id = device_ids[device_counter.value % len(device_ids)]
                device_counter.value += 1
            ASSIGNED_DEVICE_ID = device_id
            provider_config = [("CUDAExecutionProvider", {"device_id": device_id, "use_tf32": 0}), "CPUExecutionProvider"]
        else:
            provider_config = [("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"]
    else:
        provider_config = [provider] if provider in available else ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    SESSION = ort.InferenceSession(model_path, sess_options=sess_options, providers=provider_config)
    INPUT_NAME = SESSION.get_inputs()[0].name


def list_videos(input_dir: Path, *, recursive: bool, limit: int | None) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    videos = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    return videos[:limit] if limit is not None else videos


def read_completed(results_path: Path) -> set[str]:
    if not results_path.is_file():
        return set()
    completed: set[str] = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        video_path = row.get("video_path")
        if isinstance(video_path, str):
            completed.add(video_path)
    return completed


def run_json_command(cmd: list[str]) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def probe_video(path: Path) -> dict[str, Any]:
    data = run_json_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,nb_frames,duration,width,height",
            "-of",
            "json",
            str(path),
        ]
    )
    if not data:
        return {}
    streams = data.get("streams") or []
    if not streams:
        return {}
    stream = streams[0]
    fps = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(key)
        if isinstance(value, str) and value and value != "0/0":
            try:
                fps = float(Fraction(value))
                break
            except (ValueError, ZeroDivisionError):
                pass
    duration = None
    try:
        duration = float(stream["duration"]) if stream.get("duration") is not None else None
    except (TypeError, ValueError):
        duration = None
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps,
        "duration_sec": duration,
        "nb_frames": stream.get("nb_frames"),
    }


def decode_video(path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed: {stderr.strip() or exc}") from exc

    frame_size = FRAME_HEIGHT * FRAME_WIDTH * FRAME_CHANNELS
    if len(completed.stdout) % frame_size != 0:
        raise RuntimeError(
            f"decoded byte count is not divisible by frame size: bytes={len(completed.stdout)} frame_size={frame_size}"
        )
    frames = np.frombuffer(completed.stdout, np.uint8)
    frames = frames.reshape((-1, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS))
    if len(frames) == 0:
        raise RuntimeError("ffmpeg decoded zero frames")
    return frames


def iter_windows(frames: np.ndarray) -> Any:
    remainder = len(frames) % WINDOW_STRIDE
    no_padded_frames_end = WINDOW_KEEP_START + WINDOW_STRIDE - (remainder if remainder != 0 else WINDOW_STRIDE)
    start_frame = frames[0:1]
    end_frame = frames[-1:]
    padded = np.concatenate(
        [np.repeat(start_frame, WINDOW_KEEP_START, axis=0), frames, np.repeat(end_frame, no_padded_frames_end, axis=0)],
        axis=0,
    )
    ptr = 0
    while ptr + WINDOW_SIZE <= len(padded):
        yield padded[ptr : ptr + WINDOW_SIZE][np.newaxis].astype(np.float32, copy=False)
        ptr += WINDOW_STRIDE


def predict_frames(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if SESSION is None or INPUT_NAME is None:
        raise RuntimeError("ONNX session was not initialized.")

    single_predictions: list[np.ndarray] = []
    many_predictions: list[np.ndarray] = []
    for window in iter_windows(frames):
        outputs = SESSION.run(None, {INPUT_NAME: window})
        single_predictions.append(outputs[0][0, WINDOW_KEEP_START:WINDOW_KEEP_END, 0])
        many_predictions.append(outputs[1][0, WINDOW_KEEP_START:WINDOW_KEEP_END, 0])

    single = np.concatenate(single_predictions, axis=0)[: len(frames)]
    many = np.concatenate(many_predictions, axis=0)[: len(frames)]
    return single, many


def positive_segments(predictions: np.ndarray, threshold: float) -> list[list[int]]:
    positive = predictions > threshold
    segments: list[list[int]] = []
    start: int | None = None
    for index, is_positive in enumerate(positive):
        if is_positive and start is None:
            start = index
        elif not is_positive and start is not None:
            segments.append([start, index - 1])
            start = None
    if start is not None:
        segments.append([start, len(positive) - 1])
    return segments


def predictions_to_scenes(predictions: np.ndarray, threshold: float) -> list[list[int]]:
    binary = (predictions > threshold).astype(np.uint8)
    scenes: list[list[int]] = []
    t = -1
    t_prev = 0
    start = 0
    for i, t in enumerate(binary):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = int(t)
    if t == 0:
        scenes.append([start, i])
    if not scenes:
        return [[0, len(predictions) - 1]]
    return scenes


def process_video(video_path: str, threshold: float, predictions_dir: str | None) -> dict[str, Any]:
    started = time.monotonic()
    path = Path(video_path)
    row: dict[str, Any] = {
        "video_path": str(path),
        "filename": path.name,
        "status": "failed",
        "error": None,
        "has_transition": False,
        "transition_count": 0,
        "scene_count": 0,
        "frame_count": 0,
        "threshold": threshold,
        "onnx_device_id": ASSIGNED_DEVICE_ID,
    }
    try:
        probe = probe_video(path)
        frames = decode_video(path)
        single, many = predict_frames(frames)
        segments = positive_segments(single, threshold)
        scenes = predictions_to_scenes(single, threshold)
        fps = probe.get("fps") if isinstance(probe, dict) else None
        transition_seconds = None
        if fps:
            transition_seconds = [round(segment[0] / float(fps), 3) for segment in segments]

        predictions_key = None
        if predictions_dir is not None:
            pred_dir = Path(predictions_dir)
            safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in path.stem).strip("._") or "video"
            pred_path = pred_dir / f"{safe_name}.predictions.npz"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pred_path, single_frame=single, many_hot=many)
            predictions_key = str(pred_path)

        row.update(
            {
                "status": "done",
                "has_transition": bool(segments),
                "transition_count": len(segments),
                "scene_count": len(scenes),
                "frame_count": int(len(frames)),
                "duration_sec": probe.get("duration_sec"),
                "fps": fps,
                "source_width": probe.get("width"),
                "source_height": probe.get("height"),
                "max_single_frame_score": float(single.max()) if len(single) else None,
                "max_many_hot_score": float(many.max()) if len(many) else None,
                "transition_segments": segments,
                "transition_seconds": transition_seconds,
                "scenes": scenes,
                "predictions_path": predictions_key,
                "elapsed_sec": round(time.monotonic() - started, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001 - keep batch processing alive and report per-file errors.
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_sec"] = round(time.monotonic() - started, 3)
    return row


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize(rows: list[dict[str, Any]], *, started_at: float, output_dir: Path, videos: list[Path]) -> dict[str, Any]:
    done = [row for row in rows if row.get("status") == "done"]
    failed = [row for row in rows if row.get("status") != "done"]
    transition_rows = [row for row in done if row.get("has_transition")]
    no_transition_rows = [row for row in done if not row.get("has_transition")]
    return {
        "input_count": len(videos),
        "processed_count": len(rows),
        "done_count": len(done),
        "failed_count": len(failed),
        "transition_video_count": len(transition_rows),
        "no_transition_video_count": len(no_transition_rows),
        "elapsed_sec": round(time.monotonic() - started_at, 3),
        "output_dir": str(output_dir),
        "transition_videos": [row["video_path"] for row in transition_rows],
        "failed_videos": [{"video_path": row.get("video_path"), "error": row.get("error")} for row in failed],
    }


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 2
    if not args.input_dir.is_dir():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 2

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    predictions_dir = output_dir / "predictions" if args.save_predictions else None

    videos = list_videos(args.input_dir, recursive=args.recursive, limit=args.limit)
    device_ids = parse_device_ids(args.device_ids)
    if args.resume:
        completed = read_completed(results_path)
        videos = [video for video in videos if str(video) not in completed]
    if not videos:
        print("No videos to process.")
        return 0

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    print(f"Processing {len(videos)} videos with workers={args.workers} provider={args.provider}")
    if device_ids:
        print(f"Using device ids: {device_ids}")
    device_counter = Value("i", 0)
    device_lock = Lock()
    with results_path.open("a" if args.resume else "w", encoding="utf-8") as results_file:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(str(args.model), args.provider, device_ids, device_counter, device_lock),
        ) as executor:
            futures = {
                executor.submit(process_video, str(video), args.threshold, str(predictions_dir) if predictions_dir else None): video
                for video in videos
            }
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                results_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                results_file.flush()
                state = "transition" if row.get("has_transition") else "no-transition"
                if row.get("status") != "done":
                    state = "failed"
                print(f"[{index}/{len(videos)}] {state}: {row.get('video_path')} ({row.get('elapsed_sec')}s)")

    all_rows = rows
    if args.resume and results_path.is_file():
        all_rows = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    summary = summarize(all_rows, started_at=started, output_dir=output_dir, videos=videos)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "transition_videos.txt").write_text(
        "".join(path + "\n" for path in summary["transition_videos"]),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
