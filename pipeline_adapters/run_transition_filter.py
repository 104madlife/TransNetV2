#!/usr/bin/env python3
"""BOS GBuffer pipeline adapter for TransNetV2 ONNX transition filtering."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = "/data/models/elya5/transnetv2/transnetv2.onnx"
DEFAULT_OUTPUT_PREFIX = "output/transnetv2_filter_v1/"
STRICTNESS_PRESETS: dict[str, dict[str, float | int]] = {
    "strict": {
        "score_threshold": 0.35,
        "min_transition_count": 1,
        "min_segment_frames": 1,
        "boundary_margin_frames": 0,
    },
    "normal": {
        "score_threshold": 0.5,
        "min_transition_count": 1,
        "min_segment_frames": 1,
        "boundary_margin_frames": 5,
    },
    "loose": {
        "score_threshold": 0.65,
        "min_transition_count": 1,
        "min_segment_frames": 1,
        "boundary_margin_frames": 10,
    },
}


class RuntimeFailure(Exception):
    """Failure returned to the main pipeline as status=failed."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TransNetV2 ONNX as a BOS pipeline filter.")
    parser.add_argument("--input", required=True, help="Path to runtime_input.json.")
    parser.add_argument("--output", required=True, help="Path to runtime_output.json.")
    return parser


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_runtime_output(
    path: Path,
    *,
    status: str,
    artifacts: dict[str, Any] | None = None,
    output_files: list[dict[str, str]] | None = None,
    reason: str | None = None,
    stop_pipeline: bool = False,
) -> None:
    write_json(
        path,
        {
            "status": status,
            "artifacts": artifacts or {},
            "output_files": output_files or [],
            "next_stage_override": None,
            "stop_pipeline": stop_pipeline,
            "reason": reason,
        },
    )


def ensure_trailing_slash(prefix: str) -> str:
    value = prefix.strip().lstrip("/")
    if not value:
        return ""
    return value if value.endswith("/") else f"{value}/"


def safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._")
    return safe or "item"


def output_prefix(params: dict[str, Any], task_id: str) -> str:
    raw_prefix = str(params.get("output_prefix") or DEFAULT_OUTPUT_PREFIX)
    formatted = raw_prefix.format(task_id=task_id, safe_task_id=safe_name(task_id))
    prefix = ensure_trailing_slash(formatted)
    if "{task_id}" in raw_prefix or "{safe_task_id}" in raw_prefix:
        return prefix
    return ensure_trailing_slash(prefix + safe_name(task_id))


def input_video_path(runtime_input: dict[str, Any]) -> Path:
    stage_input = runtime_input.get("stage_input") or {}
    local_paths = stage_input.get("local_paths") or []
    if not local_paths:
        raise RuntimeFailure("runtime_input.stage_input.local_paths is empty.")
    path = Path(str(local_paths[0]))
    if not path.is_file():
        raise RuntimeFailure(f"Input video does not exist: {path}")
    return path


def source_raw_key(runtime_input: dict[str, Any]) -> str | None:
    stage_input = runtime_input.get("stage_input") or {}
    key = stage_input.get("key")
    if key:
        return str(key)
    task = runtime_input.get("task") or {}
    source = task.get("source") or {}
    raw_key = source.get("raw_key")
    return str(raw_key) if raw_key else None


def bool_param(params: dict[str, Any], name: str, default: bool) -> bool:
    value = params.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def int_param(params: dict[str, Any], name: str, default: int) -> int:
    value = params.get(name, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeFailure(f"Parameter {name} must be an integer: {value!r}") from exc
    return parsed


def float_param(params: dict[str, Any], name: str, default: float) -> float:
    value = params.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeFailure(f"Parameter {name} must be a number: {value!r}") from exc


def decision_params(params: dict[str, Any]) -> dict[str, float | int]:
    strictness = str(params.get("strictness") or "normal").strip().lower()
    preset = dict(STRICTNESS_PRESETS.get(strictness, STRICTNESS_PRESETS["normal"]))
    if "score_threshold" in params:
        preset["score_threshold"] = float_param(params, "score_threshold", float(preset["score_threshold"]))
    if "min_transition_count" in params:
        preset["min_transition_count"] = int_param(params, "min_transition_count", int(preset["min_transition_count"]))
    if "min_segment_frames" in params:
        preset["min_segment_frames"] = int_param(params, "min_segment_frames", int(preset["min_segment_frames"]))
    if "boundary_margin_frames" in params:
        preset["boundary_margin_frames"] = int_param(params, "boundary_margin_frames", int(preset["boundary_margin_frames"]))

    if float(preset["score_threshold"]) < 0 or float(preset["score_threshold"]) > 1:
        raise RuntimeFailure("score_threshold must be between 0 and 1.")
    if int(preset["min_transition_count"]) <= 0:
        raise RuntimeFailure("min_transition_count must be positive.")
    if int(preset["min_segment_frames"]) <= 0:
        raise RuntimeFailure("min_segment_frames must be positive.")
    if int(preset["boundary_margin_frames"]) < 0:
        raise RuntimeFailure("boundary_margin_frames must be non-negative.")
    return preset


def contiguous_segments(scores: Any, *, threshold: float, min_segment_frames: int) -> list[list[int]]:
    positive = scores >= threshold
    segments: list[list[int]] = []
    start: int | None = None
    for index, is_positive in enumerate(positive):
        if bool(is_positive) and start is None:
            start = index
        elif not bool(is_positive) and start is not None:
            end = index - 1
            if end - start + 1 >= min_segment_frames:
                segments.append([start, end])
            start = None
    if start is not None:
        end = len(positive) - 1
        if end - start + 1 >= min_segment_frames:
            segments.append([start, end])
    return segments


def internal_segments(segments: list[list[int]], *, frame_count: int, boundary_margin_frames: int) -> list[list[int]]:
    if frame_count <= 0:
        return []
    margin = min(boundary_margin_frames, max(0, (frame_count - 1) // 2))
    internal_start = margin
    internal_end = frame_count - 1 - margin
    return [segment for segment in segments if segment[1] >= internal_start and segment[0] <= internal_end]


def transition_seconds(segments: list[list[int]], fps: Any) -> list[float] | None:
    if not fps:
        return None
    try:
        fps_value = float(fps)
    except (TypeError, ValueError):
        return None
    if fps_value <= 0:
        return None
    return [round(segment[0] / fps_value, 3) for segment in segments]


def init_onnx_session(model_path: Path, *, provider: str, device_id: int | None) -> tuple[Any, str, list[str]]:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeFailure("onnxruntime is not installed in the runtime python environment.") from exc

    if not model_path.is_file():
        raise RuntimeFailure(f"TransNetV2 ONNX model not found: {model_path}")

    if provider == "CUDAExecutionProvider" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    available = ort.get_available_providers()
    if provider == "CUDAExecutionProvider" and provider in available:
        provider_config: list[Any]
        if device_id is None:
            provider_config = [("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"]
        else:
            provider_config = [("CUDAExecutionProvider", {"device_id": device_id, "use_tf32": 0}), "CPUExecutionProvider"]
    elif provider in available:
        provider_config = [provider]
    else:
        raise RuntimeFailure(f"ONNX Runtime provider is unavailable: {provider}; available={available}")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=provider_config)
    return session, session.get_inputs()[0].name, session.get_providers()


def run_inference(video_path: Path, params: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        import numpy as np
        from tools import batch_transnetv2_onnx as batch
    except ModuleNotFoundError as exc:
        raise RuntimeFailure(f"Missing TransNetV2 ONNX runtime dependency: {exc}") from exc

    model_path = Path(str(params.get("model_path") or DEFAULT_MODEL_PATH))
    if not model_path.is_file():
        fallback_model = Path("/data/bc/models/elya5/transnetv2/transnetv2.onnx")
        if fallback_model.is_file():
            model_path = fallback_model
    provider = str(params.get("provider") or "CPUExecutionProvider")
    device_id = params.get("device_id")
    device_id_int = int(device_id) if device_id is not None and str(device_id).strip() != "" else None

    started = time.monotonic()
    session, input_name, used_providers = init_onnx_session(model_path, provider=provider, device_id=device_id_int)
    batch.SESSION = session
    batch.INPUT_NAME = input_name
    batch.ASSIGNED_DEVICE_ID = device_id_int

    probe = batch.probe_video(video_path)
    frames = batch.decode_video(video_path)
    single, many = batch.predict_frames(frames)
    decision = decision_params(params)
    threshold = float(decision["score_threshold"])
    min_segment_frames = int(decision["min_segment_frames"])
    boundary_margin_frames = int(decision["boundary_margin_frames"])
    min_transition_count = int(decision["min_transition_count"])
    all_segments = contiguous_segments(single, threshold=threshold, min_segment_frames=min_segment_frames)
    kept_segments = internal_segments(
        all_segments,
        frame_count=len(frames),
        boundary_margin_frames=boundary_margin_frames,
    )
    scenes = batch.predictions_to_scenes(single, threshold)
    fps = probe.get("fps") if isinstance(probe, dict) else None

    return {
        "status": "done",
        "has_transition": len(kept_segments) >= min_transition_count,
        "transition_count": len(kept_segments),
        "all_transition_count": len(all_segments),
        "transition_segments": kept_segments,
        "all_transition_segments": all_segments,
        "transition_seconds": transition_seconds(kept_segments, fps),
        "scene_count": len(scenes),
        "scenes": scenes,
        "frame_count": int(len(frames)),
        "duration_sec": probe.get("duration_sec") if isinstance(probe, dict) else None,
        "fps": fps,
        "source_width": probe.get("width") if isinstance(probe, dict) else None,
        "source_height": probe.get("height") if isinstance(probe, dict) else None,
        "max_single_frame_score": float(single.max()) if len(single) else None,
        "max_many_hot_score": float(many.max()) if len(many) else None,
        "threshold": threshold,
        "min_segment_frames": min_segment_frames,
        "min_transition_count": min_transition_count,
        "boundary_margin_frames": boundary_margin_frames,
        "model_path": str(model_path),
        "provider": provider,
        "used_providers": used_providers,
        "device_id": device_id_int,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "elapsed_sec": round(time.monotonic() - started, 3),
        "_single_predictions": single,
        "_many_predictions": many,
        "_np": np,
    }


def write_prediction_files(
    *,
    inference: dict[str, Any],
    output_dir: Path,
    save_predictions: bool,
    save_scenes: bool,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    np = inference.pop("_np")
    single = inference.pop("_single_predictions")
    many = inference.pop("_many_predictions")

    if save_predictions:
        predictions_path = output_dir / "predictions.txt"
        predictions = np.stack([single, many], axis=1)
        np.savetxt(predictions_path, predictions, fmt="%.6f")
        paths["predictions"] = predictions_path

    if save_scenes:
        scenes_path = output_dir / "scenes.txt"
        scenes = inference.get("scenes") or []
        scenes_path.write_text("".join(f"{start} {end}\n" for start, end in scenes), encoding="utf-8")
        paths["scenes"] = scenes_path

    return paths


def build_output_files(paths: dict[str, Path], prefix: str) -> list[dict[str, str]]:
    output_files = []
    for name, path in paths.items():
        output_files.append({"local_path": str(path), "output_key": f"{prefix}{path.name}"})
    return output_files


def main() -> int:
    args = build_parser().parse_args()
    runtime_input_path = Path(args.input)
    runtime_output_path = Path(args.output)
    runtime_input = json.loads(runtime_input_path.read_text(encoding="utf-8"))
    params = runtime_input.get("parameters") or {}
    if not isinstance(params, dict):
        params = {}

    task_id = str(runtime_input.get("task_id") or "unknown_task")
    output_dir = Path(str(runtime_input["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix(params, task_id)
    result_path = output_dir / "result.json"

    try:
        backend = str(params.get("backend") or "onnx").strip().lower()
        if backend != "onnx":
            raise RuntimeFailure(f"Unsupported backend for this adapter: {backend}")

        video_path = input_video_path(runtime_input)
        inference = run_inference(video_path, params)
        inference.update(
            {
                "schema_version": "transnetv2_filter_result.v1",
                "task_id": task_id,
                "source_key": source_raw_key(runtime_input),
                "local_video_path": str(video_path),
                "strictness": str(params.get("strictness") or "normal"),
            }
        )

        extra_paths = write_prediction_files(
            inference=inference,
            output_dir=output_dir,
            save_predictions=bool_param(params, "save_predictions", False),
            save_scenes=bool_param(params, "save_scenes", True),
        )
        write_json(result_path, inference)

        paths = {"result": result_path, **extra_paths}
        output_files = build_output_files(paths, prefix)
        artifacts = {
            "has_transition": bool(inference["has_transition"]),
            "transition_count": int(inference["transition_count"]),
            "all_transition_count": int(inference["all_transition_count"]),
            "max_single_frame_score": inference.get("max_single_frame_score"),
            "max_many_hot_score": inference.get("max_many_hot_score"),
            "threshold": inference.get("threshold"),
            "boundary_margin_frames": inference.get("boundary_margin_frames"),
            "frame_count": inference.get("frame_count"),
            "duration_sec": inference.get("duration_sec"),
            "result_key": f"{prefix}result.json",
        }
        if "predictions" in paths:
            artifacts["predictions_key"] = f"{prefix}{paths['predictions'].name}"
        if "scenes" in paths:
            artifacts["scenes_key"] = f"{prefix}{paths['scenes'].name}"

        if inference["has_transition"]:
            write_runtime_output(
                runtime_output_path,
                status="rejected",
                artifacts=artifacts,
                output_files=output_files,
                reason="transition_detected",
                stop_pipeline=True,
            )
            return 0

        write_runtime_output(
            runtime_output_path,
            status="done",
            artifacts=artifacts,
            output_files=output_files,
            stop_pipeline=False,
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - return structured failure to the parent worker.
        error = f"{type(exc).__name__}: {exc}"
        error_result = {
            "schema_version": "transnetv2_filter_result.v1",
            "task_id": task_id,
            "status": "failed",
            "error": error,
        }
        write_json(result_path, error_result)
        write_runtime_output(
            runtime_output_path,
            status="failed",
            artifacts={"result_key": f"{prefix}result.json"},
            output_files=build_output_files({"result": result_path}, prefix),
            reason=error,
            stop_pipeline=True,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
