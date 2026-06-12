# Pipeline adapters

This directory contains adapters used by the BOS GBuffer data pipeline. The main
pipeline should call these adapters through its subprocess runtime protocol. It
should not import TransNetV2 internals directly.

## `run_transition_filter.py`

`run_transition_filter.py` runs TransNetV2 ONNX inference for one Worker-resolved
video and returns a pipeline status:

```text
no internal transition -> status=done
internal transition    -> status=rejected, reason=transition_detected
runtime failure        -> status=failed
```

The adapter expects the main Worker to provide `runtime_input.json` with a
downloaded local raw video in `stage_input.local_paths[0]`.

Example stage parameters:

```json
{
  "backend": "onnx",
  "model_path": "/data/models/elya5/transnetv2/transnetv2.onnx",
  "provider": "CPUExecutionProvider",
  "strictness": "normal",
  "score_threshold": 0.5,
  "min_transition_count": 1,
  "min_segment_frames": 1,
  "boundary_margin_frames": 5,
  "save_predictions": false,
  "save_scenes": true,
  "output_prefix": "test/transnet_filter/output/transnetv2_filter_v1/"
}
```

For CUDA ONNX Runtime:

```json
{
  "provider": "CUDAExecutionProvider",
  "device_id": 0
}
```

When `CUDA_VISIBLE_DEVICES` is set by the parent stage runtime, `device_id` is the
local CUDA id inside that visible set.

Output files are returned through `runtime_output.json` so the parent Worker can
upload them to BOS before finalizing the task:

```text
result.json
scenes.txt
predictions.txt  # only when save_predictions=true
```

The adapter treats `output_prefix` as a base directory. If it does not contain
`{task_id}` or `{safe_task_id}`, the adapter appends a task-specific directory to
avoid collisions between per-video filter tasks.
