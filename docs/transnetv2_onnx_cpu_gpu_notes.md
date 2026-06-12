# TransNetV2 ONNX local benchmark and BOS integration notes

This note summarizes the local ONNX runtime setup and benchmark results for
running TransNetV2 over `/data/bc/data`. It is intended as handoff material for
future agents that may integrate this repository into
`/data/bc/bosGBufferDataPipeline` as an algorithm handler submodule.

## Scope

Current objective already completed:

- Use the Hugging Face ONNX export at `elya5/transnetv2`.
- Run batch transition detection over `/data/bc/data`.
- Compare CPU ONNX Runtime vs GPU ONNX Runtime speed.
- Keep all runtime environments, model files, and benchmark outputs outside this
  git repository, except for the reusable batch script.

This note does not implement BOS pipeline integration. It records constraints
and recommended integration shape for later work.

## Repository changes

Reusable local batch script:

```text
tools/batch_transnetv2_onnx.py
```

The script:

- enumerates video files under an input directory,
- decodes video frames with `ffmpeg` into RGB `48x27`,
- applies the original TransNetV2 video-window logic,
- runs ONNX Runtime inference,
- writes `results.jsonl`, `summary.json`, and `transition_videos.txt`,
- supports CPU and CUDA execution providers,
- supports multi-GPU worker binding via `--device-ids`.

Important behavior:

- It does not modify source videos.
- It does not write sidecar files next to source videos.
- It keeps per-video failures in `results.jsonl` so a large batch can continue.

## Model

Model file used:

```text
/data/models/elya5/transnetv2/transnetv2.onnx
```

Downloaded from:

```text
https://huggingface.co/elya5/transnetv2/resolve/main/transnetv2.onnx
```

Size observed locally:

```text
31250929 bytes
about 30 MiB
```

Expected ONNX signature:

```text
input:
  input tensor(float) [1, 100, 27, 48, 3]

outputs:
  534 tensor(float) [1, 100, 1]
  535 tensor(float) [1, 100, 1]
```

The script assumes:

- input is `float32`,
- input layout is `[batch, frames, height, width, channels]`,
- frame size is `27x48 RGB`,
- output 0 is the single-frame transition head,
- output 1 is the many-hot transition head.

The current decision rule uses the single-frame head and threshold `0.5`.

## Video-window logic

The ONNX model consumes one 100-frame window. A full video must still be
preprocessed the same way as the original TensorFlow inference code:

- resize/decode all frames to RGB `48x27`,
- pad the beginning with 25 copies of the first frame,
- pad the end so windowing aligns to stride 50,
- run windows of 100 frames,
- advance by 50 frames per window,
- keep only output frames `[25:75]` from each 100-frame window,
- concatenate kept predictions and trim back to the original frame count.

This is important. Do not feed arbitrary non-overlapping 100-frame chunks and
expect video-level predictions to match the original algorithm semantics.

## CPU environment

CPU environment path:

```text
/data/bc/.transnetv2_onnx_runtime
```

Python:

```text
python 3.10
```

Installed packages:

```text
numpy==2.2.6
onnxruntime==1.23.2
```

Observed environment size:

```text
377M /data/bc/.transnetv2_onnx_runtime
```

Recreate command:

```bash
conda create -p /data/bc/.transnetv2_onnx_runtime python=3.10 -y
/data/bc/.transnetv2_onnx_runtime/bin/python -m pip install numpy onnxruntime
```

CPU provider check:

```bash
/data/bc/.transnetv2_onnx_runtime/bin/python - <<'PY'
import onnxruntime as ort
sess = ort.InferenceSession(
    "/data/models/elya5/transnetv2/transnetv2.onnx",
    providers=["CPUExecutionProvider"],
)
print("onnxruntime", ort.__version__)
print("available", ort.get_available_providers())
print("used", sess.get_providers())
for i in sess.get_inputs():
    print("input", i.name, i.type, i.shape)
for o in sess.get_outputs():
    print("output", o.name, o.type, o.shape)
PY
```

Expected provider result:

```text
used ['CPUExecutionProvider']
```

## GPU environment

GPU environment path:

```text
/data/bc/.transnetv2_onnx_gpu_runtime
```

Python:

```text
python 3.10
```

Installed packages include:

```text
numpy==2.2.6
onnxruntime-gpu==1.23.2
nvidia-cudnn-cu12==9.23.1.3
nvidia-cublas-cu12==12.9.2.10
nvidia-cuda-runtime-cu12==12.9.79
nvidia-cuda-nvrtc-cu12==12.9.86
nvidia-cufft-cu12==11.4.1.4
nvidia-curand-cu12==10.3.10.19
nvidia-nvjitlink-cu12==12.9.86
```

Observed environment size:

```text
3.4G /data/bc/.transnetv2_onnx_gpu_runtime
```

Recreate command:

```bash
conda create -p /data/bc/.transnetv2_onnx_gpu_runtime python=3.10 -y
/data/bc/.transnetv2_onnx_gpu_runtime/bin/python -m pip install numpy
/data/bc/.transnetv2_onnx_gpu_runtime/bin/python -m pip install --index-url https://pypi.org/simple onnxruntime-gpu
/data/bc/.transnetv2_onnx_gpu_runtime/bin/python -m pip install --index-url https://pypi.org/simple 'onnxruntime-gpu[cuda,cudnn]'
```

Reason for using official PyPI for GPU packages:

- The CPU packages installed fine from the configured mirror.
- The `onnxruntime-gpu` 300 MiB wheel stalled repeatedly through the mirror/proxy path.
- Official PyPI completed the GPU wheel and CUDA/cuDNN dependency downloads.

The first CUDA provider attempt failed before CUDA/cuDNN wheels were installed:

```text
Failed to load library libonnxruntime_providers_cuda.so with error:
libcudnn.so.9: cannot open shared object file: No such file or directory
```

After installing `onnxruntime-gpu[cuda,cudnn]`, CUDA provider initialized
successfully.

GPU provider smoke check:

```bash
CUDA_VISIBLE_DEVICES=0 /data/bc/.transnetv2_onnx_gpu_runtime/bin/python - <<'PY'
import numpy as np
import onnxruntime as ort

ort.preload_dlls()
sess = ort.InferenceSession(
    "/data/models/elya5/transnetv2/transnetv2.onnx",
    providers=[("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"],
)
print("onnxruntime", ort.__version__)
print("available", ort.get_available_providers())
print("used", sess.get_providers())
x = np.zeros((1, 100, 27, 48, 3), dtype=np.float32)
y = sess.run(None, {sess.get_inputs()[0].name: x})
print([(a.shape, float(a.min()), float(a.max())) for a in y])
PY
```

Expected provider result:

```text
used ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Important script detail:

```python
ort.preload_dlls()
```

This is needed so ONNX Runtime can find CUDA/cuDNN dynamic libraries installed
from pip wheels inside the environment.

## Hardware observed during benchmark

Host had:

```text
8 x NVIDIA GeForce RTX 4090
driver 595.71.05
about 23028 MiB per GPU
84 logical CPUs
about 944 GiB RAM
```

Before the GPU benchmark, all 8 GPUs were effectively idle:

```text
GPU 0-7: utilization 0%, memory used about 1 MiB
```

The 6-GPU run used devices:

```text
0,1,2,3,4,5
```

During the GPU run, each selected GPU used about:

```text
680-690 MiB
```

After the run, GPU memory returned to about:

```text
1 MiB per GPU
```

## Dataset used for benchmark

Input directory:

```text
/data/bc/data
```

Observed dataset:

```text
31 videos
4.7G total
mostly 3200x1800, 30 fps, about 60-65 seconds per video
one 100-second video
```

Video extensions handled by the script:

```text
.mp4, .mov, .mkv, .avi, .webm
```

## Baseline CPU full run

Earlier CPU run with 4 workers:

```bash
/data/bc/.transnetv2_onnx_runtime/bin/python tools/batch_transnetv2_onnx.py \
  --input-dir /data/bc/data \
  --output-dir /data/bc/transnetv2_onnx_results/full \
  --model /data/models/elya5/transnetv2/transnetv2.onnx \
  --workers 4
```

Result:

```text
processed_count: 31
done_count: 31
failed_count: 0
transition_video_count: 3
no_transition_video_count: 28
elapsed_sec: 469.295
```

This run established basic correctness and output format, but it is not the
recommended speed setting.

## CPU workers=32 benchmark

Command:

```bash
/usr/bin/time -v /data/bc/.transnetv2_onnx_runtime/bin/python tools/batch_transnetv2_onnx.py \
  --input-dir /data/bc/data \
  --output-dir /data/bc/transnetv2_onnx_benchmark/cpu_w32 \
  --model /data/models/elya5/transnetv2/transnetv2.onnx \
  --workers 32 \
  --provider CPUExecutionProvider
```

Summary:

```text
processed_count: 31
done_count: 31
failed_count: 0
transition_video_count: 3
no_transition_video_count: 28
script elapsed_sec: 114.6
/usr/bin/time wall time: 1:54.75
```

Resource data from `/usr/bin/time -v`:

```text
User time: 3957.40s
System time: 63.15s
CPU utilization: 3503%
Max RSS: 384684 KB
```

Throughput:

```text
31 / 114.75s = 0.270 videos/sec
about 16.2 videos/min
```

Per-video elapsed under this high concurrency:

```text
average: 83.588s
min: 76.306s
max: 114.375s
```

Observation:

- `workers=32` launched nearly all videos concurrently.
- `ffmpeg` decode dominated a large part of CPU usage.
- System load reached about 60 during the run.
- This was much faster than 4 workers, but already stresses decode and process scheduling.

## GPU 6-card benchmark

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 /usr/bin/time -v \
  /data/bc/.transnetv2_onnx_gpu_runtime/bin/python tools/batch_transnetv2_onnx.py \
  --input-dir /data/bc/data \
  --output-dir /data/bc/transnetv2_onnx_benchmark/gpu_6x \
  --model /data/models/elya5/transnetv2/transnetv2.onnx \
  --workers 6 \
  --provider CUDAExecutionProvider \
  --device-ids 0,1,2,3,4,5
```

Summary:

```text
processed_count: 31
done_count: 31
failed_count: 0
transition_video_count: 3
no_transition_video_count: 28
script elapsed_sec: 53.996
/usr/bin/time wall time: 0:54.15
```

Resource data from `/usr/bin/time -v`:

```text
User time: 1742.82s
System time: 26.15s
CPU utilization: 3266%
Max RSS: 804876 KB
```

Throughput:

```text
31 / 54.15s = 0.573 videos/sec
about 34.4 videos/min
```

Per-video elapsed:

```text
average: 8.977s
min: 7.788s
max: 13.797s
```

Speedup over CPU workers=32:

```text
114.75 / 54.15 = about 2.12x
```

Observation:

- GPU single-video processing is much faster than CPU single-video processing.
- End-to-end speedup is smaller than per-video speedup because `ffmpeg` decode
  and input preprocessing remain CPU-bound.
- Simply adding more GPUs is unlikely to scale linearly unless decode and
  scheduling are also tuned.

## Detected transition videos

Both CPU and GPU runs found the same set of videos that contain transitions:

```text
/data/bc/data/2026-05-29 00-07-49.mp4
/data/bc/data/2026-06-01 01-39-19.mp4
/data/bc/data/2026-06-01 03-55-55.mp4
```

CPU transition details:

```text
2026-05-29 00-07-49.mp4
  transition_count: 1
  transition_seconds: [44.9]
  max_single_frame_score: 0.647456

2026-06-01 01-39-19.mp4
  transition_count: 1
  transition_seconds: [46.733]
  max_single_frame_score: 0.680870

2026-06-01 03-55-55.mp4
  transition_count: 2
  transition_seconds: [4.4, 10.233]
  max_single_frame_score: 0.893877
```

GPU transition details:

```text
2026-05-29 00-07-49.mp4
  transition_count: 1
  transition_seconds: [44.9]
  max_single_frame_score: 0.757046
  onnx_device_id: 4

2026-06-01 01-39-19.mp4
  transition_count: 2
  transition_seconds: [43.733, 46.733]
  max_single_frame_score: 0.786544
  onnx_device_id: 0

2026-06-01 03-55-55.mp4
  transition_count: 3
  transition_seconds: [2.867, 4.4, 10.233]
  max_single_frame_score: 0.920492
  onnx_device_id: 2
```

## Correctness caveat: CPU vs CUDA provider

The CPU and GPU runs agreed on video-level `has_transition` for all 31 videos.

However, CPU and CUDA providers did not produce numerically identical scores for
the same ONNX model and same inputs. Disabling TF32 did not remove the
difference. As a result:

- video-level transition classification matched on this dataset,
- transition segment counts differed for two positive videos,
- GPU output should not be treated as production-equivalent to CPU output until
  a calibration/acceptance pass is done.

For production use, decide which provider is authoritative. If exact transition
frames or scene counts matter, prefer one of:

- use CPU provider as the reference path,
- calibrate GPU thresholds against CPU results,
- add a post-processing merge rule for nearby positive frames,
- validate both providers on a larger labeled sample.

For a first BOS filter stage where only `has_transition` matters, GPU results
may be acceptable after a small validation set confirms video-level agreement.

## CPU worker selection guidance

The script uses process-level parallelism over videos. Inside each worker,
ONNX Runtime is configured with:

```python
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
```

Keep this setting when using many workers. Otherwise, total runnable threads can
explode as:

```text
worker_count * ORT_threads * ffmpeg_threads
```

Recommended starting points:

```text
Logical CPUs <= 32:   workers = 8, 16, 24
Logical CPUs 64-96:   workers = 16, 24, 32, 48
Logical CPUs 128:     workers = 32, 48, 64, optionally 80
```

Do not blindly set `workers` equal to logical CPU count. `ffmpeg` decode is a
major part of the cost, and each ffmpeg process may itself use multiple threads.

For a 128-core CPU, expected speed relative to the current 84-logical-CPU host
with `workers=32`:

```text
Current observed CPU workers=32: 114.6s for 31 videos
Conservative estimate: 95-100s
Middle estimate:       80-90s
Optimistic estimate:   65-75s
```

Practical recommendation on a 128-core host:

```text
First try workers=48
Then try workers=64
Try workers=80 only if CPU is not saturated and I/O is healthy
```

If future tuning is needed, add a script option for ffmpeg threads, for example:

```text
--ffmpeg-threads 1
--ffmpeg-threads 2
```

Then compare:

```text
workers=48, ffmpeg_threads=auto
workers=64, ffmpeg_threads=auto
workers=48, ffmpeg_threads=1
workers=64, ffmpeg_threads=1
workers=96, ffmpeg_threads=1
```

This separates ONNX CPU inference scaling from ffmpeg decode scaling.

## GPU worker selection guidance

For CUDA provider, bind workers to explicit device ids:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  ... --workers 6 --provider CUDAExecutionProvider --device-ids 0,1,2,3,4,5
```

When `CUDA_VISIBLE_DEVICES` is set, `--device-ids` are local CUDA ids within the
visible set. In the example above, local ids `0-5` map to the six visible cards.

The observed 6-GPU run used only about 680-690 MiB per GPU. If throughput is
important, consider testing:

```text
6 GPUs, workers=6
6 GPUs, workers=12
8 GPUs, workers=8
8 GPUs, workers=16
```

The likely next bottleneck is ffmpeg decode, not GPU memory.

## Output locations

Benchmark outputs are outside this repo:

```text
/data/bc/transnetv2_onnx_results/full
/data/bc/transnetv2_onnx_benchmark/cpu_w32
/data/bc/transnetv2_onnx_benchmark/gpu_6x
```

Each output directory contains:

```text
results.jsonl
summary.json
transition_videos.txt
```

These outputs are small and can be regenerated.

## BOS pipeline integration recommendation

Recommended shape:

```text
/data/bc/bosGBufferDataPipeline/
  algorithms/
    TransNetV2/                  # this repository as a submodule
      tools/
      pipeline_adapters/
        run_transition_detect_onnx.py
```

Main BOS pipeline should call the adapter through the subprocess runtime
protocol. Do not directly import model code into the main BOS framework.

Suggested stage role:

```text
raw video -> transnet_transition_v1/filter_v1 -> slice_v1 -> caption_v2
```

Suggested runtime output for a video with transition:

```json
{
  "status": "done",
  "artifacts": {
    "has_transition": true,
    "transition_count": 1,
    "scene_count": 2,
    "report_key": "..."
  },
  "stop_pipeline": false
}
```

Suggested runtime output for a video without transition:

```json
{
  "status": "done",
  "artifacts": {
    "has_transition": false,
    "transition_count": 0,
    "scene_count": 1,
    "report_key": "..."
  },
  "stop_pipeline": true,
  "reason": "no_transition"
}
```

Using `done + stop_pipeline=true` treats no-transition videos as successfully
classified and intentionally stops downstream work. Use `rejected` only if the
pipeline wants no-transition videos to appear in rejected state.

Adapter outputs should be compact by default:

```text
transition_report.json
scenes.json
_SUCCESS
```

Avoid uploading full per-frame prediction arrays unless a debug parameter is
explicitly enabled.

## Risks for future agents

Key risks:

- Third-party ONNX export from Hugging Face is not the original repository's
  official runtime artifact.
- CPU provider and CUDA provider have measurable numerical differences.
- GPU video-level classification matched CPU on the current 31-video dataset,
  but segment-level counts differed.
- Full video processing still depends heavily on ffmpeg decode throughput.
- GPU environment is much larger than CPU environment because CUDA/cuDNN wheels
  are installed inside the conda prefix.
- For BOS, do not mix BOS task routing/storage logic into this algorithm
  repository. Keep the adapter as a subprocess boundary.

Recommended validation before production BOS use:

```text
1. Re-run CPU provider on a representative sample.
2. Re-run GPU provider if GPU will be used in production.
3. Compare video-level has_transition.
4. Compare transition_count and transition_seconds only if downstream needs them.
5. Decide provider-specific threshold or merge policy.
6. Run one BOS smoke task and verify runtime_output.json artifacts and uploaded report files.
```

## Quick commands

CPU full batch:

```bash
/data/bc/.transnetv2_onnx_runtime/bin/python tools/batch_transnetv2_onnx.py \
  --input-dir /data/bc/data \
  --output-dir /data/bc/transnetv2_onnx_results/full \
  --model /data/models/elya5/transnetv2/transnetv2.onnx \
  --workers 32 \
  --provider CPUExecutionProvider
```

GPU full batch on six visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
/data/bc/.transnetv2_onnx_gpu_runtime/bin/python tools/batch_transnetv2_onnx.py \
  --input-dir /data/bc/data \
  --output-dir /data/bc/transnetv2_onnx_benchmark/gpu_6x \
  --model /data/models/elya5/transnetv2/transnetv2.onnx \
  --workers 6 \
  --provider CUDAExecutionProvider \
  --device-ids 0,1,2,3,4,5
```

Inspect transition list:

```bash
cat /data/bc/transnetv2_onnx_benchmark/gpu_6x/transition_videos.txt
```
