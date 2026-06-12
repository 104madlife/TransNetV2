# TransNetV2 ONNX runtime environment

This file records the runtime setup used for local ONNX inference. Keep model
weights, benchmark outputs, and conda environments outside this git repository
when using this project as a `bosGBufferDataPipeline` algorithm submodule.

## Model source

Use the third-party ONNX export from Hugging Face:

```text
https://huggingface.co/elya5/transnetv2
https://huggingface.co/elya5/transnetv2/resolve/main/transnetv2.onnx
```

Recommended local model path:

```text
/data/models/elya5/transnetv2/transnetv2.onnx
```

Download command:

```bash
mkdir -p /data/models/elya5/transnetv2
curl -L --fail --retry 3 \
  -o /data/models/elya5/transnetv2/transnetv2.onnx \
  https://huggingface.co/elya5/transnetv2/resolve/main/transnetv2.onnx
```

Observed model signature:

```text
input:
  input tensor(float) [1, 100, 27, 48, 3]

outputs:
  534 tensor(float) [1, 100, 1]
  535 tensor(float) [1, 100, 1]
```

The batch script treats output `534` as the single-frame transition head and
output `535` as the many-hot transition head. The default `has_transition`
decision uses the single-frame head with threshold `0.5`.

## CPU runtime

Recommended CPU environment path:

```text
/data/bc/.transnetv2_onnx_runtime
```

Create it with Python 3.10:

```bash
conda create -p /data/bc/.transnetv2_onnx_runtime python=3.10 -y
/data/bc/.transnetv2_onnx_runtime/bin/python -m pip install numpy onnxruntime
```

Versions used in the local benchmark:

```text
numpy==2.2.6
onnxruntime==1.23.2
```

Verify CPU provider:

```bash
/data/bc/.transnetv2_onnx_runtime/bin/python - <<'PY'
import onnxruntime as ort

model = "/data/models/elya5/transnetv2/transnetv2.onnx"
sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
print("onnxruntime", ort.__version__)
print("available", ort.get_available_providers())
print("used", sess.get_providers())
for item in sess.get_inputs():
    print("input", item.name, item.type, item.shape)
for item in sess.get_outputs():
    print("output", item.name, item.type, item.shape)
PY
```

Expected provider:

```text
used ['CPUExecutionProvider']
```

Example CPU run:

```bash
/data/bc/.transnetv2_onnx_runtime/bin/python tools/batch_transnetv2_onnx.py \
  --input-dir /data/bc/data \
  --output-dir /data/bc/transnetv2_onnx_benchmark/cpu_w32 \
  --model /data/models/elya5/transnetv2/transnetv2.onnx \
  --workers 32 \
  --provider CPUExecutionProvider
```

## GPU runtime

Recommended GPU environment path:

```text
/data/bc/.transnetv2_onnx_gpu_runtime
```

Create it with Python 3.10:

```bash
conda create -p /data/bc/.transnetv2_onnx_gpu_runtime python=3.10 -y
/data/bc/.transnetv2_onnx_gpu_runtime/bin/python -m pip install numpy
/data/bc/.transnetv2_onnx_gpu_runtime/bin/python -m pip install --index-url https://pypi.org/simple onnxruntime-gpu
/data/bc/.transnetv2_onnx_gpu_runtime/bin/python -m pip install --index-url https://pypi.org/simple 'onnxruntime-gpu[cuda,cudnn]'
```

Versions used in the local benchmark:

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

The local mirror stalled while downloading the large `onnxruntime-gpu` wheel.
Using official PyPI completed the GPU package and CUDA/cuDNN dependency
downloads.

If CUDA provider fails with this error:

```text
libcudnn.so.9: cannot open shared object file: No such file or directory
```

install the CUDA/cuDNN extras:

```bash
/data/bc/.transnetv2_onnx_gpu_runtime/bin/python -m pip install --index-url https://pypi.org/simple 'onnxruntime-gpu[cuda,cudnn]'
```

Verify CUDA provider:

```bash
CUDA_VISIBLE_DEVICES=0 /data/bc/.transnetv2_onnx_gpu_runtime/bin/python - <<'PY'
import numpy as np
import onnxruntime as ort

ort.preload_dlls()
model = "/data/models/elya5/transnetv2/transnetv2.onnx"
sess = ort.InferenceSession(
    model,
    providers=[("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"],
)
print("onnxruntime", ort.__version__)
print("available", ort.get_available_providers())
print("used", sess.get_providers())
x = np.zeros((1, 100, 27, 48, 3), dtype=np.float32)
y = sess.run(None, {sess.get_inputs()[0].name: x})
print([(item.shape, float(item.min()), float(item.max())) for item in y])
PY
```

Expected provider:

```text
used ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Example 6-GPU run:

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

When `CUDA_VISIBLE_DEVICES` is set, `--device-ids` are local CUDA ids inside
the visible set.

## Runtime notes

The ONNX model only handles one 100-frame window. Full-video inference must:

```text
decode to RGB 48x27
pad 25 frames at the beginning
pad the end to align stride 50
run 100-frame windows with stride 50
keep output frames [25:75] from each window
trim predictions to the original frame count
```

The batch script implements this logic. Do not feed independent non-overlapping
100-frame chunks for video-level inference.

CPU and CUDA providers agreed on `has_transition` for the 31-video local test
set, but their raw scores and segment counts differed. Treat CPU and GPU as
separate calibrated inference modes if exact transition frames matter.

## Local artifacts that must stay out of git

Do not commit:

```text
/data/models/elya5/transnetv2/transnetv2.onnx
/data/bc/.transnetv2_onnx_runtime
/data/bc/.transnetv2_onnx_gpu_runtime
/data/bc/transnetv2_onnx_results
/data/bc/transnetv2_onnx_benchmark
```

The repository `.gitignore` also ignores common local model and output
locations such as `models/`, `ckpt/`, `checkpoints/`, `outputs/`, `results/`,
`benchmark/`, `*.onnx`, `*.npz`, `*.pth`, and `*.safetensors`.
