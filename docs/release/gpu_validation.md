# Real-GPU release validation

GPU validation is independent from CPU CI. A CPU test run, config import, or
synthetic JSON document never proves CUDA support.

## Required evidence

The manual `Real GPU smoke` workflow must run on a clean, exact commit using a
self-hosted Linux runner with a real CUDA device. The runner must already have a
CUDA-enabled PyTorch and TorchVision installation, the MVTec AD `bottle`
category, and the pretrained model files required by offline execution.

`tools/run_gpu_smoke.py` trains and then runs inference for PatchCore, Reverse
Distillation, and FastFlow. It writes `gpu-evidence.json` using the versioned
[evidence schema](gpu_smoke_evidence.schema.json). Validated evidence records:

- the exact Git commit and a clean-repository assertion;
- the Python, PyTorch, TorchVision, and CUDA runtime versions;
- the real device name, compute capability, memory, and driver version;
- whether the installed MMCV distribution is `mmcv` or `mmcv-lite`, plus its
  exact version;
- the required TorchVision CUDA operator probe and the state of optional MMCV
  custom operators;
- passed training and inference phases for all three key methods, with retained
  log digests;
- measured peak process VRAM for every phase and the overall maximum.

The workflow uploads the JSON and logs whether the smoke succeeds or fails. It
does not upload datasets or generated checkpoints.

Commands in JSON use repository-relative paths or fixed placeholders such as
`<DATASET_ROOT>`, `<WORK_DIR>`, and `<CHECKPOINT>`. Logs are redacted line by
line before hashing. The checker rejects POSIX, macOS, Windows, UNC, and
`file://` absolute paths, along with raw dataset, work-directory, or checkpoint
path fields. Do not publish an artifact that has not passed the checker.

## Blocking states

`status: not_validated` means only that real-GPU validation did not complete. It
is not a warning or partial pass. Missing CUDA, missing data or cached weights,
dependency failures, command failures, absent VRAM observations, a dirty
checkout, a commit mismatch, and missing or modified logs all keep the state
not validated and make the workflow fail.

For the G007 exact-commit go/no-go, when the approved release scope requires
GPU validation, dispatch the workflow for the release candidate commit and
retain its artifact. The required gate is:

```bash
python tools/check_gpu_evidence.py \
  --repo-root . \
  --evidence /path/to/gpu-evidence.json \
  --require-validated
```

Run the checker from the same clean commit and a compatible live CUDA runtime.
`--require-validated` returns non-zero for missing, malformed, synthetic, stale,
or explicitly `not_validated` evidence. The GPU-required G007 gate remains
blocked until this command passes for the exact release candidate.
