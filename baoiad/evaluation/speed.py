"""Speed metric: FPS and GPU memory measurement."""

import time
import torch
import numpy as np


def measure_speed(
    model: torch.nn.Module,
    input_size: tuple = (1, 3, 256, 256),
    device: str = 'cpu',
    n_warmup: int = 10,
    n_runs: int = 50,
) -> dict:
    """Measure inference speed and memory usage.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape.
        device: Device to run on.
        n_warmup: Number of warmup iterations.
        n_runs: Number of timed iterations.

    Returns:
        Dict with fps, latency_ms, gpu_memory_mb.
    """
    model = model.to(device).eval()
    dummy = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)

    # Measure GPU memory (if CUDA)
    gpu_mem = 0.0
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model(dummy)
        gpu_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024  # MB

    # Measure time
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    latency = np.mean(times) * 1000  # ms
    fps = 1000.0 / latency

    return {
        'fps': round(fps, 1),
        'latency_ms': round(latency, 2),
        'gpu_memory_mb': round(gpu_mem, 1),
    }
