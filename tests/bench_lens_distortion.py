from __future__ import annotations

import gc
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import kornia
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import lens_distortion  # noqa: E402

SEED = 20260419
SKIP_REASON_CUDA = "NOT EVALUATED: CUDA not available on this runner"


@dataclass(slots=True)
class BenchCase:
    name: str
    device: str
    resolution: int
    direction: str
    warmup_iters: int
    measure_iters: int


@dataclass(slots=True)
class BenchResult:
    name: str
    device: str
    resolution: int
    direction: str
    status: str
    reason: str
    min_ms: float | None
    median_ms: float | None
    mean_ms: float | None


CASES = [
    BenchCase("cpu-512-inverse", "cpu", 512, "inverse", 2, 5),
    BenchCase("cpu-1024-inverse", "cpu", 1024, "inverse", 1, 3),
    BenchCase("cpu-2048-inverse", "cpu", 2048, "inverse", 1, 2),
    BenchCase("cpu-512-forward", "cpu", 512, "forward", 1, 3),
    BenchCase("cpu-1024-forward", "cpu", 1024, "forward", 1, 2),
    BenchCase("cpu-2048-forward", "cpu", 2048, "forward", 1, 1),
    BenchCase("gpu-512-inverse", "cuda", 512, "inverse", 2, 5),
    BenchCase("gpu-1024-inverse", "cuda", 1024, "inverse", 1, 3),
    BenchCase("gpu-2048-inverse", "cuda", 2048, "inverse", 1, 2),
    BenchCase("gpu-512-forward", "cuda", 512, "forward", 1, 3),
    BenchCase("gpu-1024-forward", "cuda", 1024, "forward", 1, 2),
    BenchCase("gpu-2048-forward", "cuda", 2048, "forward", 1, 1),
]


def _latency_runner(device: torch.device):
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        def measure_once(fn) -> float:
            torch.cuda.synchronize(device)
            start_event.record()
            fn()
            end_event.record()
            torch.cuda.synchronize(device)
            return start_event.elapsed_time(end_event)

        return measure_once

    def measure_once(fn) -> float:
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        return (end - start) * 1000.0

    return measure_once


def _make_image(size: int, device: torch.device) -> torch.Tensor:
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, 1, 1, size)
    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, 1, size, 1)
    image = torch.cat(
        [
            xx.expand(1, 1, size, size),
            yy.expand(1, 1, size, size),
            ((xx + yy) / 2.0).expand(1, 1, size, size),
        ],
        dim=1,
    )
    return image.to(device=device)


def _stats(samples_ms: list[float]) -> tuple[float, float, float]:
    return min(samples_ms), statistics.median(samples_ms), statistics.mean(samples_ms)


def _run_case(case: BenchCase) -> BenchResult:
    if case.device == "cuda" and not torch.cuda.is_available():
        return BenchResult(
            name=case.name,
            device=case.device,
            resolution=case.resolution,
            direction=case.direction,
            status="NOT EVALUATED",
            reason=SKIP_REASON_CUDA,
            min_ms=None,
            median_ms=None,
            mean_ms=None,
        )

    device = torch.device(case.device)
    image = _make_image(case.resolution, device)
    measure_once = _latency_runner(device)

    def run_once(image_bchw: torch.Tensor = image) -> torch.Tensor:
        return lens_distortion(
            image_bchw,
            direction=case.direction,
            k1=-0.18,
            k2=0.08,
            k3=-0.02,
            p1=0.0,
            p2=0.0,
        )

    with torch.inference_mode():
        for _ in range(case.warmup_iters):
            run_once()
        samples_ms = [measure_once(run_once) for _ in range(case.measure_iters)]

    stats = _stats(samples_ms)
    del image
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return BenchResult(
        name=case.name,
        device=case.device,
        resolution=case.resolution,
        direction=case.direction,
        status="EXECUTED",
        reason="",
        min_ms=stats[0],
        median_ms=stats[1],
        mean_ms=stats[2],
    )


def _hardware_info() -> dict[str, str]:
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.device("cuda"))
        device_name = properties.name
        total_vram_gb = f"{properties.total_memory / (1024**3):.2f}"
    else:
        device_name = "n/a"
        total_vram_gb = "n/a"

    return {
        "cuda_available": str(torch.cuda.is_available()).lower(),
        "device_name": device_name,
        "total_vram_gb": total_vram_gb,
        "cuda_runtime": torch.version.cuda or "n/a",
        "torch_version": torch.__version__,
        "kornia_version": kornia.__version__,
        "python_version": sys.version.split()[0],
        "platform": os.name,
        "seed": str(SEED),
    }


def _format_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def render_markdown(hardware: dict[str, str], results: list[BenchResult]) -> str:
    lines = [
        "# Lens Distortion Benchmark",
        "",
        "## Hardware",
        "",
        f"- CUDA available: {hardware['cuda_available']}",
        f"- Device name: {hardware['device_name']}",
        f"- Total VRAM (GiB): {hardware['total_vram_gb']}",
        f"- CUDA runtime: {hardware['cuda_runtime']}",
        f"- PyTorch: {hardware['torch_version']}",
        f"- Kornia: {hardware['kornia_version']}",
        f"- Python: {hardware['python_version']}",
        f"- OS family: {hardware['platform']}",
        f"- Seed: {hardware['seed']}",
        "",
        "## Cases",
        "",
        (
            "| case | device | resolution | direction | status | reason | min_ms | "
            "median_ms | mean_ms |"
        ),
        "|---|---|---:|---|---|---|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row.name} | {row.device} | {row.resolution} | {row.direction} | {row.status} | "
            f"{row.reason or '—'} | {_format_metric(row.min_ms)} | "
            f"{_format_metric(row.median_ms)} | "
            f"{_format_metric(row.mean_ms)} |"
        )
    return "\n".join(lines)


def main() -> int:
    torch.manual_seed(SEED)
    hardware = _hardware_info()
    results = [_run_case(case) for case in CASES]
    print(render_markdown(hardware, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
