from __future__ import annotations

import gc
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import tone_curve  # noqa: E402

SEED = 20260419
SKIP_REASON_CUDA = "NOT EVALUATED: CUDA not available on this runner"
LINEAR_POINTS = torch.tensor(
    [
        [0.0, 0.0],
        [0.14, 0.14],
        [0.29, 0.29],
        [0.43, 0.43],
        [0.57, 0.57],
        [0.71, 0.71],
        [0.86, 0.86],
        [1.0, 1.0],
    ],
    dtype=torch.float32,
)


@dataclass(slots=True)
class BenchCase:
    name: str
    device: str
    resolution: int
    channel: str
    warmup_iters: int
    measure_iters: int


@dataclass(slots=True)
class BenchResult:
    name: str
    device: str
    resolution: int
    channel: str
    status: str
    reason: str
    min_ms: float | None
    median_ms: float | None
    mean_ms: float | None


CASES = [
    BenchCase("cpu-512-rgb_master", "cpu", 512, "rgb_master", 2, 5),
    BenchCase("cpu-1024-rgb_master", "cpu", 1024, "rgb_master", 1, 3),
    BenchCase("cpu-2048-rgb_master", "cpu", 2048, "rgb_master", 0, 1),
    BenchCase("cpu-512-r", "cpu", 512, "r", 2, 5),
    BenchCase("cpu-1024-r", "cpu", 1024, "r", 1, 3),
    BenchCase("cpu-2048-r", "cpu", 2048, "r", 0, 1),
    BenchCase("gpu-512-rgb_master", "cuda", 512, "rgb_master", 2, 5),
    BenchCase("gpu-1024-rgb_master", "cuda", 1024, "rgb_master", 1, 3),
    BenchCase("gpu-2048-rgb_master", "cuda", 2048, "rgb_master", 0, 1),
    BenchCase("gpu-512-r", "cuda", 512, "r", 2, 5),
    BenchCase("gpu-1024-r", "cuda", 1024, "r", 1, 3),
    BenchCase("gpu-2048-r", "cuda", 2048, "r", 0, 1),
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


def _make_image(resolution: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + resolution)
    image = torch.rand((1, 3, resolution, resolution), generator=generator, dtype=torch.float32)
    return image.to(device=device)


def _stats(samples_ms: list[float]) -> tuple[float, float, float]:
    return min(samples_ms), statistics.median(samples_ms), statistics.mean(samples_ms)


def _run_case(case: BenchCase) -> BenchResult:
    if case.device == "cuda" and not torch.cuda.is_available():
        return BenchResult(
            name=case.name,
            device=case.device,
            resolution=case.resolution,
            channel=case.channel,
            status="NOT EVALUATED",
            reason=SKIP_REASON_CUDA,
            min_ms=None,
            median_ms=None,
            mean_ms=None,
        )

    device = torch.device(case.device)
    image = _make_image(case.resolution, device)
    control_points = LINEAR_POINTS.to(device=device)
    measure_once = _latency_runner(device)

    def run_once(
        image_bchw: torch.Tensor = image,
        points: torch.Tensor = control_points,
    ) -> torch.Tensor:
        return tone_curve(image_bchw, control_points=points, channel=case.channel, strength=1.0)

    with torch.inference_mode():
        for _ in range(case.warmup_iters):
            run_once()
        samples_ms = [measure_once(run_once) for _ in range(case.measure_iters)]

    stats = _stats(samples_ms)
    del image, control_points
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return BenchResult(
        name=case.name,
        device=case.device,
        resolution=case.resolution,
        channel=case.channel,
        status="EXECUTED",
        reason="",
        min_ms=stats[0],
        median_ms=stats[1],
        mean_ms=stats[2],
    )


def _hardware_info() -> dict[str, str]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        properties = torch.cuda.get_device_properties(device)
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
        "python_version": sys.version.split()[0],
        "platform": os.name,
        "seed": str(SEED),
    }


def _format_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def render_markdown(hardware: dict[str, str], results: list[BenchResult]) -> str:
    lines = [
        "# Tone Curve Benchmark",
        "",
        "## Hardware",
        "",
        f"- CUDA available: {hardware['cuda_available']}",
        f"- Device name: {hardware['device_name']}",
        f"- Total VRAM (GiB): {hardware['total_vram_gb']}",
        f"- CUDA runtime: {hardware['cuda_runtime']}",
        f"- PyTorch: {hardware['torch_version']}",
        f"- Python: {hardware['python_version']}",
        f"- OS family: {hardware['platform']}",
        f"- Seed: {hardware['seed']}",
        "",
        "## Cases",
        "",
        "| case | device | resolution | channel | status | reason | min_ms | median_ms | mean_ms |",
        "|---|---|---:|---|---|---|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row.name} | {row.device} | {row.resolution} | {row.channel} | "
            f"{row.status} | {row.reason or '—'} | {_format_metric(row.min_ms)} | "
            f"{_format_metric(row.median_ms)} | {_format_metric(row.mean_ms)} |"
        )
    return "\n".join(lines)


def main() -> int:
    torch.manual_seed(SEED)
    logging.getLogger("core.tone_curve").setLevel(logging.ERROR)
    hardware = _hardware_info()
    results = [_run_case(case) for case in CASES]
    print(render_markdown(hardware, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
