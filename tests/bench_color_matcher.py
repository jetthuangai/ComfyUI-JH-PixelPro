from __future__ import annotations

import gc
import logging
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

from core import color_matcher  # noqa: E402

SEED = 20260419
SKIP_REASON_CUDA = "NOT EVALUATED: CUDA not available on this runner"


@dataclass(slots=True)
class BenchCase:
    name: str
    device: str
    resolution: int
    channels: str
    warmup_iters: int
    measure_iters: int


@dataclass(slots=True)
class BenchResult:
    name: str
    device: str
    resolution: int
    channels: str
    status: str
    reason: str
    min_ms: float | None
    median_ms: float | None
    mean_ms: float | None


CASES = [
    BenchCase("cpu-512-ab", "cpu", 512, "ab", 2, 5),
    BenchCase("cpu-1024-ab", "cpu", 1024, "ab", 1, 3),
    BenchCase("cpu-2048-ab", "cpu", 2048, "ab", 0, 1),
    BenchCase("cpu-512-lab", "cpu", 512, "lab", 2, 5),
    BenchCase("cpu-1024-lab", "cpu", 1024, "lab", 1, 3),
    BenchCase("cpu-2048-lab", "cpu", 2048, "lab", 0, 1),
    BenchCase("gpu-512-ab", "cuda", 512, "ab", 2, 5),
    BenchCase("gpu-1024-ab", "cuda", 1024, "ab", 1, 3),
    BenchCase("gpu-2048-ab", "cuda", 2048, "ab", 0, 1),
    BenchCase("gpu-512-lab", "cuda", 512, "lab", 2, 5),
    BenchCase("gpu-1024-lab", "cuda", 1024, "lab", 1, 3),
    BenchCase("gpu-2048-lab", "cuda", 2048, "lab", 0, 1),
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


def _make_pair(resolution: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + resolution)
    target = torch.rand((1, 3, resolution, resolution), generator=generator, dtype=torch.float32)
    reference = torch.rand((1, 3, resolution, resolution), generator=generator, dtype=torch.float32)
    return target.to(device=device), reference.to(device=device)


def _stats(samples_ms: list[float]) -> tuple[float, float, float]:
    return min(samples_ms), statistics.median(samples_ms), statistics.mean(samples_ms)


def _run_case(case: BenchCase) -> BenchResult:
    if case.device == "cuda" and not torch.cuda.is_available():
        return BenchResult(
            name=case.name,
            device=case.device,
            resolution=case.resolution,
            channels=case.channels,
            status="NOT EVALUATED",
            reason=SKIP_REASON_CUDA,
            min_ms=None,
            median_ms=None,
            mean_ms=None,
        )

    device = torch.device(case.device)
    target, reference = _make_pair(case.resolution, device)
    measure_once = _latency_runner(device)

    def run_once(
        target_bchw: torch.Tensor = target,
        reference_bchw: torch.Tensor = reference,
    ) -> torch.Tensor:
        return color_matcher(
            target_bchw,
            reference_bchw,
            channels=case.channels,
            strength=1.0,
        )

    with torch.inference_mode():
        for _ in range(case.warmup_iters):
            run_once()
        samples_ms = [measure_once(run_once) for _ in range(case.measure_iters)]

    stats = _stats(samples_ms)
    del target, reference
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return BenchResult(
        name=case.name,
        device=case.device,
        resolution=case.resolution,
        channels=case.channels,
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
        "kornia_version": kornia.__version__,
        "python_version": sys.version.split()[0],
        "platform": os.name,
        "seed": str(SEED),
    }


def _format_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def render_markdown(hardware: dict[str, str], results: list[BenchResult]) -> str:
    lines = [
        "# Color Matcher Benchmark",
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
            "| case | device | resolution | channels | status | reason | min_ms | "
            "median_ms | mean_ms |"
        ),
        "|---|---|---:|---|---|---|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row.name} | {row.device} | {row.resolution} | {row.channels} | "
            f"{row.status} | {row.reason or '—'} | {_format_metric(row.min_ms)} | "
            f"{_format_metric(row.median_ms)} | {_format_metric(row.mean_ms)} |"
        )
    return "\n".join(lines)


def main() -> int:
    torch.manual_seed(SEED)
    logging.getLogger("core.color_matcher").setLevel(logging.ERROR)
    hardware = _hardware_info()
    results = [_run_case(case) for case in CASES]
    print(render_markdown(hardware, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
