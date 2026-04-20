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

from core import edge_aware_smooth  # noqa: E402

SEED = 20260418
SKIP_REASON_CUDA = "NOT EVALUATED: CUDA not available on this runner"


@dataclass(slots=True)
class BenchCase:
    name: str
    device: str
    resolution: int
    sigma_space: float
    tile_mode: bool
    warmup_iters: int
    measure_iters: int
    expect_runtime_error: bool = False


@dataclass(slots=True)
class BenchResult:
    name: str
    device: str
    resolution: int
    sigma_space: float
    tile_mode: bool
    status: str
    reason: str
    min_ms: float | None
    median_ms: float | None
    mean_ms: float | None
    peak_memory_mb: float | None


CASES = [
    BenchCase(
        name="gpu-1k-sigma6-nontile",
        device="cuda",
        resolution=1024,
        sigma_space=6.0,
        tile_mode=False,
        warmup_iters=1,
        measure_iters=3,
    ),
    BenchCase(
        name="gpu-2k-sigma6-nontile",
        device="cuda",
        resolution=2048,
        sigma_space=6.0,
        tile_mode=False,
        warmup_iters=0,
        measure_iters=1,
        expect_runtime_error=True,
    ),
    BenchCase(
        name="gpu-2k-sigma6-tile",
        device="cuda",
        resolution=2048,
        sigma_space=6.0,
        tile_mode=True,
        warmup_iters=1,
        measure_iters=3,
    ),
    BenchCase(
        name="gpu-2k-sigma8-tile",
        device="cuda",
        resolution=2048,
        sigma_space=8.0,
        tile_mode=True,
        warmup_iters=1,
        measure_iters=3,
    ),
    BenchCase(
        name="cpu-4k-sigma6-tile",
        device="cpu",
        resolution=4096,
        sigma_space=6.0,
        tile_mode=True,
        warmup_iters=0,
        measure_iters=1,
    ),
    BenchCase(
        name="cpu-4k-sigma6-nontile",
        device="cpu",
        resolution=4096,
        sigma_space=6.0,
        tile_mode=False,
        warmup_iters=0,
        measure_iters=1,
        expect_runtime_error=True,
    ),
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


def _make_image(resolution: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
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
            sigma_space=case.sigma_space,
            tile_mode=case.tile_mode,
            status="NOT EVALUATED",
            reason=SKIP_REASON_CUDA,
            min_ms=None,
            median_ms=None,
            mean_ms=None,
            peak_memory_mb=None,
        )

    device = torch.device(case.device)
    measure_once = _latency_runner(device)
    image = _make_image(case.resolution, device, SEED + case.resolution)

    def run_once(image_tensor: torch.Tensor = image) -> torch.Tensor:
        return edge_aware_smooth(
            image_tensor,
            sigma_space=case.sigma_space,
            device=case.device,
            tile_mode=case.tile_mode,
        )

    if case.expect_runtime_error:
        try:
            run_once()
        except RuntimeError as exc:
            return BenchResult(
                name=case.name,
                device=case.device,
                resolution=case.resolution,
                sigma_space=case.sigma_space,
                tile_mode=case.tile_mode,
                status="EXPECTED_ERROR",
                reason=str(exc),
                min_ms=None,
                median_ms=None,
                mean_ms=None,
                peak_memory_mb=None,
            )
        raise AssertionError(f"{case.name} was expected to raise RuntimeError.")

    peak_memory_mb: float | None = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(case.warmup_iters):
            run_once()

        samples_ms = [measure_once(run_once) for _ in range(case.measure_iters)]

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    stats = _stats(samples_ms)
    del image
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return BenchResult(
        name=case.name,
        device=case.device,
        resolution=case.resolution,
        sigma_space=case.sigma_space,
        tile_mode=case.tile_mode,
        status="EXECUTED",
        reason="",
        min_ms=stats[0],
        median_ms=stats[1],
        mean_ms=stats[2],
        peak_memory_mb=peak_memory_mb,
    )


def _hardware_info() -> dict[str, str]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        properties = torch.cuda.get_device_properties(device)
        cuda_name = properties.name
        total_vram_gb = f"{properties.total_memory / (1024**3):.2f}"
    else:
        cuda_name = "n/a"
        total_vram_gb = "n/a"

    return {
        "cuda_available": str(torch.cuda.is_available()).lower(),
        "device_name": cuda_name,
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
        "# Edge-Aware Smoother Benchmark v1.1",
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
        "## Methodology",
        "",
        "- Hotfix cases from S-03 v1.1 / T-11.",
        "- GPU-unavailable cases render as NOT EVALUATED.",
        "- Runtime-error cases are expected and reported explicitly.",
        (
            "- CPU peak memory is reported as n/a because PyTorch does not expose "
            "a reliable CPU peak allocator metric here."
        ),
        "",
        "## Cases",
        "",
        (
            "| case | device | resolution | sigma_space | tile_mode | status | reason | "
            "min_ms | median_ms | mean_ms | peak_memory_mb |"
        ),
        "|---|---|---:|---:|---|---|---|---:|---:|---:|---:|",
    ]

    for row in results:
        lines.append(
            f"| {row.name} | {row.device} | {row.resolution} | {row.sigma_space:.1f} | "
            f"{row.tile_mode} | {row.status} | {row.reason or '—'} | "
            f"{_format_metric(row.min_ms)} | {_format_metric(row.median_ms)} | "
            f"{_format_metric(row.mean_ms)} | {_format_metric(row.peak_memory_mb)} |"
        )

    return "\n".join(lines)


def main() -> int:
    torch.manual_seed(SEED)
    logging.getLogger("core.smoother").setLevel(logging.ERROR)
    hardware = _hardware_info()
    results = [_run_case(case) for case in CASES]
    print(render_markdown(hardware, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
