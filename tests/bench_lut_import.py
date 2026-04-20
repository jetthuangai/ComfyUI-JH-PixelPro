from __future__ import annotations

import gc
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

from core.lut import apply_lut_3d  # noqa: E402

SEED = 20260420
SKIP_REASON_CUDA = "GPU NOT EVALUATED — Codex runner CPU-only per bench_color_matcher precedent"


@dataclass(slots=True)
class CpuBenchResult:
    resolution: int
    lut_size: int
    status: str
    median_ms: float | None
    p95_ms: float | None
    notes: str


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


def _identity_lut_grid(size: int, device: torch.device) -> torch.Tensor:
    coords = torch.linspace(0.0, 1.0, size, dtype=torch.float32, device=device)
    blue, green, red = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.stack([red, green, blue], dim=-1)


def _make_image(resolution: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + resolution)
    image = torch.rand((1, resolution, resolution, 3), generator=generator, dtype=torch.float32)
    return image.to(device=device)


def _p95(samples_ms: list[float]) -> float:
    if len(samples_ms) == 1:
        return samples_ms[0]
    return statistics.quantiles(samples_ms, n=100, method="inclusive")[94]


def _run_cpu_case(resolution: int, lut_size: int) -> CpuBenchResult:
    device = torch.device("cpu")
    image = _make_image(resolution, device)
    lut = _identity_lut_grid(lut_size, device)
    measure_once = _latency_runner(device)

    def run_once(
        image_bhwc: torch.Tensor = image,
        lut_grid: torch.Tensor = lut,
    ) -> torch.Tensor:
        return apply_lut_3d(image_bhwc, lut_grid)

    try:
        with torch.inference_mode():
            run_once()
            samples_ms = [measure_once(run_once) for _ in range(3)]
    except RuntimeError as exc:
        message = str(exc).lower()
        if (
            "out of memory" in message
            or "can't allocate memory" in message
            or "bad allocation" in message
        ):
            return CpuBenchResult(
                resolution=resolution,
                lut_size=lut_size,
                status="SKIPPED",
                median_ms=None,
                p95_ms=None,
                notes="CPU OOM on Codex runner — T-N-07-a precedent",
            )
        raise
    finally:
        del image, lut
        gc.collect()

    return CpuBenchResult(
        resolution=resolution,
        lut_size=lut_size,
        status="EXECUTED",
        median_ms=statistics.median(samples_ms),
        p95_ms=_p95(samples_ms),
        notes="identity LUT apply",
    )


def _gpu_parity_markdown() -> tuple[str, str]:
    if not torch.cuda.is_available():
        return "NOT EVALUATED", SKIP_REASON_CUDA

    cpu_device = torch.device("cpu")
    cuda_device = torch.device("cuda")
    image_cpu = _make_image(1024, cpu_device)
    lut_cpu = _identity_lut_grid(33, cpu_device)
    expected = apply_lut_3d(image_cpu, lut_cpu)
    actual = apply_lut_3d(image_cpu.to(cuda_device), lut_cpu.to(cuda_device)).cpu()
    max_diff = torch.max(torch.abs(actual - expected)).item()
    return "EXECUTED", f"1024x1024 LUT=33 max_abs_diff={max_diff:.6f}"


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


def render_markdown(hardware: dict[str, str], cpu_results: list[CpuBenchResult]) -> str:
    gpu_status, gpu_notes = _gpu_parity_markdown()
    lines = [
        "# LUT Import Benchmark",
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
        "## CPU Grid",
        "",
        "| size_px x lut_N | median_ms | p95_ms | notes |",
        "|---|---:|---:|---|",
    ]
    for row in cpu_results:
        lines.append(
            f"| {row.resolution} x {row.lut_size} | {_format_metric(row.median_ms)} | "
            f"{_format_metric(row.p95_ms)} | {row.status}: {row.notes} |"
        )

    lines.extend(
        [
            "",
            "## GPU Parity",
            "",
            f"- Status: {gpu_status}",
            f"- Notes: {gpu_notes}",
            "",
            "## Conclusion",
            "",
            (
                "Bench is observational only. CPU grid executed for identity LUT apply across "
                "{512,1024,2048} x {17,33,65}; GPU parity is a single 1024/33 smoke cell or "
                "NOT EVALUATED on CPU-only runners."
            ),
        ]
    )
    return "\n".join(lines)


def main() -> int:
    torch.manual_seed(SEED)
    hardware = _hardware_info()
    cpu_results = [
        _run_cpu_case(resolution, lut_size)
        for resolution in (512, 1024, 2048)
        for lut_size in (17, 33, 65)
    ]
    print(render_markdown(hardware, cpu_results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
