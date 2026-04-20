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

from core import subpixel_mask_refine  # noqa: E402

WARMUP_ITERS = 3
MEASURE_ITERS = 10
STRESS_ITERS = 100
RESOLUTIONS = (512, 1024, 2048)
BATCHES = (1, 4)
CONFIGS = ((2, 4, 2.0), (4, 8, 3.0))
SHORT_CIRCUIT_CONFIG = (0, 0, 2.0)
SEED = 20260417


@dataclass(slots=True)
class BenchCell:
    device: str
    resolution: int
    batch: int
    erosion_radius: int
    dilation_radius: int
    feather_sigma: float
    median_ms: float


def _device_for_run() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def _make_disk_mask(batch: int, resolution: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(resolution, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    center = (resolution - 1) / 2.0
    radius = resolution / 4.0
    disk = ((yy - center) ** 2 + (xx - center) ** 2 <= radius**2).to(dtype=torch.float32)
    return disk.unsqueeze(0).repeat(batch, 1, 1)


def _run_grid(device: torch.device) -> list[BenchCell]:
    results: list[BenchCell] = []
    measure_once = _latency_runner(device)

    for resolution in RESOLUTIONS:
        for batch in BATCHES:
            for erosion_radius, dilation_radius, feather_sigma in CONFIGS:
                mask = _make_disk_mask(batch, resolution, device)

                with torch.inference_mode():
                    for _ in range(WARMUP_ITERS):
                        subpixel_mask_refine(
                            mask,
                            erosion_radius=erosion_radius,
                            dilation_radius=dilation_radius,
                            feather_sigma=feather_sigma,
                        )

                    def run_once(
                        mask_tensor: torch.Tensor = mask,
                        er: int = erosion_radius,
                        dr: int = dilation_radius,
                        sigma: float = feather_sigma,
                    ) -> None:
                        subpixel_mask_refine(
                            mask_tensor,
                            erosion_radius=er,
                            dilation_radius=dr,
                            feather_sigma=sigma,
                        )

                    samples_ms = [measure_once(run_once) for _ in range(MEASURE_ITERS)]

                results.append(
                    BenchCell(
                        device=device.type,
                        resolution=resolution,
                        batch=batch,
                        erosion_radius=erosion_radius,
                        dilation_radius=dilation_radius,
                        feather_sigma=feather_sigma,
                        median_ms=statistics.median(samples_ms),
                    )
                )

                del mask
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

    return results


def _short_circuit_smoke(device: torch.device) -> dict[str, float | int]:
    measure_once = _latency_runner(device)
    mask = _make_disk_mask(batch=1, resolution=512, device=device)
    erosion_radius, dilation_radius, feather_sigma = SHORT_CIRCUIT_CONFIG

    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            subpixel_mask_refine(
                mask,
                erosion_radius=erosion_radius,
                dilation_radius=dilation_radius,
                feather_sigma=feather_sigma,
            )

        latency_ms = measure_once(
            lambda: subpixel_mask_refine(
                mask,
                erosion_radius=erosion_radius,
                dilation_radius=dilation_radius,
                feather_sigma=feather_sigma,
            )
        )

    return {
        "resolution": 512,
        "batch": 1,
        "erosion_radius": erosion_radius,
        "dilation_radius": dilation_radius,
        "feather_sigma": feather_sigma,
        "latency_ms": latency_ms,
    }


def _stress_test(device: torch.device) -> dict[str, float | int | str | bool | None]:
    if device.type != "cuda":
        return {
            "mode": "skipped",
            "reason": "stress test requires CUDA — skipped on CPU runner",
            "iterations": STRESS_ITERS,
            "memory_before": None,
            "memory_after": None,
            "delta_percent": None,
            "passed": None,
        }

    mask = _make_disk_mask(batch=1, resolution=2048, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    memory_before = torch.cuda.memory_allocated(device)

    with torch.inference_mode():
        for iteration in range(STRESS_ITERS):
            subpixel_mask_refine(
                mask,
                erosion_radius=4,
                dilation_radius=8,
                feather_sigma=3.0,
            )
            if iteration == 49:
                torch.cuda.empty_cache()

    torch.cuda.synchronize(device)
    memory_after = torch.cuda.memory_allocated(device)
    delta_bytes = memory_after - memory_before
    delta_percent = 0.0 if memory_before == 0 else (delta_bytes / memory_before) * 100.0

    return {
        "mode": "cuda-vram",
        "reason": "",
        "iterations": STRESS_ITERS,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "delta_percent": delta_percent,
        "passed": delta_percent < 5.0,
    }


def _hardware_info(device: torch.device) -> dict[str, str]:
    cuda_name = "n/a"
    total_vram_gb = "n/a"
    cuda_runtime = torch.version.cuda or "n/a"

    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        cuda_name = properties.name
        total_vram_gb = f"{properties.total_memory / (1024**3):.2f}"

    return {
        "device_type": device.type,
        "device_name": cuda_name,
        "total_vram_gb": total_vram_gb,
        "cuda_runtime": cuda_runtime,
        "torch_version": torch.__version__,
        "kornia_version": kornia.__version__,
        "python_version": sys.version.split()[0],
        "platform": os.name,
        "seed": str(SEED),
    }


def render_markdown(
    hardware: dict[str, str],
    grid_results: list[BenchCell],
    short_circuit: dict[str, float | int],
    stress: dict[str, float | int | str | bool | None],
) -> str:
    lines = [
        "# Sub-Pixel Mask Refiner Benchmark",
        "",
        "## Hardware",
        "",
        f"- Device type: {hardware['device_type']}",
        f"- Device name: {hardware['device_name']}",
        f"- Total VRAM (GiB): {hardware['total_vram_gb']}",
        f"- CUDA runtime: {hardware['cuda_runtime']}",
        f"- PyTorch: {hardware['torch_version']}",
        f"- Kornia: {hardware['kornia_version']}",
        f"- Python: {hardware['python_version']}",
        f"- OS family: {hardware['platform']}",
        f"- Seed: {hardware['seed']}",
        "",
        "## Grid",
        "",
        "| device | resolution | batch | er | dr | sigma | median_ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for cell in grid_results:
        lines.append(
            f"| {cell.device} | {cell.resolution} | {cell.batch} | "
            f"{cell.erosion_radius} | {cell.dilation_radius} | "
            f"{cell.feather_sigma:.1f} | {cell.median_ms:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Short-Circuit Smoke",
            "",
            f"- Resolution: {int(short_circuit['resolution'])}",
            f"- Batch: {int(short_circuit['batch'])}",
            f"- Config: er={int(short_circuit['erosion_radius'])}, "
            f"dr={int(short_circuit['dilation_radius'])}, "
            f"sigma={float(short_circuit['feather_sigma']):.1f}",
            f"- Latency ms: {float(short_circuit['latency_ms']):.3f}",
            "",
            "## Stress",
            "",
        ]
    )

    if stress["mode"] == "cuda-vram":
        lines.extend(
            [
                f"- Iterations: {stress['iterations']}",
                f"- Memory before (bytes): {int(stress['memory_before'])}",
                f"- Memory after (bytes): {int(stress['memory_after'])}",
                f"- Delta percent: {float(stress['delta_percent']):.3f}",
                f"- Verdict: {'PASS' if stress['passed'] else 'FAIL'}",
            ]
        )
    else:
        lines.extend(
            [
                f"- Iterations: {stress['iterations']}",
                f"- Verdict: {stress['reason']}",
            ]
        )

    return "\n".join(lines)


def main() -> int:
    torch.manual_seed(SEED)
    device = _device_for_run()
    logging.getLogger("core.mask_refiner").setLevel(logging.ERROR)
    hardware = _hardware_info(device)
    grid_results = _run_grid(device)
    short_circuit = _short_circuit_smoke(device)
    stress = _stress_test(device)
    print(render_markdown(hardware, grid_results, short_circuit, stress))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
