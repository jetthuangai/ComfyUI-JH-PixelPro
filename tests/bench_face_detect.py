from __future__ import annotations

import gc
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import kornia
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import face_detect  # noqa: E402

SEED = 20260419


@dataclass(slots=True)
class BenchCase:
    name: str
    long_side: int
    warmup_iters: int
    measure_iters: int


@dataclass(slots=True)
class BenchResult:
    name: str
    long_side: int
    status: str
    reason: str
    min_ms: float | None
    median_ms: float | None
    mean_ms: float | None


CASES = [
    BenchCase("cpu-512", 512, 1, 3),
    BenchCase("cpu-1024", 1024, 1, 3),
    BenchCase("cpu-2048", 2048, 1, 3),
]


def _latency_runner():
    def measure_once(fn) -> float:
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        return (end - start) * 1000.0

    return measure_once


def _load_image(long_side: int) -> torch.Tensor:
    image_path = REPO_ROOT / "workflows" / "sample_portrait.jpg"
    image = (
        torch.from_numpy(np.array(Image.open(image_path).convert("RGB"), copy=True)).float() / 255.0
    )
    bchw = image.permute(2, 0, 1).unsqueeze(0)
    _, _, height, width = bchw.shape
    scale = long_side / max(height, width)
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    resized = interpolate(
        bchw,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.permute(0, 2, 3, 1).contiguous()


def _stats(samples_ms: list[float]) -> tuple[float, float, float]:
    return min(samples_ms), statistics.median(samples_ms), statistics.mean(samples_ms)


def _run_case(case: BenchCase) -> BenchResult:
    image = _load_image(case.long_side)
    measure_once = _latency_runner()

    def run_once(
        image_bhwc: torch.Tensor = image,
    ) -> tuple[list[list[list[float]]], list[dict], int]:
        return face_detect(
            image_bhwc,
            mode="single_largest",
            max_faces=1,
            confidence_threshold=0.5,
        )

    with torch.inference_mode():
        for _ in range(case.warmup_iters):
            run_once()
        samples_ms = [measure_once(run_once) for _ in range(case.measure_iters)]

    stats = _stats(samples_ms)
    del image
    gc.collect()
    return BenchResult(
        name=case.name,
        long_side=case.long_side,
        status="EXECUTED",
        reason="",
        min_ms=stats[0],
        median_ms=stats[1],
        mean_ms=stats[2],
    )


def _hardware_info() -> dict[str, str]:
    return {
        "cuda_available": str(torch.cuda.is_available()).lower(),
        "device_name": "CPU (MediaPipe tasks)",
        "total_vram_gb": "n/a",
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
        "# Face Detect Benchmark",
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
        "| case | long_side | status | reason | min_ms | median_ms | mean_ms |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row.name} | {row.long_side} | {row.status} | {row.reason or '—'} | "
            f"{_format_metric(row.min_ms)} | {_format_metric(row.median_ms)} | "
            f"{_format_metric(row.mean_ms)} |"
        )
    return "\n".join(lines)


def main() -> int:
    torch.manual_seed(SEED)
    results = [_run_case(case) for case in CASES]
    print(render_markdown(_hardware_info(), results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
