from __future__ import annotations

import gc
import logging
import math
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

WARMUP_ITERS = 3
MEASURE_ITERS = 10
STRESS_ITERS = 100
RESOLUTIONS = (512, 1024, 2048)
BATCHES = (1, 4)
MASK_VARIANTS = ("none", "mask50")
DEVICES = ("cpu", "cuda")
SEED = 20260418
DEFAULT_STRENGTH = 0.4
DEFAULT_SIGMA_COLOR = 0.1
DEFAULT_SIGMA_SPACE = 6.0
INVARIANT_CPU_BUDGET_MS = 50.0
INVARIANT_GPU_BUDGET_MS = 5.0
GPU_BUDGET_MS = 400.0
# H-20260418-1620 documented that bilateral smoothing at >=1024 on CPU is
# not a viable smoke path on this runner; keep the skip policy explicit and reproducible.
SKIP_CPU_SIZES = [1024, 2048]
SKIP_REASON_CPU = "NOT EVALUATED: bilateral CPU resource limit (H-20260418-1620)"
SKIP_REASON_CUDA = "NOT EVALUATED: CUDA not available on this runner"


@dataclass(slots=True)
class BenchRow:
    device: str
    resolution: int
    batch: int
    mask_variant: str
    status: str
    reason: str
    min_ms: float | None
    median_ms: float | None
    mean_ms: float | None
    p95_ms: float | None


@dataclass(slots=True)
class InvariantRow:
    device: str
    case_name: str
    status: str
    reason: str
    min_ms: float | None
    median_ms: float | None
    mean_ms: float | None
    p95_ms: float | None
    budget_ms: float
    passed_budget: bool | None


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


def _percentile(samples_ms: list[float], percentile: float) -> float:
    if not samples_ms:
        raise ValueError("samples_ms must not be empty.")

    ordered = sorted(samples_ms)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _stats(samples_ms: list[float]) -> tuple[float, float, float, float]:
    return (
        min(samples_ms),
        statistics.median(samples_ms),
        statistics.mean(samples_ms),
        _percentile(samples_ms, 0.95),
    )


def _make_image(batch: int, resolution: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    image = torch.rand((batch, 3, resolution, resolution), generator=generator, dtype=torch.float32)
    return image.to(device=device)


def _make_mask(batch: int, resolution: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    mask = torch.rand(
        (batch, 1, resolution, resolution),
        generator=generator,
        dtype=torch.float32,
    )
    mask = (mask > 0.5).to(dtype=torch.float32)
    return mask.to(device=device)


def _run_main_grid() -> list[BenchRow]:
    rows: list[BenchRow] = []

    for device_name in DEVICES:
        if device_name == "cuda" and not torch.cuda.is_available():
            for resolution in RESOLUTIONS:
                for batch in BATCHES:
                    for mask_variant in MASK_VARIANTS:
                        rows.append(
                            BenchRow(
                                device=device_name,
                                resolution=resolution,
                                batch=batch,
                                mask_variant=mask_variant,
                                status="NOT EVALUATED",
                                reason=SKIP_REASON_CUDA,
                                min_ms=None,
                                median_ms=None,
                                mean_ms=None,
                                p95_ms=None,
                            )
                        )
            continue

        device = torch.device(device_name)
        measure_once = _latency_runner(device)

        for resolution in RESOLUTIONS:
            for batch in BATCHES:
                for mask_variant in MASK_VARIANTS:
                    if device.type == "cpu" and resolution in SKIP_CPU_SIZES:
                        rows.append(
                            BenchRow(
                                device=device_name,
                                resolution=resolution,
                                batch=batch,
                                mask_variant=mask_variant,
                                status="NOT EVALUATED",
                                reason=SKIP_REASON_CPU,
                                min_ms=None,
                                median_ms=None,
                                mean_ms=None,
                                p95_ms=None,
                            )
                        )
                        continue

                    image = _make_image(batch, resolution, device, SEED + resolution + batch)
                    mask = (
                        None
                        if mask_variant == "none"
                        else _make_mask(batch, resolution, device, SEED + 1000 + resolution + batch)
                    )

                    with torch.inference_mode():
                        for _ in range(WARMUP_ITERS):
                            edge_aware_smooth(
                                image,
                                strength=DEFAULT_STRENGTH,
                                sigma_color=DEFAULT_SIGMA_COLOR,
                                sigma_space=DEFAULT_SIGMA_SPACE,
                                mask_bchw=mask,
                            )

                        def run_once(
                            image_tensor: torch.Tensor = image,
                            mask_tensor: torch.Tensor | None = mask,
                        ) -> None:
                            edge_aware_smooth(
                                image_tensor,
                                strength=DEFAULT_STRENGTH,
                                sigma_color=DEFAULT_SIGMA_COLOR,
                                sigma_space=DEFAULT_SIGMA_SPACE,
                                mask_bchw=mask_tensor,
                            )

                        samples_ms = [measure_once(run_once) for _ in range(MEASURE_ITERS)]

                    stats = _stats(samples_ms)
                    rows.append(
                        BenchRow(
                            device=device_name,
                            resolution=resolution,
                            batch=batch,
                            mask_variant=mask_variant,
                            status="EXECUTED",
                            reason="",
                            min_ms=stats[0],
                            median_ms=stats[1],
                            mean_ms=stats[2],
                            p95_ms=stats[3],
                        )
                    )

                    del image, mask
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

    return rows


def _run_invariant_rows() -> list[InvariantRow]:
    rows: list[InvariantRow] = []

    for device_name in DEVICES:
        if device_name == "cuda" and not torch.cuda.is_available():
            for case_name in ("strength=0", "mask=0"):
                rows.append(
                    InvariantRow(
                        device=device_name,
                        case_name=case_name,
                        status="NOT EVALUATED",
                        reason=SKIP_REASON_CUDA,
                        min_ms=None,
                        median_ms=None,
                        mean_ms=None,
                        p95_ms=None,
                        budget_ms=INVARIANT_GPU_BUDGET_MS,
                        passed_budget=None,
                    )
                )
            continue

        device = torch.device(device_name)
        measure_once = _latency_runner(device)
        image = _make_image(1, 512, device, SEED + 9000)
        zero_mask = torch.zeros((1, 1, 512, 512), dtype=torch.float32, device=device)

        cases = (
            (
                "strength=0",
                lambda image_tensor=image: edge_aware_smooth(
                    image_tensor,
                    strength=0.0,
                    sigma_color=DEFAULT_SIGMA_COLOR,
                    sigma_space=DEFAULT_SIGMA_SPACE,
                ),
            ),
            (
                "mask=0",
                lambda image_tensor=image, zero_mask_tensor=zero_mask: edge_aware_smooth(
                    image_tensor,
                    strength=1.0,
                    sigma_color=DEFAULT_SIGMA_COLOR,
                    sigma_space=DEFAULT_SIGMA_SPACE,
                    mask_bchw=zero_mask_tensor,
                ),
            ),
        )

        with torch.inference_mode():
            for case_name, runner in cases:
                for _ in range(WARMUP_ITERS):
                    runner()

                samples_ms = [measure_once(runner) for _ in range(MEASURE_ITERS)]
                stats = _stats(samples_ms)
                budget_ms = (
                    INVARIANT_GPU_BUDGET_MS if device.type == "cuda" else INVARIANT_CPU_BUDGET_MS
                )
                rows.append(
                    InvariantRow(
                        device=device_name,
                        case_name=case_name,
                        status="EXECUTED",
                        reason="",
                        min_ms=stats[0],
                        median_ms=stats[1],
                        mean_ms=stats[2],
                        p95_ms=stats[3],
                        budget_ms=budget_ms,
                        passed_budget=stats[1] < budget_ms,
                    )
                )

        del image, zero_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return rows


def _run_stress() -> dict[str, float | int | str | bool | None]:
    if not torch.cuda.is_available():
        return {
            "status": "NOT EVALUATED",
            "reason": SKIP_REASON_CUDA,
            "iterations": STRESS_ITERS,
            "memory_before": None,
            "memory_after": None,
            "delta_percent": None,
            "passed": None,
        }

    device = torch.device("cuda")
    image = _make_image(1, 2048, device, SEED + 12000)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    memory_before = torch.cuda.memory_allocated(device)

    with torch.inference_mode():
        for iteration in range(STRESS_ITERS):
            edge_aware_smooth(
                image,
                strength=DEFAULT_STRENGTH,
                sigma_color=DEFAULT_SIGMA_COLOR,
                sigma_space=DEFAULT_SIGMA_SPACE,
            )
            if iteration == 49:
                torch.cuda.empty_cache()

    torch.cuda.synchronize(device)
    memory_after = torch.cuda.memory_allocated(device)
    delta_bytes = memory_after - memory_before
    delta_percent = 0.0 if memory_before == 0 else (delta_bytes / memory_before) * 100.0
    return {
        "status": "EXECUTED",
        "reason": "",
        "iterations": STRESS_ITERS,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "delta_percent": delta_percent,
        "passed": delta_percent < 5.0,
    }


def _hardware_info() -> dict[str, str]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        properties = torch.cuda.get_device_properties(device)
        cuda_name = properties.name
        total_vram_gb = f"{properties.total_memory / (1024 ** 3):.2f}"
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


def render_markdown(
    hardware: dict[str, str],
    main_rows: list[BenchRow],
    invariant_rows: list[InvariantRow],
    stress: dict[str, float | int | str | bool | None],
) -> str:
    lines = [
        "# Edge-Aware Smoother Benchmark",
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
        "- Warmup: 3 iterations per executed cell.",
        "- Timed: 10 iterations, report min / median / mean / p95.",
        (
            f"- Defaults: strength={DEFAULT_STRENGTH}, sigma_color={DEFAULT_SIGMA_COLOR}, "
            f"sigma_space={DEFAULT_SIGMA_SPACE}."
        ),
        f"- CPU skip sizes: {SKIP_CPU_SIZES} with reason `{SKIP_REASON_CPU}`.",
        f"- GPU unavailable reason: `{SKIP_REASON_CUDA}`.",
        "",
        "## Main Grid",
        "",
        (
            "| device | resolution | batch | mask | status | reason | "
            "min_ms | median_ms | mean_ms | p95_ms |"
        ),
        "|---|---:|---:|---|---|---|---:|---:|---:|---:|",
    ]

    for row in main_rows:
        lines.append(
            f"| {row.device} | {row.resolution} | {row.batch} | {row.mask_variant} | "
            f"{row.status} | {row.reason or '—'} | {_format_metric(row.min_ms)} | "
            f"{_format_metric(row.median_ms)} | {_format_metric(row.mean_ms)} | "
            f"{_format_metric(row.p95_ms)} |"
        )

    lines.extend(
        [
            "",
            "## Invariant Cells",
            "",
            (
                "| device | case | status | reason | min_ms | median_ms | mean_ms | "
                "p95_ms | budget_ms | passed |"
            ),
            "|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )

    for row in invariant_rows:
        passed = "—" if row.passed_budget is None else ("PASS" if row.passed_budget else "FAIL")
        lines.append(
            f"| {row.device} | {row.case_name} | {row.status} | {row.reason or '—'} | "
            f"{_format_metric(row.min_ms)} | {_format_metric(row.median_ms)} | "
            f"{_format_metric(row.mean_ms)} | {_format_metric(row.p95_ms)} | "
            f"{row.budget_ms:.3f} | {passed} |"
        )

    lines.extend(["", "## Stress", ""])

    if stress["status"] == "EXECUTED":
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

    gpu_budget_row = next(
        (
            row
            for row in main_rows
            if row.device == "cuda"
            and row.resolution == 2048
            and row.batch == 1
            and row.mask_variant == "none"
        ),
        None,
    )
    lines.extend(["", "## Verdict", ""])

    if gpu_budget_row is None or gpu_budget_row.status != "EXECUTED":
        lines.append(f"- GPU 2K budget (< {GPU_BUDGET_MS:.0f} ms): NOT EVALUATED")
    else:
        verdict = (
            "PASS"
            if gpu_budget_row.median_ms is not None and gpu_budget_row.median_ms < GPU_BUDGET_MS
            else "FAIL"
        )
        lines.append(
            f"- GPU 2K budget (< {GPU_BUDGET_MS:.0f} ms): {verdict} "
            f"(median {gpu_budget_row.median_ms:.3f} ms)"
        )

    cpu_executed = sum(1 for row in main_rows if row.device == "cpu" and row.status == "EXECUTED")
    cpu_skipped = sum(
        1 for row in main_rows if row.device == "cpu" and row.status == "NOT EVALUATED"
    )
    lines.append(
        f"- CPU smoke: {cpu_executed} executed cell(s), "
        f"{cpu_skipped} NOT EVALUATED heavy cell(s)"
    )
    lines.append("- Rescope reference: H-20260418-1620 -> H-20260418-1630")
    return "\n".join(lines)


def main() -> int:
    torch.manual_seed(SEED)
    logging.getLogger("core.smoother").setLevel(logging.ERROR)
    hardware = _hardware_info()
    main_rows = _run_main_grid()
    invariant_rows = _run_invariant_rows()
    stress = _run_stress()

    for row in invariant_rows:
        if row.status == "EXECUTED" and row.passed_budget is False:
            raise AssertionError(
                f"Invariant budget failed for {row.device} {row.case_name}: {row.median_ms:.3f} ms"
            )

    budget_row = next(
        (
            row
            for row in main_rows
            if row.device == "cuda"
            and row.resolution == 2048
            and row.batch == 1
            and row.mask_variant == "none"
            and row.status == "EXECUTED"
        ),
        None,
    )
    if (
        budget_row is not None
        and budget_row.median_ms is not None
        and budget_row.median_ms >= GPU_BUDGET_MS
    ):
        raise AssertionError(
            f"GPU 2K budget failed: median {budget_row.median_ms:.3f} ms >= {GPU_BUDGET_MS:.0f} ms."
        )

    if stress["status"] == "EXECUTED" and stress["passed"] is False:
        raise AssertionError(
            f"GPU stress delta failed: {float(stress['delta_percent']):.3f}% >= 5.000%."
        )

    print(render_markdown(hardware, main_rows, invariant_rows, stress))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
