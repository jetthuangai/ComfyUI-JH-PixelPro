from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import kornia
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import frequency_separation  # noqa: E402
from tests.conftest import (  # noqa: E402
    BENCH_N_RUNS,
    assert_bench_within_threshold,
    build_multi_snapshot_fields,
    stabilize_bench_payload,
)

BASELINE_PATH = Path(__file__).with_name("bench_baselines") / "bench_frequency.json"
WRITE_BASELINES = os.environ.get("JH_PIXELPRO_WRITE_BENCH_BASELINES") == "1"
RADIUS = 8
WARMUP_ITERS = 3
MEASURE_ITERS = 10
STRESS_ITERS = 100
RESOLUTIONS = (512, 1024, 2048)
BATCHES = (1, 4)
PRECISIONS = ("float32", "float16")


@dataclass(slots=True)
class BenchCell:
    device: str
    resolution: int
    batch: int
    precision: str
    median_ms: float
    mean_ms: float = 0.0
    stdev_ms: float = 0.0
    n_samples: int = 0
    n_runs: int = 0
    measure_iters: int = 0
    policy: str = ""


def _torch_dtype(precision: str) -> torch.dtype:
    return torch.float16 if precision == "float16" else torch.float32


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


def _make_input(
    batch: int,
    resolution: int,
    precision: str,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.rand(
        (batch, 3, resolution, resolution),
        generator=generator,
        device=device,
        dtype=_torch_dtype(precision),
    )


def _run_grid(device: torch.device) -> list[BenchCell]:
    results: list[BenchCell] = []
    measure_once = _latency_runner(device)

    for resolution in RESOLUTIONS:
        for batch in BATCHES:
            for precision in PRECISIONS:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(20260417)
                image = _make_input(batch, resolution, precision, device, generator)

                def run_once(
                    image_tensor: torch.Tensor = image,
                    selected_precision: str = precision,
                ) -> None:
                    frequency_separation(
                        image_tensor,
                        radius=RADIUS,
                        precision=selected_precision,
                    )

                samples_ms: list[float] = []
                with torch.inference_mode():
                    for _run in range(BENCH_N_RUNS):
                        for _ in range(WARMUP_ITERS):
                            frequency_separation(image, radius=RADIUS, precision=precision)
                        for _ in range(MEASURE_ITERS):
                            samples_ms.append(measure_once(run_once))

                multi = build_multi_snapshot_fields(samples_ms, measure_iters=MEASURE_ITERS)
                results.append(
                    BenchCell(
                        device=device.type,
                        resolution=resolution,
                        batch=batch,
                        precision=precision,
                        median_ms=multi["median_ms"],
                        mean_ms=multi["mean_ms"],
                        stdev_ms=multi["stdev_ms"],
                        n_samples=multi["n_samples"],
                        n_runs=multi["n_runs"],
                        measure_iters=multi["measure_iters"],
                        policy=multi["policy"],
                    )
                )

                del image
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

    return results


def _stress_test(device: torch.device) -> dict[str, float | int | str | bool | None]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260417)
    image = _make_input(1, 2048, "float32", device, generator)

    if device.type != "cuda":
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(STRESS_ITERS):
                frequency_separation(image, radius=RADIUS, precision="float32")
        ended = time.perf_counter()
        return {
            "mode": "cpu-smoke",
            "iterations": STRESS_ITERS,
            "elapsed_s": ended - started,
            "memory_before": None,
            "memory_after": None,
            "delta_percent": None,
            "passed": True,
        }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    memory_before = torch.cuda.memory_allocated(device)

    with torch.inference_mode():
        for iteration in range(STRESS_ITERS):
            frequency_separation(image, radius=RADIUS, precision="float32")
            if iteration == 49:
                torch.cuda.empty_cache()

    torch.cuda.synchronize(device)
    memory_after = torch.cuda.memory_allocated(device)
    delta_bytes = memory_after - memory_before
    delta_percent = 0.0 if memory_before == 0 else (delta_bytes / memory_before) * 100.0

    return {
        "mode": "cuda-vram",
        "iterations": STRESS_ITERS,
        "elapsed_s": None,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "delta_percent": delta_percent,
        "passed": delta_percent < 5.0,
    }


def _load_baselines() -> list[dict[str, object]]:
    if not BASELINE_PATH.exists():
        return []
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _write_baseline(payload: dict[str, object]) -> None:
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = {str(row["case"]): row for row in _load_baselines()}
    rows[str(payload["case"])] = payload
    ordered = [rows[key] for key in sorted(rows)]
    BASELINE_PATH.write_text(
        json.dumps(ordered, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    }


def render_markdown(
    hardware: dict[str, str],
    grid_results: list[BenchCell],
    stress: dict[str, float | int | str | bool | None],
) -> str:
    lines = [
        "# Frequency Separation Benchmark",
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
        "",
        "## Grid",
        "",
        "| device | resolution | batch | precision | median_ms |",
        "|---|---:|---:|---|---:|",
    ]

    for cell in grid_results:
        lines.append(
            f"| {cell.device} | {cell.resolution} | {cell.batch} | "
            f"{cell.precision} | {cell.median_ms:.3f} |"
        )

    lines.extend(["", "## Stress", ""])

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
                f"- Elapsed seconds: {float(stress['elapsed_s']):.3f}",
                "- VRAM delta percent: not evaluated (CPU-only runner)",
                "- Verdict: smoke only",
            ]
        )

    return "\n".join(lines)


def _payload_from_result(result: object) -> dict[str, object]:
    payload = asdict(result) if is_dataclass(result) else dict(result)
    if "case" not in payload and "name" in payload:
        payload["case"] = payload["name"]
    if payload.get("mean_ms") is None and payload.get("median_ms") is not None:
        payload["mean_ms"] = payload["median_ms"]
    payload.setdefault("module", Path(__file__).stem.removeprefix("bench_"))
    payload.setdefault("device", "cpu")
    return payload


def _assert_bench_guardrail(payload: dict[str, object]) -> None:
    if WRITE_BASELINES:
        _write_baseline(stabilize_bench_payload(payload))
        return

    baseline_rows = _load_baselines()
    baseline_row = next(
        (row for row in baseline_rows if row["case"] == payload["case"]),
        None,
    )
    assert baseline_row is not None, f"baseline missing for {payload['case']}"
    stabilized = stabilize_bench_payload(payload, baseline_row)
    assert_bench_within_threshold(stabilized, baseline_row)


def _assert_baseline_present(payload: dict[str, object]) -> None:
    _assert_bench_guardrail(payload)


def _payload_from_frequency_cell(cell: BenchCell) -> dict[str, object]:
    return {
        "case": f"{cell.device}-{cell.resolution}-b{cell.batch}-{cell.precision}",
        "module": "frequency",
        "scenario": "frequency_separation",
        "device": cell.device,
        "resolution": cell.resolution,
        "batch": cell.batch,
        "precision": cell.precision,
        "mean_ms": round(cell.mean_ms, 4),
        "median_ms": round(cell.median_ms, 4),
        "stdev_ms": round(cell.stdev_ms, 4),
        "n_samples": cell.n_samples,
        "n_runs": cell.n_runs,
        "measure_iters": cell.measure_iters,
        "warmup_iters": WARMUP_ITERS,
        "policy": cell.policy,
    }


@pytest.mark.bench_guardrail
def test_bench_frequency(capsys: pytest.CaptureFixture[str]) -> None:
    device = _device_for_run()
    payloads = [_payload_from_frequency_cell(cell) for cell in _run_grid(device)]
    for payload in payloads:
        _assert_bench_guardrail(payload)
    with capsys.disabled():
        print(json.dumps(payloads, sort_keys=True))


def main() -> int:
    device = _device_for_run()
    logging.getLogger("core.frequency").setLevel(logging.ERROR)
    hardware = _hardware_info(device)
    grid_results = _run_grid(device)
    stress = _stress_test(device)
    print(render_markdown(hardware, grid_results, stress))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
