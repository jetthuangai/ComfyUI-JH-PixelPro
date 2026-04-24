from __future__ import annotations

import gc
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.lut import apply_lut_3d  # noqa: E402
from tests.conftest import (  # noqa: E402
    BENCH_N_RUNS,
    assert_bench_within_threshold,
    build_multi_snapshot_fields,
    stabilize_bench_payload,
)

SEED = 20260420
SKIP_REASON_CUDA = "GPU NOT EVALUATED — Codex runner CPU-only per bench_color_matcher precedent"
BASELINE_PATH = Path(__file__).with_name("bench_baselines") / "bench_lut_import.json"
WRITE_BASELINES = os.environ.get("JH_PIXELPRO_WRITE_BENCH_BASELINES") == "1"


MEASURE_ITERS = 3
WARMUP_ITERS = 1


@dataclass(slots=True)
class CpuBenchResult:
    resolution: int
    lut_size: int
    status: str
    median_ms: float | None
    p95_ms: float | None
    notes: str
    mean_ms: float | None = None
    stdev_ms: float | None = None
    n_samples: int | None = None
    n_runs: int | None = None
    measure_iters: int | None = None
    warmup_iters: int | None = None
    policy: str | None = None


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

    samples_ms: list[float] = []
    try:
        with torch.inference_mode():
            for _run in range(BENCH_N_RUNS):
                for _ in range(WARMUP_ITERS):
                    run_once()
                for _ in range(MEASURE_ITERS):
                    samples_ms.append(measure_once(run_once))
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

    multi = build_multi_snapshot_fields(samples_ms, measure_iters=MEASURE_ITERS)
    return CpuBenchResult(
        resolution=resolution,
        lut_size=lut_size,
        status="EXECUTED",
        median_ms=multi["median_ms"],
        p95_ms=_p95(samples_ms),
        notes="identity LUT apply",
        mean_ms=multi["mean_ms"],
        stdev_ms=multi["stdev_ms"],
        n_samples=multi["n_samples"],
        n_runs=multi["n_runs"],
        measure_iters=multi["measure_iters"],
        warmup_iters=WARMUP_ITERS,
        policy=multi["policy"],
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


def _payload_from_lut_import_result(result: CpuBenchResult) -> dict[str, object]:
    payload = _payload_from_result(result)
    payload["case"] = f"cpu-{result.resolution}-lut{result.lut_size}"
    payload["module"] = "lut_import"
    payload["scenario"] = "identity_lut_apply"
    payload["device"] = "cpu"
    return payload


@pytest.mark.bench_guardrail
@pytest.mark.parametrize(
    ("resolution", "lut_size"),
    [(resolution, lut_size) for resolution in (512, 1024, 2048) for lut_size in (17, 33, 65)],
)
def test_bench_lut_import(
    resolution: int,
    lut_size: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _payload_from_lut_import_result(_run_cpu_case(resolution, lut_size))
    _assert_bench_guardrail(payload)
    with capsys.disabled():
        print(json.dumps(payload, sort_keys=True))


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
