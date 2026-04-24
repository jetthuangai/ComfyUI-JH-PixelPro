from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import kornia
import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn.functional import interpolate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import face_detect  # noqa: E402
from tests.conftest import (  # noqa: E402
    BENCH_N_RUNS,
    assert_bench_within_threshold,
    build_multi_snapshot_fields,
    stabilize_bench_payload,
)

SEED = 20260419
BASELINE_PATH = Path(__file__).with_name("bench_baselines") / "bench_face_detect.json"
WRITE_BASELINES = os.environ.get("JH_PIXELPRO_WRITE_BENCH_BASELINES") == "1"


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
    stdev_ms: float | None = None
    n_samples: int | None = None
    n_runs: int | None = None
    measure_iters: int | None = None
    warmup_iters: int | None = None
    policy: str | None = None


CASES = [
    BenchCase("cpu-512", 512, 2, 15),
    BenchCase("cpu-1024", 1024, 2, 15),
    BenchCase("cpu-2048", 2048, 2, 15),
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

    samples_ms: list[float] = []
    with torch.inference_mode():
        for _run in range(BENCH_N_RUNS):
            for _ in range(case.warmup_iters):
                run_once()
            for _ in range(case.measure_iters):
                samples_ms.append(measure_once(run_once))

    multi = build_multi_snapshot_fields(samples_ms, measure_iters=case.measure_iters)
    del image
    gc.collect()
    return BenchResult(
        name=case.name,
        long_side=case.long_side,
        status="EXECUTED",
        reason="",
        min_ms=min(samples_ms) if samples_ms else None,
        median_ms=multi["median_ms"],
        mean_ms=multi["mean_ms"],
        stdev_ms=multi["stdev_ms"],
        n_samples=multi["n_samples"],
        n_runs=multi["n_runs"],
        measure_iters=multi["measure_iters"],
        warmup_iters=case.warmup_iters,
        policy=multi["policy"],
    )


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


@pytest.mark.bench_guardrail
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_bench_face_detect(case: BenchCase, capsys: pytest.CaptureFixture[str]) -> None:
    payload = _payload_from_result(_run_case(case))
    _assert_bench_guardrail(payload)
    with capsys.disabled():
        print(json.dumps(payload, sort_keys=True))


def main() -> int:
    torch.manual_seed(SEED)
    results = [_run_case(case) for case in CASES]
    print(render_markdown(_hardware_info(), results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
