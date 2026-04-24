from __future__ import annotations

import gc
import json
import os
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
from core.lut_preset import list_presets, load_preset  # noqa: E402
from tests.conftest import (  # noqa: E402
    BENCH_N_RUNS,
    assert_bench_within_threshold,
    build_multi_snapshot_fields,
    stabilize_bench_payload,
)

SEED = 20260424
BASELINE_PATH = Path(__file__).with_name("bench_baselines") / "bench_lut_preset.json"
WRITE_BASELINES = os.environ.get("JH_PIXELPRO_WRITE_BENCH_BASELINES") == "1"
MEASURE_ITERS = 2
WARMUP_ITERS = 1


@dataclass(slots=True)
class BenchCase:
    preset: str
    resolution: int

    @property
    def name(self) -> str:
        return f"cpu-{self.resolution}-{self.preset}"


@dataclass(slots=True)
class BenchResult:
    name: str
    preset: str
    resolution: int
    status: str
    mean_ms: float | None
    median_ms: float | None
    stdev_ms: float | None
    n_samples: int | None
    n_runs: int | None
    measure_iters: int | None
    warmup_iters: int | None
    policy: str | None


def _make_image(resolution: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + resolution)
    return torch.rand((1, resolution, resolution, 3), generator=generator, dtype=torch.float32)


def _measure_once(fn) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000.0


def _run_case(case: BenchCase) -> BenchResult:
    image = _make_image(case.resolution)
    parsed = load_preset(case.preset)
    lut = parsed["lut"]

    def run_once(image_bhwc: torch.Tensor = image, lut_grid: torch.Tensor = lut) -> torch.Tensor:
        return apply_lut_3d(image_bhwc, lut_grid)

    samples_ms: list[float] = []
    with torch.inference_mode():
        for _run in range(BENCH_N_RUNS):
            for _ in range(WARMUP_ITERS):
                run_once()
            for _ in range(MEASURE_ITERS):
                samples_ms.append(_measure_once(run_once))

    multi = build_multi_snapshot_fields(samples_ms, measure_iters=MEASURE_ITERS)
    del image, lut
    gc.collect()
    return BenchResult(
        name=case.name,
        preset=case.preset,
        resolution=case.resolution,
        status="EXECUTED",
        mean_ms=multi["mean_ms"],
        median_ms=multi["median_ms"],
        stdev_ms=multi["stdev_ms"],
        n_samples=multi["n_samples"],
        n_runs=multi["n_runs"],
        measure_iters=multi["measure_iters"],
        warmup_iters=WARMUP_ITERS,
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


def _payload_from_result(result: object) -> dict[str, object]:
    payload = asdict(result) if is_dataclass(result) else dict(result)
    if "case" not in payload and "name" in payload:
        payload["case"] = payload["name"]
    payload.setdefault("module", Path(__file__).stem.removeprefix("bench_"))
    payload.setdefault("device", "cpu")
    payload["scenario"] = "preset_lut_apply"
    return payload


def _assert_bench_guardrail(payload: dict[str, object]) -> None:
    if WRITE_BASELINES:
        _write_baseline(stabilize_bench_payload(payload))
        return

    baseline_row = next(
        (row for row in _load_baselines() if row["case"] == payload["case"]),
        None,
    )
    assert baseline_row is not None, f"baseline missing for {payload['case']}"
    stabilized = stabilize_bench_payload(payload, baseline_row)
    assert_bench_within_threshold(stabilized, baseline_row)


CASES = [BenchCase(preset, resolution) for preset in list_presets() for resolution in (256, 512)]


@pytest.mark.bench_guardrail
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_bench_lut_preset(case: BenchCase, capsys: pytest.CaptureFixture[str]) -> None:
    payload = _payload_from_result(_run_case(case))
    _assert_bench_guardrail(payload)
    with capsys.disabled():
        print(json.dumps(payload, sort_keys=True))


def main() -> int:
    for case in CASES:
        print(json.dumps(_payload_from_result(_run_case(case)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
