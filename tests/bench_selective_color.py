from __future__ import annotations

import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.selective_color import (  # noqa: E402
    apply_hue_sat_shift,
    hue_range_mask,
    saturation_range_mask,
)

SEED = 20260421
WARMUP_ITERS = 3
MEASURE_ITERS = 10
BASELINE_PATH = Path(__file__).with_name("bench_baselines") / "selective_color.json"
WRITE_BASELINES = os.environ.get("JH_PIXELPRO_WRITE_BENCH_BASELINES") == "1"


@dataclass(frozen=True, slots=True)
class BenchCase:
    scenario: str
    resolution: int
    batch: int

    @property
    def name(self) -> str:
        return f"cpu-{self.resolution}-b{self.batch}-{self.scenario}"


CASES = (
    BenchCase("hue_mask", 512, 1),
    BenchCase("hue_shift", 512, 1),
    BenchCase("sat_mask", 512, 1),
    BenchCase("hue_mask", 1024, 1),
    BenchCase("hue_shift", 1024, 1),
    BenchCase("sat_mask", 1024, 1),
)


def _make_image(case: BenchCase) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + case.resolution + case.batch)
    return torch.rand((case.batch, case.resolution, case.resolution, 3), generator=generator)


def _measure(fn: Callable[[], torch.Tensor]) -> tuple[float, float]:
    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            fn()

        samples: list[float] = []
        for _ in range(MEASURE_ITERS):
            started = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.mean(samples), statistics.stdev(samples)


def _run_case(case: BenchCase) -> dict[str, float | int | str]:
    torch.manual_seed(SEED)
    image = _make_image(case)
    hue_mask = hue_range_mask(image, 30.0, 45.0, device="cpu")

    if case.scenario == "hue_mask":

        def fn() -> torch.Tensor:
            return hue_range_mask(image, 30.0, 45.0, device="cpu")

    elif case.scenario == "hue_shift":

        def fn() -> torch.Tensor:
            return apply_hue_sat_shift(
                image,
                hue_mask,
                hue_shift=18.0,
                sat_mult=1.12,
                sat_add=0.03,
                device="cpu",
            )

    elif case.scenario == "sat_mask":

        def fn() -> torch.Tensor:
            return saturation_range_mask(image, 0.2, 0.85, feather=0.1, device="cpu")

    else:
        raise ValueError(f"unknown scenario: {case.scenario}")

    mean_ms, stdev_ms = _measure(fn)
    return {
        "case": case.name,
        "module": "selective_color",
        "scenario": case.scenario,
        "device": "cpu",
        "resolution": case.resolution,
        "batch": case.batch,
        "warmup_iters": WARMUP_ITERS,
        "measure_iters": MEASURE_ITERS,
        "mean_ms": round(mean_ms, 4),
        "stdev_ms": round(stdev_ms, 4),
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
    BASELINE_PATH.write_text(json.dumps(ordered, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assert_baseline_present(payload: dict[str, object]) -> None:
    if WRITE_BASELINES:
        _write_baseline(payload)
        return
    baselines = _load_baselines()
    assert any(row.get("case") == payload["case"] for row in baselines)


@pytest.mark.bench_guardrail
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_bench_selective_color(case: BenchCase, capsys: pytest.CaptureFixture[str]) -> None:
    payload = _run_case(case)
    _assert_baseline_present(payload)
    with capsys.disabled():
        print(json.dumps(payload, sort_keys=True))


def main() -> int:
    print(json.dumps([_run_case(case) for case in CASES], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
