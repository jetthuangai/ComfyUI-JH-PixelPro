from __future__ import annotations

import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.mask_morphology import mask_morphology  # noqa: E402

SEED = 20260421
WARMUP_ITERS = 3
MEASURE_ITERS = 10
BASELINE_PATH = Path(__file__).with_name("bench_baselines") / "bench_mask_morphology.json"
WRITE_BASELINES = os.environ.get("JH_PIXELPRO_WRITE_BENCH_BASELINES") == "1"


@dataclass(frozen=True, slots=True)
class BenchCase:
    resolution: int
    batch: int
    operation: str

    @property
    def name(self) -> str:
        return f"cpu-{self.resolution}-b{self.batch}-{self.operation}"


CASES = (
    BenchCase(512, 1, "close"),
    BenchCase(1024, 1, "gradient"),
)


def _make_mask(case: BenchCase) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + case.resolution)
    return (
        torch.rand((case.batch, case.resolution, case.resolution), generator=generator) > 0.5
    ).float()


def _load_baselines() -> list[dict[str, object]]:
    if not BASELINE_PATH.exists():
        return []
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _write_baseline(payload: dict[str, object]) -> None:
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = {str(row["case"]): row for row in _load_baselines()}
    rows[str(payload["case"])] = payload
    BASELINE_PATH.write_text(
        json.dumps([rows[key] for key in sorted(rows)], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_case(case: BenchCase) -> dict[str, float | int | str]:
    mask = _make_mask(case)

    def run_once() -> torch.Tensor:
        return mask_morphology(mask, operation=case.operation, radius=3, iterations=1)

    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            run_once()
        samples = []
        for _ in range(MEASURE_ITERS):
            started = time.perf_counter()
            run_once()
            samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "case": case.name,
        "module": "mask_morphology",
        "scenario": f"{case.operation}_ellipse_r3",
        "device": "cpu",
        "resolution": case.resolution,
        "batch": case.batch,
        "warmup_iters": WARMUP_ITERS,
        "measure_iters": MEASURE_ITERS,
        "mean_ms": round(statistics.mean(samples), 4),
        "stdev_ms": round(statistics.stdev(samples), 4),
    }


@pytest.mark.bench_guardrail
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_bench_mask_morphology(case: BenchCase, capsys: pytest.CaptureFixture[str]) -> None:
    payload = _run_case(case)
    if WRITE_BASELINES:
        _write_baseline(payload)
    else:
        assert any(row.get("case") == payload["case"] for row in _load_baselines())
    with capsys.disabled():
        print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    print(json.dumps([_run_case(case) for case in CASES], indent=2, sort_keys=True))
