from __future__ import annotations

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

from core.mask_combine import combine_masks  # noqa: E402
from tests.conftest import (  # noqa: E402
    BENCH_N_RUNS,
    assert_bench_within_threshold,
    build_multi_snapshot_fields,
    stabilize_bench_payload,
)

SEED = 20260421
WARMUP_ITERS = 3
MEASURE_ITERS = 10
BASELINE_PATH = Path(__file__).with_name("bench_baselines") / "bench_mask_combine.json"
WRITE_BASELINES = os.environ.get("JH_PIXELPRO_WRITE_BENCH_BASELINES") == "1"


@dataclass(frozen=True, slots=True)
class BenchCase:
    resolution: int
    batch: int
    blend_mode: str

    @property
    def name(self) -> str:
        return f"cpu-{self.resolution}-b{self.batch}-union-{self.blend_mode}"


CASES = (
    BenchCase(512, 1, "hard"),
    BenchCase(1024, 1, "soft_feather"),
)


def _make_masks(case: BenchCase) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + case.resolution)
    mask_a = (
        torch.rand((case.batch, case.resolution, case.resolution), generator=generator) > 0.5
    ).float()
    mask_b = (
        torch.rand((case.batch, case.resolution, case.resolution), generator=generator) > 0.5
    ).float()
    return mask_a, mask_b


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
    mask_a, mask_b = _make_masks(case)

    def run_once() -> torch.Tensor:
        return combine_masks(
            mask_a,
            mask_b,
            operation="union",
            blend_mode=case.blend_mode,
            opacity=1.0,
            feather_sigma=1.25,
        )

    samples: list[float] = []
    with torch.inference_mode():
        for _run in range(BENCH_N_RUNS):
            for _ in range(WARMUP_ITERS):
                run_once()
            for _ in range(MEASURE_ITERS):
                started = time.perf_counter()
                run_once()
                samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "case": case.name,
        "module": "mask_combine",
        "scenario": f"union_{case.blend_mode}",
        "device": "cpu",
        "resolution": case.resolution,
        "batch": case.batch,
        "warmup_iters": WARMUP_ITERS,
        **build_multi_snapshot_fields(samples, measure_iters=MEASURE_ITERS),
    }


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
def test_bench_mask_combine(case: BenchCase, capsys: pytest.CaptureFixture[str]) -> None:
    payload = _run_case(case)
    _assert_bench_guardrail(payload)
    with capsys.disabled():
        print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    print(json.dumps([_run_case(case) for case in CASES], indent=2, sort_keys=True))
