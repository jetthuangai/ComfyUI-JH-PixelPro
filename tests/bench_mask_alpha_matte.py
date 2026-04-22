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

from core.mask_alpha_matte import alpha_matte_extract  # noqa: E402

SEED = 20260422
WARMUP_ITERS = 1
MEASURE_ITERS = 3
BASELINE_PATH = Path(__file__).with_name("bench_baselines") / "bench_mask_alpha_matte.json"
WRITE_BASELINES = os.environ.get("JH_PIXELPRO_WRITE_BENCH_BASELINES") == "1"


@dataclass(frozen=True, slots=True)
class BenchCase:
    resolution: int
    batch: int = 1

    @property
    def name(self) -> str:
        return f"cpu-{self.resolution}-b{self.batch}-levin_laplacian"


CASES = (BenchCase(64), BenchCase(128))


def _case_payload(case: BenchCase) -> tuple[torch.Tensor, torch.Tensor]:
    size = case.resolution
    trimap = torch.zeros((case.batch, size, size), dtype=torch.float32)
    margin = max(8, size // 4)
    trimap[:, margin : size - margin, margin : size - margin] = 0.5
    core_margin = max(margin + 4, size // 2 - size // 8)
    trimap[:, core_margin : size - core_margin, core_margin : size - core_margin] = 1.0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + size)
    guide = torch.rand((case.batch, size, size, 3), generator=generator)
    return trimap, guide


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
    trimap, guide = _case_payload(case)

    def run_once() -> torch.Tensor:
        return alpha_matte_extract(
            trimap,
            guide,
            epsilon=1e-7,
            window_radius=1,
            lambda_constraint=100.0,
        )

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
        "module": "mask_alpha_matte",
        "scenario": "levin_closed_form_laplacian",
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
def test_bench_mask_alpha_matte(case: BenchCase, capsys: pytest.CaptureFixture[str]) -> None:
    payload = _run_case(case)
    if WRITE_BASELINES:
        _write_baseline(payload)
    else:
        assert any(row.get("case") == payload["case"] for row in _load_baselines())
    with capsys.disabled():
        print(json.dumps(payload, sort_keys=True))


def main() -> int:
    print(json.dumps([_run_case(case) for case in CASES], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
