from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.mask_alpha_matte import alpha_matte_extract  # noqa: E402

SEED = 20260421
WARMUP_ITERS = 1
MEASURE_ITERS = 3
BASELINE_PATH = Path(__file__).with_name("bench_baselines") / "bench_mask_alpha_matte.json"
WRITE_BASELINES = os.environ.get("JH_PIXELPRO_WRITE_BENCH_BASELINES") == "1"


def _case_payload() -> tuple[torch.Tensor, torch.Tensor]:
    size = 128
    trimap = torch.zeros((1, size, size), dtype=torch.float32)
    trimap[:, 36:92, 36:92] = 0.5
    trimap[:, 48:80, 48:80] = 1.0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    guide = torch.rand((1, size, size, 3), generator=generator)
    return trimap, guide


def _load_baselines() -> list[dict[str, object]]:
    if not BASELINE_PATH.exists():
        return []
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _write_baseline(payload: dict[str, object]) -> None:
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(
        json.dumps([payload], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_case() -> dict[str, float | int | str]:
    trimap, guide = _case_payload()

    def run_once() -> torch.Tensor:
        return alpha_matte_extract(trimap, guide, epsilon=1e-4, window_radius=1)

    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            run_once()
        samples = []
        for _ in range(MEASURE_ITERS):
            started = time.perf_counter()
            run_once()
            samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "case": "cpu-128-b1-sparse_unknown_band",
        "module": "mask_alpha_matte",
        "scenario": "sparse_alpha_matte",
        "device": "cpu",
        "resolution": 128,
        "batch": 1,
        "warmup_iters": WARMUP_ITERS,
        "measure_iters": MEASURE_ITERS,
        "mean_ms": round(statistics.mean(samples), 4),
        "stdev_ms": round(statistics.stdev(samples), 4),
    }


@pytest.mark.bench_guardrail
def test_bench_mask_alpha_matte(capsys: pytest.CaptureFixture[str]) -> None:
    payload = _run_case()
    if WRITE_BASELINES:
        _write_baseline(payload)
    else:
        assert any(row.get("case") == payload["case"] for row in _load_baselines())
    with capsys.disabled():
        print(json.dumps(payload, sort_keys=True))


def main() -> int:
    print(json.dumps([_run_case()], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
