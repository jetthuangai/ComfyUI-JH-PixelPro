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

from core.blend_modes import apply_blend  # noqa: E402

SEED = 20260421
WARMUP_ITERS = 3
MEASURE_ITERS = 10
REPRESENTATIVE_MODES = ("normal", "multiply", "screen", "overlay", "color_dodge", "difference")
BASELINE_PATH = Path(__file__).with_name("bench_baselines") / "blend_modes.json"
WRITE_BASELINES = os.environ.get("JH_PIXELPRO_WRITE_BENCH_BASELINES") == "1"


@dataclass(frozen=True, slots=True)
class BenchCase:
    mode: str
    opacity: float
    resolution: int = 512
    batch: int = 1

    @property
    def name(self) -> str:
        opacity_key = str(self.opacity).replace(".", "p")
        return f"cpu-{self.resolution}-b{self.batch}-{self.mode}-opacity{opacity_key}"


CASES = tuple(
    BenchCase(mode=mode, opacity=opacity) for mode in REPRESENTATIVE_MODES for opacity in (0.5, 1.0)
)


def _make_image(case: BenchCase, offset: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + offset + case.resolution + case.batch)
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
    base = _make_image(case, 19)
    blend = _make_image(case, 41)

    def run_once() -> torch.Tensor:
        blended = apply_blend(case.mode, base, blend)
        return torch.lerp(base, blended, case.opacity)

    mean_ms, stdev_ms = _measure(run_once)
    return {
        "case": case.name,
        "module": "blend_modes",
        "scenario": "representative_blend",
        "mode": case.mode,
        "device": "cpu",
        "resolution": case.resolution,
        "batch": case.batch,
        "opacity": case.opacity,
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
def test_bench_blend_modes(case: BenchCase, capsys: pytest.CaptureFixture[str]) -> None:
    payload = _run_case(case)
    _assert_baseline_present(payload)
    with capsys.disabled():
        print(json.dumps(payload, sort_keys=True))


def main() -> int:
    print(json.dumps([_run_case(case) for case in CASES], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
