from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.color_lab import (  # noqa: E402
    BASIC_KEYS,
    GRADE_REGIONS,
    HUE_ANCHORS,
    apply_colorlab_pipeline,
)
from tests.conftest import (  # noqa: E402
    BENCH_N_RUNS,
    assert_bench_within_threshold,
    build_multi_snapshot_fields,
    stabilize_bench_payload,
)

SEED = 20260421
WARMUP_ITERS = 3
MEASURE_ITERS = 10
BASELINE_PATH = Path(__file__).with_name("bench_baselines") / "color_lab.json"
WRITE_BASELINES = os.environ.get("JH_PIXELPRO_WRITE_BENCH_BASELINES") == "1"


@dataclass(frozen=True, slots=True)
class BenchCase:
    preset: str
    resolution: int
    batch: int = 1

    @property
    def name(self) -> str:
        return f"cpu-{self.resolution}-b{self.batch}-{self.preset}"


CASES = (
    BenchCase("identity", 512),
    BenchCase("full_stack", 512),
    BenchCase("identity", 1024),
    BenchCase("full_stack", 1024),
)


def _make_image(case: BenchCase) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + case.resolution + case.batch)
    return torch.rand((case.batch, case.resolution, case.resolution, 3), generator=generator)


def _identity_params() -> dict[str, float | bool]:
    params: dict[str, float | bool] = {key: 0.0 for key in BASIC_KEYS}
    for color in HUE_ANCHORS:
        params[f"hsl_{color}_hue"] = 0.0
        params[f"hsl_{color}_sat"] = 0.0
        params[f"hsl_{color}_lum"] = 0.0
    for region in GRADE_REGIONS:
        params[f"grade_{region}_hue"] = 0.0
        params[f"grade_{region}_sat"] = 0.0
        params[f"grade_{region}_lum"] = 0.0
        params[f"grade_{region}_bal"] = 0.0
    params["gray_enable"] = False
    for color in ("red", "orange", "yellow", "green", "aqua", "blue", "purple"):
        params[f"gray_{color}"] = 0.0
    return params


def _full_stack_params() -> dict[str, float | bool]:
    params = _identity_params()
    params.update(
        {
            "basic_exposure": 0.12,
            "basic_contrast": 14.0,
            "basic_highlights": -18.0,
            "basic_shadows": 16.0,
            "basic_whites": 7.0,
            "basic_blacks": -8.0,
            "basic_texture": 12.0,
            "basic_clarity": 9.0,
            "basic_dehaze": 5.0,
            "basic_vibrance": 18.0,
            "basic_saturation": 6.0,
            "grade_shadow_hue": 218.0,
            "grade_shadow_sat": 12.0,
            "grade_shadow_lum": -4.0,
            "grade_shadow_bal": -18.0,
            "grade_mid_hue": 36.0,
            "grade_mid_sat": 5.0,
            "grade_mid_lum": 3.0,
            "grade_mid_bal": 0.0,
            "grade_highlight_hue": 48.0,
            "grade_highlight_sat": 10.0,
            "grade_highlight_lum": 5.0,
            "grade_highlight_bal": 12.0,
        }
    )
    for color in HUE_ANCHORS:
        params[f"hsl_{color}_hue"] = 2.0
        params[f"hsl_{color}_sat"] = 4.0
        params[f"hsl_{color}_lum"] = 1.5
    return params


def _params_for_preset(preset: str) -> dict[str, float | bool]:
    if preset == "identity":
        return _identity_params()
    if preset == "full_stack":
        return _full_stack_params()
    raise ValueError(f"unknown preset: {preset}")


def _measure(fn: Callable[[], torch.Tensor]) -> list[float]:
    samples: list[float] = []
    with torch.inference_mode():
        for _run in range(BENCH_N_RUNS):
            for _ in range(WARMUP_ITERS):
                fn()
            for _ in range(MEASURE_ITERS):
                started = time.perf_counter()
                fn()
                samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def _run_case(case: BenchCase) -> dict[str, float | int | str]:
    torch.manual_seed(SEED)
    image = _make_image(case)
    params = _params_for_preset(case.preset)
    samples = _measure(lambda: apply_colorlab_pipeline(image, params))
    return {
        "case": case.name,
        "module": "color_lab",
        "scenario": case.preset,
        "device": "cpu",
        "resolution": case.resolution,
        "batch": case.batch,
        "warmup_iters": WARMUP_ITERS,
        **build_multi_snapshot_fields(samples, measure_iters=MEASURE_ITERS),
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
def test_bench_color_lab(case: BenchCase, capsys: pytest.CaptureFixture[str]) -> None:
    payload = _run_case(case)
    _assert_baseline_present(payload)
    with capsys.disabled():
        print(json.dumps(payload, sort_keys=True))


def main() -> int:
    print(json.dumps([_run_case(case) for case in CASES], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
