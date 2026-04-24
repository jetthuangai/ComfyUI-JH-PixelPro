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

from core.mask_alpha_matte import alpha_matte_extract  # noqa: E402
from tests.conftest import (  # noqa: E402
    BENCH_N_RUNS,
    assert_bench_within_threshold,
    build_multi_snapshot_fields,
    stabilize_bench_payload,
)

SEED = 20260422
WARMUP_ITERS = 1
MEASURE_ITERS = 10
BASELINE_PATH = Path(__file__).with_name("bench_baselines") / "bench_mask_alpha_matte.json"
WRITE_BASELINES = os.environ.get("JH_PIXELPRO_WRITE_BENCH_BASELINES") == "1"


@dataclass(frozen=True, slots=True)
class BenchCase:
    resolution: int
    batch: int = 1

    @property
    def name(self) -> str:
        return f"cpu-{self.resolution}-b{self.batch}-levin_laplacian"

    @property
    def measure_iters(self) -> int:
        return 1 if self.resolution >= 512 else MEASURE_ITERS

    @property
    def warmup_iters(self) -> int:
        return 0 if self.resolution >= 512 else WARMUP_ITERS


CASES = (BenchCase(64), BenchCase(128), BenchCase(512))


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
            compute_device="cpu",
        )

    samples: list[float] = []
    with torch.inference_mode():
        for _run in range(BENCH_N_RUNS):
            for _ in range(case.warmup_iters):
                run_once()
            for _ in range(case.measure_iters):
                started = time.perf_counter()
                run_once()
                samples.append((time.perf_counter() - started) * 1000.0)
        if torch.cuda.is_available():
            cuda_trimap = trimap.to("cuda")
            cuda_guide = guide.to("cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            alpha_matte_extract(
                cuda_trimap,
                cuda_guide,
                epsilon=1e-7,
                window_radius=1,
                lambda_constraint=100.0,
                compute_device="cuda",
            )
            torch.cuda.synchronize()
            print(
                f"[GPU bench] size={case.resolution} bs={case.batch} "
                f"elapsed={(time.perf_counter() - started):.4f}s"
            )
    return {
        "case": case.name,
        "module": "mask_alpha_matte",
        "scenario": "levin_closed_form_laplacian",
        "device": "cpu",
        "resolution": case.resolution,
        "batch": case.batch,
        "warmup_iters": case.warmup_iters,
        **build_multi_snapshot_fields(samples, measure_iters=case.measure_iters),
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
def test_bench_mask_alpha_matte(case: BenchCase, capsys: pytest.CaptureFixture[str]) -> None:
    payload = _run_case(case)
    _assert_bench_guardrail(payload)
    with capsys.disabled():
        print(json.dumps(payload, sort_keys=True))


def main() -> int:
    print(json.dumps([_run_case(case) for case in CASES], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
