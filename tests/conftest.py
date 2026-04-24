"""Shared pytest fixtures and bench-harness helpers for the pack."""

from __future__ import annotations

import os
import statistics
from collections.abc import Callable
from typing import Any

import pytest
import torch

# ---------------------------------------------------------------------------
# Bench guardrail constants
# ---------------------------------------------------------------------------

# Canonical 10% regression floor — preserved from H-2100 stabilization as the
# lower bound of the variance-aware composite threshold (H-2340 Option C).
BENCH_GUARDRAIL_THRESHOLD = 0.10

# Low-millisecond cases use the median of the current-run sample pool instead
# of the mean when reporting `mean_ms`, to absorb single-outlier spikes on
# short-duration kernels (H-2100 stabilization rule).
BENCH_LOW_MS_MEDIAN_THRESHOLD_MS = 15.0

# Multi-snapshot baseline policy (H-2340 Option C) — each baseline case stores
# raw-sample aggregates collected over N_RUNS outer runs × measure_iters inner
# iterations. Compare-path gates on max(median × 1.10, median + K·σ) so the
# 10% canonical floor is preserved for stable cases while high-CoV cases get
# empirical variance headroom.
BENCH_N_RUNS = 3
BENCH_GUARDRAIL_SIGMA_K = 2.0
BENCH_BASELINE_POLICY = "multi_snapshot_v1"

# Report-only vs strict gate (Batch-13 v1 ships report-only; Batch-13.1 flips
# this to "1" via the CI workflow env after baselines are recalibrated on the
# actual GHA ubuntu-latest enforce-runner — addressing the baseline-runner
# ≠ enforce-runner mismatch surfaced through Batch-13 E-2 → E-8 escalation).
BENCH_STRICT_GATE = os.environ.get("JH_PIXELPRO_BENCH_STRICT_GATE", "0") == "1"


def pytest_configure(config: pytest.Config) -> None:
    """Register project-local markers used by benchmark smoke tests."""
    config.addinivalue_line(
        "markers",
        "bench_guardrail: CPU benchmark guardrail smoke test "
        "(multi-snapshot baseline, variance-aware threshold)",
    )


@pytest.fixture(scope="session")
def device() -> torch.device:
    """CUDA if available, otherwise CPU (for CI runners without a GPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def rng() -> torch.Generator:
    """Deterministic RNG for reproducible tests."""
    g = torch.Generator()
    g.manual_seed(20260417)
    return g


# ---------------------------------------------------------------------------
# Multi-snapshot bench harness helpers
# ---------------------------------------------------------------------------


def collect_multi_snapshot_samples(
    sample_once: Callable[[], float],
    *,
    measure_iters: int,
    n_runs: int = BENCH_N_RUNS,
    warmup: int = 0,
) -> list[float]:
    """Collect ``n_runs × measure_iters`` raw timings (ms) from ``sample_once``.

    ``sample_once`` must be a zero-argument callable that runs one measured
    iteration and returns the elapsed time in milliseconds. Warmup iterations
    per outer run are discarded (not appended to the pool).
    """
    samples: list[float] = []
    for _run in range(n_runs):
        for _ in range(warmup):
            sample_once()
        for _ in range(measure_iters):
            samples.append(sample_once())
    return samples


def build_multi_snapshot_fields(
    samples_ms: list[float],
    *,
    measure_iters: int,
    n_runs: int = BENCH_N_RUNS,
) -> dict[str, Any]:
    """Compute canonical multi-snapshot aggregate fields from a pooled sample list."""
    if not samples_ms:
        return {
            "mean_ms": None,
            "median_ms": None,
            "stdev_ms": 0.0,
            "n_samples": 0,
            "n_runs": n_runs,
            "measure_iters": measure_iters,
            "policy": BENCH_BASELINE_POLICY,
        }

    stdev = statistics.stdev(samples_ms) if len(samples_ms) >= 2 else 0.0
    return {
        "mean_ms": round(statistics.mean(samples_ms), 4),
        "median_ms": round(statistics.median(samples_ms), 4),
        "stdev_ms": round(stdev, 4),
        "n_samples": len(samples_ms),
        "n_runs": n_runs,
        "measure_iters": measure_iters,
        "policy": BENCH_BASELINE_POLICY,
    }


def stabilize_bench_payload(
    payload: dict[str, Any],
    baseline_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the H-2100 median-for-low-ms rule on the current-run payload.

    The payload's ``mean_ms`` is swapped for its ``median_ms`` when the
    reference baseline is a low-millisecond case, absorbing single-outlier
    spikes. The reference uses the baseline's ``median_ms`` under the
    multi-snapshot schema, falling back to ``mean_ms`` for legacy baselines.
    """
    stabilized = dict(payload)
    mean_ms = stabilized.get("mean_ms")
    median_ms = stabilized.get("median_ms")
    if mean_ms is None or median_ms is None:
        return stabilized

    reference_ms = float(mean_ms)
    if baseline_row is not None:
        ref = baseline_row.get("median_ms")
        if ref is None:
            ref = baseline_row.get("mean_ms")
        if ref is not None:
            reference_ms = float(ref)

    if reference_ms < BENCH_LOW_MS_MEDIAN_THRESHOLD_MS:
        stabilized["mean_ms"] = float(median_ms)
        stabilized["mean_policy"] = "median_for_low_ms"
    else:
        stabilized.setdefault("mean_policy", "mean")
    return stabilized


def assert_bench_within_threshold(
    payload: dict[str, Any],
    baseline_row: dict[str, Any],
) -> None:
    """Compare ``payload["mean_ms"]`` against baseline, report observability, optionally gate.

    Multi-snapshot (``policy == "multi_snapshot_v1"``) uses the composite
    bound ``max(baseline_median × (1 + THRESHOLD), baseline_median + K·σ)``
    so the canonical 10% floor is preserved while variance headroom absorbs
    process-level jitter. Legacy baselines without a ``policy`` field fall
    back to ``baseline_mean × (1 + THRESHOLD)`` for backward compatibility.

    A single ``[bench-report] <case>: OK|OVER current=…ms threshold=…ms
    policy=…`` line is always printed for observability (visible in pytest
    stdout + GHA workflow log so artifact consumers can grep drift cases).

    The ``raise AssertionError`` on threshold breach is gated behind the
    module-level ``BENCH_STRICT_GATE`` flag (env var
    ``JH_PIXELPRO_BENCH_STRICT_GATE``, default ``"0"`` = report-only).
    Batch-13 v1 ships report-only; Batch-13.1 flips to ``"1"`` after
    baselines are recalibrated on the actual GHA enforce-runner.
    """
    current_ms = payload.get("mean_ms")
    if current_ms is None or baseline_row.get("mean_ms") is None:
        return

    current_ms = float(current_ms)
    case = payload.get("case", "?")
    policy = baseline_row.get("policy", "single_snapshot")

    if policy == BENCH_BASELINE_POLICY:
        baseline_median = float(baseline_row["median_ms"])
        baseline_stdev = float(baseline_row.get("stdev_ms", 0.0))
        threshold_pct = baseline_median * (1.0 + BENCH_GUARDRAIL_THRESHOLD)
        threshold_sigma = baseline_median + BENCH_GUARDRAIL_SIGMA_K * baseline_stdev
        threshold_ms = max(threshold_pct, threshold_sigma)
        breach_detail = (
            f"baseline_median={baseline_median:.4f}ms, "
            f"stdev={baseline_stdev:.4f}ms, "
            f"pct_bound={threshold_pct:.4f}ms, "
            f"sigma_bound={threshold_sigma:.4f}ms, "
            f"policy={policy}"
        )
    else:
        baseline_mean = float(baseline_row["mean_ms"])
        threshold_ms = baseline_mean * (1.0 + BENCH_GUARDRAIL_THRESHOLD)
        breach_detail = (
            f"legacy single_snapshot × {1 + BENCH_GUARDRAIL_THRESHOLD:.2f}, "
            f"baseline_mean={baseline_mean:.4f}ms"
        )

    within_threshold = current_ms <= threshold_ms
    status = "OK" if within_threshold else "OVER"
    # Always emit observability line so JH/Cowork can grep [bench-report] in
    # GHA workflow log to identify drift cases per CI run.
    print(
        f"[bench-report] {case}: {status} "
        f"current={current_ms:.4f}ms threshold={threshold_ms:.4f}ms policy={policy}"
    )

    if BENCH_STRICT_GATE and not within_threshold:
        raise AssertionError(
            f"{case} bench regression: "
            f"current {current_ms:.4f}ms > threshold {threshold_ms:.4f}ms "
            f"({breach_detail}, BENCH_STRICT_GATE=1)"
        )
    # BENCH_STRICT_GATE=0 (Batch-13 v1 default): no raise, report-only.
