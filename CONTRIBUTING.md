# Contributing to ComfyUI-JH-PixelPro

Thanks for considering a contribution! This guide covers the development workflow, test suites, commit style, and CI expectations for the pack.

## Development environment

Clone the pack into your ComfyUI custom nodes directory and install in editable mode with the dev extras:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jetthuangai/ComfyUI-JH-PixelPro.git
cd ComfyUI-JH-PixelPro
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```

If the `dev` extra is unavailable on your environment, fall back to plain editable install and add the tooling explicitly:

```bash
pip install -e .
pip install pytest ruff
```

## Running tests

Unit tests (fast, skip CUDA-only cases on CPU runners):

```bash
pytest tests/ -v --ignore-glob='tests/bench_*.py'
```

Benchmark guardrail suite (CPU baselines):

```bash
pytest tests/bench_*.py -v
```

The bench suite is gated by a **multi-snapshot variance-aware threshold** (see below). On a CPU runner, expect the full bench suite to take roughly 18–20 minutes per pass (N=1) and up to 55 minutes when regenerating baselines (N=3).

## Baseline regeneration

Every bench case is compared against a committed baseline JSON in `tests/bench_baselines/`. When the observed runtime drifts past the variance-aware threshold, the bench guardrail fails and CI goes red.

To regenerate the baselines (for example, after a legitimate perf improvement or new hardware context):

```bash
JH_PIXELPRO_WRITE_BENCH_BASELINES=1 pytest tests/bench_*.py -v
```

This sets `WRITE_BASELINES=1`, which short-circuits the threshold comparison and writes a fresh snapshot of `mean_ms`, `median_ms`, `stdev_ms`, and sample counts across `BENCH_N_RUNS` outer runs × `measure_iters` inner iterations per case.

Regen protocol:

1. Confirm the perf delta is intentional. Do not silence a regression with a regen.
2. Run the regen on a clean, idle machine (close background apps; close Chrome; pause sync clients). Non-isolated CPU jitter inflates stdev and weakens the guardrail.
3. Tag the regen commit with a `[bench-baseline-regen]` suffix and link the reason in the commit body (for example, `perf(batch-14 M1): vectorized tone_curve LUT apply [bench-baseline-regen]`).
4. Request a review from Cowork before merging.

## Commit style

This pack follows [Conventional Commits](https://www.conventionalcommits.org/). Allowed types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `perf`, `ci`, `style`, `build`, `revert`. Scopes map to the batch or milestone under development (`feat(batch-13 M1): …`, `docs(batch-13): …`, `fix(n-29): …`).

Short line rules:

- Subject ≤ 72 characters, imperative mood, no trailing period.
- Body wraps at 100 characters with a blank line between subject and body.
- Footers may carry `Co-Authored-By:` lines for multi-agent sessions.

## Pull request checklist

Before pushing or opening a pull request:

```bash
pytest tests/ -q --ignore-glob='tests/bench_*.py'
pytest tests/bench_*.py -v
ruff check .
ruff format --check .
```

All four must pass locally. CI will reject a PR that fails any of these.

## CI bench guardrail (multi-snapshot variance-aware, report-only v1)

`.github/workflows/ci.yml` runs on every pull request to `main` and every push to `main`. It performs the Gate 6.0 triple-command (ruff check + ruff format --check + unit pytest) and then runs the bench guardrail suite.

The bench guardrail ships in **report-only mode** for the Batch-13 v1 release. Each run collects multi-snapshot variance data and prints per-case comparison lines like `[bench-report] cpu-512-b1-normal-opacity0p5: OK current=0.18ms threshold=0.20ms policy=multi_snapshot_v1` to the workflow log, but does NOT fail the build on threshold breach. The `raise AssertionError` path is gated behind the env flag `JH_PIXELPRO_BENCH_STRICT_GATE` (default `"0"`). Strict gate enablement is deferred to a Batch-13.1 follow-up that will recapture baselines from the actual GHA ubuntu-latest runner (the Batch-13 E-2 → E-8 escalation chain proved that baselines captured on dev machines are statistically incompatible with CI enforce-runner profile due to thermal/cache/scheduler differences). Locally, contributors can simulate the future strict gate behavior with `JH_PIXELPRO_BENCH_STRICT_GATE=1 pytest tests/bench_*.py -v` for diagnostic purposes.

Each baseline row carries `policy: "multi_snapshot_v1"` plus:

| Field | Meaning |
|---|---|
| `median_ms` | Median across `n_samples = n_runs × measure_iters` raw timings. The canonical baseline point used by the compare path. |
| `stdev_ms` | Sample standard deviation (ddof=1) across the same pool. |
| `mean_ms` | Arithmetic mean across the pool (back-compat, and the value the low-ms median-for-low-ms stabilization overrides). |
| `n_samples` | Total sample count (= `n_runs × measure_iters`). |
| `n_runs` | Outer repetition count (`BENCH_N_RUNS`, default 3). |
| `measure_iters` | Inner sample count per run. |

The compare path gates each current-run `mean_ms` against:

```
threshold = max(baseline_median × 1.10, baseline_median + 2 × baseline_stdev)
```

This preserves the canonical 10% regression floor for stable cases while absorbing process-level jitter for high-CoV kernels on non-isolated CPU runners. Baselines written under the legacy single-snapshot policy (no `policy` key) fall back to `baseline_mean × 1.10`.

Tunable constants live in `tests/conftest.py`:

- `BENCH_GUARDRAIL_THRESHOLD = 0.10` — canonical 10% floor.
- `BENCH_GUARDRAIL_SIGMA_K = 2.0` — variance-band multiplier.
- `BENCH_N_RUNS = 3` — outer repetition count.
- `BENCH_LOW_MS_MEDIAN_THRESHOLD_MS = 15.0` — swap mean for median on short-duration kernels.

## Documentation

For local docs preview:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Before pushing doc changes, run a strict build:

```bash
mkdocs build --strict
```

The production deploy path lives in `.github/workflows/docs.yml`, which publishes the site to GitHub Pages on every push to `main`.

## Architecture and ownership

Pack authorship is split across three agents (Cowork architect, Codex core math, Claude Code integration). The detailed ownership matrix lives in the agent hub at `.agent-hub/00_charter/agent-coordination.md` (repo-local, not shipped).

In short:

- `core/**` and `tests/bench_*.py` are Codex-owned.
- `nodes/**`, `workflows/**`, `__init__.py`, and `tests/test_*_node.py` are Claude Code-owned.
- `.agent-hub/10_plan/`, `/20_specs/`, `/00_charter/` are Cowork-owned.

External contributors: target `custom_nodes/ComfyUI-JH-PixelPro/` as a single pack and open a PR against `main`. Comment in the PR if you need a regen of any baseline; a pack maintainer will run it from a clean machine and push the updated JSON alongside your change.
