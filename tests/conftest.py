"""Shared pytest fixtures for the pack."""

from __future__ import annotations

import pytest
import torch

BENCH_GUARDRAIL_THRESHOLD = 0.10


def pytest_configure(config: pytest.Config) -> None:
    """Register project-local markers used by benchmark smoke tests."""
    config.addinivalue_line(
        "markers",
        "bench_guardrail: CPU benchmark guardrail smoke test (10% threshold baseline)",
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
