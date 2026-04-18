"""Shared pytest fixtures for the pack."""

from __future__ import annotations

import pytest
import torch


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
