"""pytest fixtures chung cho pack."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    """CUDA nếu có, fallback CPU cho CI runner không có GPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def rng() -> torch.Generator:
    """Deterministic RNG để test reproducible."""
    g = torch.Generator()
    g.manual_seed(20260417)
    return g
