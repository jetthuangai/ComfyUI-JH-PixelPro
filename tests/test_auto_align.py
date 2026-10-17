from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from kornia.geometry.transform import warp_perspective

from core import auto_align as auto_align_module
from core.auto_align import auto_align, loftr_weights_path

_SAMPLE = Path(__file__).resolve().parents[1] / "workflows" / "sample_portrait.jpg"
_SIDE = 384


def _reference(side: int = _SIDE) -> torch.Tensor:
    bgr = cv2.imread(str(_SAMPLE))
    assert bgr is not None, "sample_portrait.jpg missing"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    scale = side / max(rgb.shape[:2])
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return torch.from_numpy(small).float().div(255.0)[None]


def _transform(ref: torch.Tensor, angle: float = 5.0, scale: float = 0.95, shift=(20.0, 12.0)):
    """Return (image, G) where image = ref warped by G (ref -> image coordinates)."""

    height, width = ref.shape[1:3]
    affine = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    affine[:, 2] += shift
    g = np.eye(3)
    g[:2] = affine
    image = warp_perspective(
        ref.permute(0, 3, 1, 2),
        torch.tensor(g, dtype=torch.float32)[None],
        (height, width),
        align_corners=True,
    ).permute(0, 2, 3, 1)
    return image.contiguous(), g


def _corners(width: int, height: int) -> np.ndarray:
    return np.array(
        [[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1], [0, height - 1, 1]]
    ).T


def _project(matrix, points: np.ndarray) -> np.ndarray:
    out = np.array(matrix) @ points
    return out[:2] / out[2]


def _corner_error(matrix, g_ref_to_image, shape) -> float:
    pts = _corners(shape[1], shape[0])
    return float(np.abs(_project(matrix, pts) - _project(np.linalg.inv(g_ref_to_image), pts)).max())


@pytest.mark.parametrize("model", ["homography", "affine"])
def test_recovers_known_transform(model: str) -> None:
    ref = _reference()
    image, g = _transform(ref)
    aligned, mask, matrices, inliers = auto_align(image, ref, model=model)
    assert aligned.shape == ref.shape and aligned.dtype == torch.float32
    assert mask.shape == ref.shape[:3]
    assert len(matrices) == 1 and inliers[0] >= 12
    assert _corner_error(matrices[0], g, ref.shape[1:3]) < 2.0
    valid = mask[0] > 0.99
    assert valid.float().mean() > 0.7
    assert (aligned - ref).abs().mean(-1)[0][valid].mean().item() < 0.03


def test_identity_case() -> None:
    ref = _reference(256)
    aligned, mask, matrices, _ = auto_align(ref, ref)
    assert np.allclose(np.array(matrices[0]), np.eye(3), atol=0.5)
    assert (aligned - ref).abs().max().item() < 0.05
    assert mask.min().item() > 0.99


def test_different_reference_size_scales_matrix() -> None:
    ref = _reference(384)
    image, g = _transform(ref)
    small_ref = torch.nn.functional.interpolate(
        ref.permute(0, 3, 1, 2), scale_factor=0.5, mode="area"
    ).permute(0, 2, 3, 1)
    aligned, _, matrices, _ = auto_align(image, small_ref, model="affine")
    assert aligned.shape == small_ref.shape
    # image -> reference coords = half-scale of the exact inverse (pixel-centre convention).
    want = np.diag([0.5, 0.5, 1.0]) @ np.linalg.inv(g)
    pts = _corners(image.shape[2], image.shape[1])
    assert np.abs(_project(matrices[0], pts) - _project(want, pts)).max() < 2.0


def test_max_side_downscale_still_accurate() -> None:
    ref = _reference(512)
    image, g = _transform(ref, shift=(30.0, 16.0))
    _, _, matrices, _ = auto_align(image, ref, max_side=256)
    assert _corner_error(matrices[0], g, ref.shape[1:3]) < 5.0


def test_flat_image_falls_back(caplog: pytest.LogCaptureFixture) -> None:
    flat = torch.full((1, 64, 96, 3), 0.5)
    ref = torch.full((1, 80, 80, 3), 0.5)
    with caplog.at_level(logging.WARNING, logger="core.auto_align"):
        aligned, mask, matrices, inliers = auto_align(flat, ref)
    assert aligned.shape == (1, 80, 80, 3)
    assert torch.allclose(aligned, ref, atol=1e-4)
    assert torch.all(mask > 0.99)
    assert np.allclose(np.array(matrices[0]), auto_align_module._scale_matrix(80 / 96, 80 / 64))
    assert inliers[0] < 12
    assert "falling back" in caplog.text


def test_batch_broadcast_and_matching_batches() -> None:
    ref = _reference(256)
    image, _ = _transform(ref)
    batch = torch.cat([image, ref], dim=0)
    aligned, mask, matrices, inliers = auto_align(batch, ref)
    assert aligned.shape == (2, *ref.shape[1:]) and mask.shape == (2, *ref.shape[1:3])
    assert len(matrices) == 2 and len(inliers) == 2
    aligned2, _, _, _ = auto_align(batch, torch.cat([ref, ref], dim=0))
    assert torch.allclose(aligned, aligned2, atol=1e-4)


def test_deterministic() -> None:
    ref = _reference(256)
    image, _ = _transform(ref)
    first = auto_align(image, ref)[2]
    second = auto_align(image, ref)[2]
    assert np.allclose(np.array(first), np.array(second))


def test_validation_errors() -> None:
    ref = _reference(64)
    with pytest.raises(TypeError):
        auto_align("x", ref)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        auto_align(ref[..., :2], ref)
    with pytest.raises(ValueError, match="floating"):
        auto_align((ref * 255).to(torch.uint8), ref)
    with pytest.raises(ValueError, match="method"):
        auto_align(ref, ref, method="orb")
    with pytest.raises(ValueError, match="model"):
        auto_align(ref, ref, model="rigid")
    with pytest.raises(ValueError, match="batch"):
        auto_align(torch.cat([ref] * 3), torch.cat([ref] * 2))
    with pytest.raises(ValueError, match="max_side"):
        auto_align(ref, ref, max_side=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ransac_threshold"):
        auto_align(ref, ref, ransac_threshold=0.0)
    with pytest.raises(ValueError, match="min_inliers"):
        auto_align(ref, ref, min_inliers=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_input_stays_on_device() -> None:
    ref = _reference(256).cuda()
    image, _ = _transform(ref.cpu())
    aligned, mask, _, _ = auto_align(image.cuda(), ref)
    assert aligned.is_cuda and mask.is_cuda


def test_matcher_failure_is_logged_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(auto_align_module, "_match_sift", boom)
    ref = _reference(64)
    aligned, _, matrices, _ = auto_align(ref, ref)
    assert aligned.shape == ref.shape
    assert np.allclose(np.array(matrices[0]), np.eye(3))


@pytest.mark.skipif(not loftr_weights_path().is_file(), reason="LoFTR weights not cached")
def test_loftr_recovers_transform() -> None:
    ref = _reference(512)
    image, g = _transform(ref)
    _, _, matrices, inliers = auto_align(image, ref, method="loftr")
    assert inliers[0] >= 12
    assert _corner_error(matrices[0], g, ref.shape[1:3]) < 3.0


def _shifted_pair(side: int, dx: int, dy: int):
    """Return (image, ref, gt) with image(x) = ref(x - (dx, dy)); gt maps image -> ref pixels."""

    ref = _reference(side)
    height, width = ref.shape[1:3]
    shift = np.float32([[1, 0, dx], [0, 1, dy]])
    moved = cv2.warpAffine(
        ref[0].numpy(), shift, (width, height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE,
    )  # fmt: skip
    gt = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1.0]])
    return torch.from_numpy(moved)[None].contiguous(), ref, gt


@pytest.mark.parametrize("model", ["homography", "affine"])
def test_exact_integer_shift_subpixel_accuracy(model: str) -> None:
    image, ref, gt = _shifted_pair(1200, 13, -7)
    start = time.perf_counter()
    _, _, matrices, inliers = auto_align(image, ref, model=model, max_side=640)
    elapsed = time.perf_counter() - start
    assert inliers[0] >= 12
    assert _corner_error(matrices[0], np.linalg.inv(gt), ref.shape[1:3]) < 0.5
    assert elapsed < 20.0  # generous: CPU-only CI; CUDA runs in well under a second


def test_different_scales_use_pixel_centre_convention() -> None:
    # image matched at ~0.53x, reference at 1.0x: a missing half-pixel term biases the fit.
    image, ref, gt = _shifted_pair(1200, 13, -7)
    small_ref = torch.nn.functional.interpolate(
        ref.permute(0, 3, 1, 2), size=(ref.shape[1] // 2, ref.shape[2] // 2), mode="area"
    ).permute(0, 2, 3, 1)
    _, _, matrices, _ = auto_align(image, small_ref, model="affine", max_side=640)
    want = auto_align_module._scale_matrix(0.5, 0.5) @ gt
    pts = _corners(image.shape[2], image.shape[1])
    assert np.abs(_project(matrices[0], pts) - _project(want, pts)).max() < 0.3


def test_scale_matrix_pixel_centres() -> None:
    mat = auto_align_module._scale_matrix(0.5, 0.25)
    # centre of full pixel 0 (x=0) lies at (0+0.5)*s-0.5 in the resampled grid
    assert np.allclose(_project(mat, np.array([[0.0], [0.0], [1.0]])), [[-0.25], [-0.375]])
    assert np.allclose(_project(mat, np.array([[3.0], [3.0], [1.0]])), [[1.25], [0.375]])


@pytest.mark.parametrize("size", [8, 16, 31])
def test_tiny_reference_falls_back_instead_of_crashing(size: int) -> None:
    image = _reference(128)
    ref = torch.rand(1, size, size, 3)
    aligned, _, matrices, _ = auto_align(image, ref)
    assert aligned.shape == (1, size, size, 3)
    assert np.isfinite(np.array(matrices[0])).all()


@pytest.mark.parametrize("method", ["sift"])
def test_out_of_memory_is_not_swallowed(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    def oom(*_args, **_kwargs):
        raise torch.OutOfMemoryError("simulated")

    monkeypatch.setattr(auto_align_module, "_match_sift", oom)
    ref = _reference(64)
    with pytest.raises(torch.OutOfMemoryError):
        auto_align(ref, ref, method=method)


def test_model_construction_does_not_advance_global_rng() -> None:
    auto_align_module._MODEL_CACHE.clear()
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)
    ref = _reference(64)
    auto_align(ref, ref)  # constructs SIFT
    assert torch.equal(torch.rand(3), expected)


def test_affine_fit_leaves_numpy_global_rng_untouched() -> None:
    np.random.seed(5)
    expected = np.random.rand(2)
    np.random.seed(5)
    pts = torch.rand(60, 2) * 100
    auto_align_module._fit_affine(pts, pts + 3.0, 2.0)
    assert np.array_equal(np.random.rand(2), expected)
