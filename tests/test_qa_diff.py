from __future__ import annotations

import kornia
import pytest
import torch

from core.qa_diff import PSNR_CAP_DB, qa_diff

_G = torch.Generator().manual_seed(0)


def _img(b=1, h=64, w=64) -> torch.Tensor:
    return torch.rand(b, h, w, 3, generator=_G) * 0.6 + 0.2


def test_identical() -> None:
    a = _img()
    diff, m = qa_diff(a, a.clone(), diff_mode="abs")
    assert m[0]["ssim"] == pytest.approx(1.0, abs=1e-5)
    assert m[0]["psnr"] == PSNR_CAP_DB
    assert m[0]["mae"] == 0.0 and m[0]["max_abs"] == 0.0 and m[0]["changed_pct"] == 0.0
    assert diff.shape == a.shape and diff.dtype == torch.float32
    assert diff.max() == 0.0


def test_constant_offset() -> None:
    a = _img()
    _, m = qa_diff(a, a + 0.1)
    assert m[0]["mae"] == pytest.approx(0.1, abs=1e-5)
    assert m[0]["psnr"] == pytest.approx(20.0, abs=0.01)
    assert m[0]["max_abs"] == pytest.approx(0.1, abs=1e-5)
    assert m[0]["changed_pct"] == pytest.approx(100.0)


def test_ssim_decreases_with_noise() -> None:
    a = _img()
    lo = qa_diff(a, (a + 0.02 * torch.randn_like(a)).clamp(0, 1))[1][0]["ssim"]
    hi = qa_diff(a, (a + 0.2 * torch.randn_like(a)).clamp(0, 1))[1][0]["ssim"]
    assert hi < lo < 1.0


def test_mask_restricts_metrics() -> None:
    a = _img()
    b = a.clone()
    b[:, :, 32:] += 0.2
    mask = torch.zeros(1, 64, 64)
    mask[:, :, :32] = 1.0
    _, full = qa_diff(a, b)
    _, left = qa_diff(a, b, mask=mask)
    _, right = qa_diff(a, b, mask=1.0 - mask)
    assert full[0]["mae"] == pytest.approx(0.1, abs=1e-4)
    assert left[0]["mae"] == 0.0 and left[0]["changed_pct"] == 0.0
    assert right[0]["mae"] == pytest.approx(0.2, abs=1e-4)
    assert right[0]["psnr"] < full[0]["psnr"]


def test_empty_mask_neutral() -> None:
    a = _img()
    _, m = qa_diff(a, a + 0.1, mask=torch.zeros(1, 64, 64))
    assert m[0]["ssim"] == 1.0 and m[0]["psnr"] == PSNR_CAP_DB


@pytest.mark.parametrize("mode", ["abs", "heatmap", "ssim_map"])
def test_modes_shape_range(mode: str) -> None:
    a = _img()
    diff, _ = qa_diff(a, (a + 0.1 * torch.randn_like(a)).clamp(0, 1), diff_mode=mode)
    assert diff.shape == a.shape
    assert diff.min() >= 0.0 and diff.max() <= 1.0
    assert diff.max() > 0.0


def test_heatmap_is_colour_and_ssim_map_grey() -> None:
    a = _img()
    b = (a + 0.1 * torch.randn_like(a)).clamp(0, 1)
    heat, _ = qa_diff(a, b, diff_mode="heatmap")
    grey, _ = qa_diff(a, b, diff_mode="ssim_map")
    assert not torch.allclose(heat[..., 0], heat[..., 2])
    assert torch.equal(grey[..., 0], grey[..., 1])


def test_batch_broadcast() -> None:
    a = _img(b=3)
    b = _img(b=1)
    diff, m = qa_diff(a, b)
    assert diff.shape == a.shape and len(m) == 3
    _, m_rev = qa_diff(b, a)
    assert len(m_rev) == 3


def test_batch_mismatch() -> None:
    with pytest.raises(ValueError, match="batch"):
        qa_diff(_img(b=2), _img(b=3))


def test_size_mismatch() -> None:
    with pytest.raises(ValueError, match="spatial sizes must match"):
        qa_diff(_img(), _img(h=32))


def test_validation() -> None:
    a = _img()
    with pytest.raises(TypeError):
        qa_diff("x", a)
    with pytest.raises(ValueError, match="odd"):
        qa_diff(a, a, window_size=10)
    with pytest.raises(ValueError, match="window_size"):
        qa_diff(a, a, window_size=33)
    with pytest.raises(ValueError, match="amplify"):
        qa_diff(a, a, amplify=0.5)
    with pytest.raises(ValueError, match="amplify"):
        qa_diff(a, a, amplify=101.0)
    with pytest.raises(ValueError, match="diff_mode"):
        qa_diff(a, a, diff_mode="nope")
    with pytest.raises(ValueError, match="shape"):
        qa_diff(a[..., :1], a)
    with pytest.raises(ValueError, match="mask"):
        qa_diff(a, a, mask=torch.ones(1, 32, 32))


_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@_CUDA
@pytest.mark.parametrize("mode", ["abs", "heatmap", "ssim_map"])
def test_cuda_all_modes_with_mask(mode: str) -> None:
    a = _img().cuda()
    b = (a + 0.1 * torch.randn_like(a)).clamp(0, 1)
    mask = torch.ones(1, 64, 64, device="cuda")
    diff, m = qa_diff(a, b, diff_mode=mode, mask=mask)
    assert diff.device.type == "cuda" and diff.shape == a.shape
    cpu_diff, cpu_m = qa_diff(a.cpu(), b.cpu(), diff_mode=mode, mask=mask.cpu())
    assert torch.allclose(diff.cpu(), cpu_diff, atol=1e-3)
    assert m[0]["psnr"] == pytest.approx(cpu_m[0]["psnr"], abs=1e-2)


@_CUDA
def test_mixed_devices_follow_image_a() -> None:
    a = _img()
    b = (a + 0.1).cuda()
    diff, m = qa_diff(a, b, diff_mode="heatmap", mask=torch.ones(1, 64, 64, device="cuda"))
    assert diff.device == a.device
    assert m[0]["mae"] == pytest.approx(0.1, abs=1e-4)


@pytest.mark.parametrize("size", [3, 5, 11])
def test_small_images_clamp_window(size: int) -> None:
    a = _img(h=size, w=size)
    diff, m = qa_diff(a, (a + 0.05).clamp(0, 1), window_size=11)
    assert diff.shape == a.shape
    assert 0.0 < m[0]["ssim"] <= 1.0


@pytest.mark.parametrize("hw", [(1, 1), (2, 8), (8, 2)])
def test_too_small_images_raise(hw: tuple[int, int]) -> None:
    a = _img(h=hw[0], w=hw[1])
    with pytest.raises(ValueError, match="at least 3x3"):
        qa_diff(a, a)


def test_ssim_matches_kornia_channel_mean() -> None:
    a = _img()
    b = (a + 0.1 * torch.randn_like(a)).clamp(0, 1)
    ref = kornia.metrics.ssim(a.permute(0, 3, 1, 2), b.permute(0, 3, 1, 2), 7, max_val=1.0).mean()
    _, m = qa_diff(a, b, window_size=7)
    assert m[0]["ssim"] == pytest.approx(ref.item(), abs=1e-6)


def test_ssim_closed_form_constant_images() -> None:
    # Constant images: variances/covariance are 0, so SSIM = (2*ma*mb + C1) / (ma^2 + mb^2 + C1).
    c1 = 0.01**2
    a = torch.full((1, 16, 16, 3), 0.5)
    b = torch.full((1, 16, 16, 3), 0.3)
    expected = (2 * 0.5 * 0.3 + c1) / (0.5**2 + 0.3**2 + c1)
    _, m = qa_diff(a, b)
    assert m[0]["ssim"] == pytest.approx(expected, abs=1e-5)
