"""Regression tests for Batch-9 ColorLab core."""

from __future__ import annotations

import pytest
import torch

from core.color_lab import HUE_ANCHORS, apply_colorlab_pipeline, rgb_to_hsv, solid_hsv


def _gradient(size: int = 16) -> torch.Tensor:
    ramp = torch.linspace(0.0, 1.0, size)
    xx = ramp.view(1, 1, size, 1)
    yy = ramp.view(1, size, 1, 1)
    return torch.cat(
        [
            xx.expand(1, size, size, 1),
            yy.expand(1, size, size, 1),
            ((xx + yy) / 2.0).expand(1, size, size, 1),
        ],
        dim=-1,
    )


def _solid(color: tuple[float, float, float], size: int = 8) -> torch.Tensor:
    return torch.tensor(color, dtype=torch.float32).view(1, 1, 1, 3).expand(1, size, size, 3)


def _params(**overrides):
    params = {key: 0.0 for key in []}
    params["gray_enable"] = False
    params.update(overrides)
    return params


def _mean_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean(torch.abs(a - b)).item()


def test_identity_all_zero_preserves_input() -> None:
    image = _gradient()
    out = apply_colorlab_pipeline(image, _params())
    assert out.data_ptr() == image.data_ptr()
    assert torch.equal(out, image)


def test_gray_enable_false_preserves_color_even_with_gray_sliders() -> None:
    image = _solid((0.8, 0.2, 0.1))
    out = apply_colorlab_pipeline(image, _params(gray_red=300.0, gray_blue=-200.0))
    assert torch.equal(out, image)


def test_exposure_doubles_at_one_stop() -> None:
    image = _solid((0.2, 0.25, 0.3))
    out = apply_colorlab_pipeline(image, _params(basic_exposure=1.0))
    assert torch.allclose(out, image * 2.0, atol=1e-5)


def test_contrast_moves_values_away_from_midgray() -> None:
    image = _solid((0.25, 0.5, 0.75))
    out = apply_colorlab_pipeline(image, _params(basic_contrast=50.0))
    assert out[..., 0].mean().item() < image[..., 0].mean().item()
    assert out[..., 2].mean().item() > image[..., 2].mean().item()


def test_highlights_negative_reduces_bright_pixels_more_than_dark() -> None:
    image = torch.cat([_solid((0.2, 0.2, 0.2), 4), _solid((0.9, 0.9, 0.9), 4)], dim=2)
    out = apply_colorlab_pipeline(image, _params(basic_highlights=-80.0))
    dark_delta = _mean_delta(out[:, :, :4], image[:, :, :4])
    bright_delta = _mean_delta(out[:, :, 4:], image[:, :, 4:])
    assert bright_delta > dark_delta * 3.0


def test_shadows_positive_lifts_dark_pixels_more_than_bright() -> None:
    image = torch.cat([_solid((0.15, 0.15, 0.15), 4), _solid((0.85, 0.85, 0.85), 4)], dim=2)
    out = apply_colorlab_pipeline(image, _params(basic_shadows=80.0))
    dark_delta = _mean_delta(out[:, :, :4], image[:, :, :4])
    bright_delta = _mean_delta(out[:, :, 4:], image[:, :, 4:])
    assert dark_delta > bright_delta * 3.0


def test_whites_positive_lifts_near_white() -> None:
    image = _solid((0.82, 0.82, 0.82))
    out = apply_colorlab_pipeline(image, _params(basic_whites=100.0))
    assert out.mean().item() > image.mean().item()


def test_blacks_negative_crushes_near_black() -> None:
    image = _solid((0.18, 0.18, 0.18))
    out = apply_colorlab_pipeline(image, _params(basic_blacks=-100.0))
    assert out.mean().item() < image.mean().item()


def test_texture_changes_checker_detail() -> None:
    image = _gradient(16)
    out = apply_colorlab_pipeline(image, _params(basic_texture=80.0))
    assert _mean_delta(out, image) > 0.0005


def test_clarity_changes_midtones() -> None:
    image = _gradient(16)
    out = apply_colorlab_pipeline(image, _params(basic_clarity=80.0))
    assert _mean_delta(out, image) > 0.0005


def test_dehaze_changes_global_contrast() -> None:
    image = _gradient(16)
    out = apply_colorlab_pipeline(image, _params(basic_dehaze=80.0))
    assert _mean_delta(out, image) > 0.005


def test_vibrance_protects_skin_more_than_gray() -> None:
    skin = _solid((0.82, 0.5, 0.36), 4)
    muted_blue = _solid((0.35, 0.45, 0.55), 4)
    image = torch.cat([skin, muted_blue], dim=2)
    out = apply_colorlab_pipeline(image, _params(basic_vibrance=100.0))
    assert _mean_delta(out[:, :, :4], skin) < _mean_delta(out[:, :, 4:], muted_blue)


def test_saturation_boost_increases_channel_spread() -> None:
    image = _solid((0.45, 0.35, 0.25))
    out = apply_colorlab_pipeline(image, _params(basic_saturation=80.0))
    spread_in = (image.max(dim=-1).values - image.min(dim=-1).values).mean()
    spread_out = (out.max(dim=-1).values - out.min(dim=-1).values).mean()
    assert spread_out > spread_in


@pytest.mark.parametrize("color,anchor", list(HUE_ANCHORS.items()))
def test_hsl_anchor_hue_shift_changes_target_color(color: str, anchor: float) -> None:
    like = _solid((0.0, 0.0, 0.0))
    image = solid_hsv(anchor, 0.8, 0.7, like)
    out = apply_colorlab_pipeline(image, _params(**{f"hsl_{color}_hue": 50.0}))
    h_in, _, _ = rgb_to_hsv(image)
    h_out, _, _ = rgb_to_hsv(out)
    assert torch.mean(torch.abs(h_out - h_in)).item() > 5.0


def test_hsl_red_shift_leaves_blue_mostly_unchanged() -> None:
    image = _solid((0.05, 0.05, 0.95))
    out = apply_colorlab_pipeline(image, _params(hsl_red_hue=100.0))
    assert torch.allclose(out, image, atol=1e-3)


def test_hsl_sat_slider_changes_target_saturation() -> None:
    image = _solid((0.85, 0.4, 0.25))
    out = apply_colorlab_pipeline(image, _params(hsl_orange_sat=80.0))
    _, s_in, _ = rgb_to_hsv(image)
    _, s_out, _ = rgb_to_hsv(out)
    assert s_out.mean().item() > s_in.mean().item()


def test_hsl_lum_slider_changes_target_value() -> None:
    image = _solid((0.05, 0.25, 0.95))
    out = apply_colorlab_pipeline(image, _params(hsl_blue_lum=-80.0))
    assert out.mean().item() < image.mean().item()


def test_shadow_grading_isolates_dark_region() -> None:
    image = torch.cat([_solid((0.12, 0.12, 0.12), 4), _solid((0.86, 0.86, 0.86), 4)], dim=2)
    out = apply_colorlab_pipeline(image, _params(grade_shadow_hue=200.0, grade_shadow_sat=80.0))
    assert (
        _mean_delta(out[:, :, :4], image[:, :, :4])
        > _mean_delta(out[:, :, 4:], image[:, :, 4:]) * 2.0
    )


def test_mid_grading_isolates_mid_region() -> None:
    image = torch.cat([_solid((0.12, 0.12, 0.12), 4), _solid((0.5, 0.5, 0.5), 4)], dim=2)
    out = apply_colorlab_pipeline(image, _params(grade_mid_hue=30.0, grade_mid_sat=80.0))
    assert _mean_delta(out[:, :, 4:], image[:, :, 4:]) > _mean_delta(out[:, :, :4], image[:, :, :4])


def test_highlight_grading_isolates_bright_region() -> None:
    image = torch.cat([_solid((0.15, 0.15, 0.15), 4), _solid((0.9, 0.9, 0.9), 4)], dim=2)
    out = apply_colorlab_pipeline(
        image, _params(grade_highlight_hue=45.0, grade_highlight_sat=80.0)
    )
    assert (
        _mean_delta(out[:, :, 4:], image[:, :, 4:])
        > _mean_delta(out[:, :, :4], image[:, :, :4]) * 2.0
    )


def test_grade_balance_changes_shadow_mask_strength() -> None:
    image = _solid((0.45, 0.45, 0.45))
    left = apply_colorlab_pipeline(
        image, _params(grade_shadow_hue=200.0, grade_shadow_sat=80.0, grade_shadow_bal=-100.0)
    )
    right = apply_colorlab_pipeline(
        image, _params(grade_shadow_hue=200.0, grade_shadow_sat=80.0, grade_shadow_bal=100.0)
    )
    assert _mean_delta(left, image) != pytest.approx(_mean_delta(right, image))


def test_gray_mix_outputs_black_and_white() -> None:
    image = _solid((0.8, 0.35, 0.2))
    out = apply_colorlab_pipeline(image, _params(gray_enable=True))
    assert torch.allclose(out[..., 0], out[..., 1], atol=1e-6)
    assert torch.allclose(out[..., 1], out[..., 2], atol=1e-6)


def test_gray_mix_slider_changes_luma() -> None:
    image = _solid((0.85, 0.35, 0.2))
    base = apply_colorlab_pipeline(image, _params(gray_enable=True, gray_orange=0.0))
    bright = apply_colorlab_pipeline(image, _params(gray_enable=True, gray_orange=200.0))
    assert bright.mean().item() > base.mean().item()
