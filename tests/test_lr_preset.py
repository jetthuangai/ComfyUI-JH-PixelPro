from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from core.color_lab import apply_colorlab_pipeline, merge_params_json, params_to_json
from core.lr_preset import (
    _Lua,
    list_presets,
    load_lightroom_preset,
    parse_xmp,
    resolve_preset,
    to_colorlab,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "lr"
XMP = FIXTURES / "warm-film.xmp"
LRT = FIXTURES / "classic-bw.lrtemplate"


def test_xmp_reads_attributes_elements_and_title() -> None:
    title, props = parse_xmp(XMP.read_text("utf-8"))
    assert title == "JH Warm Film"
    assert props["Exposure2012"] == "+0.35"
    assert props["ToneCurvePV2012"] == ["0, 18", "64, 60", "192, 200", "255, 250"]


def test_xmp_maps_to_colorlab() -> None:
    preset = load_lightroom_preset(XMP)
    p = preset.params
    assert preset.title == "JH Warm Film"
    assert p["basic_exposure"] == pytest.approx(0.35)
    assert p["basic_highlights"] == -45.0 and p["basic_shadows"] == 30.0
    assert p["hsl_orange_hue"] == -5.0 and p["hsl_blue_sat"] == -20.0
    assert p["grade_shadow_hue"] == 215.0 and p["grade_shadow_sat"] == 18.0
    assert p["grade_mid_hue"] == 30.0
    assert p["grade_shadow_bal"] == p["grade_mid_bal"] == p["grade_highlight_bal"] == 10.0
    assert p["wb_temperature"] == 8.0
    assert p["tone_curve"] == [[0.0, 18.0], [64.0, 60.0], [192.0, 200.0], [255.0, 250.0]]
    assert "tone_curve_red" not in p  # identity curve
    assert "gray_enable" not in p  # ConvertToGrayscale False
    assert p["grain_amount"] == 25.0 and p["vignette_amount"] == -15.0
    assert preset.ignored == ["Look (Vintage 06)"]  # a creative profile is Adobe's own LUT
    assert preset.skipped == ["Sharpness"]  # detail, not colour
    assert "ColorGradeBlending" not in preset.ignored  # no visible effect alone
    report = preset.report()
    assert "JH Warm Film" in report and "Vintage 06" in report


def test_lrtemplate_maps_bw_preset() -> None:
    preset = load_lightroom_preset(LRT)
    assert preset.title == 'Classic "Silver" B&W'
    p = preset.params
    assert p["gray_enable"] is True
    assert p["gray_red"] == 20.0 and p["gray_blue"] == -30.0
    assert p["tone_curve"] == [[0.0, 0.0], [128.0, 140.0], [255.0, 255.0]]
    assert p["gray_magenta"] == 5.0 and preset.ignored == []
    assert preset.skipped == ["LensProfileEnable"]


def test_lua_reader_is_data_only() -> None:
    with pytest.raises(ValueError):
        _Lua('s = { title = os.execute("calc") }').document()
    with pytest.raises(ValueError):
        _Lua("s = { a = 1 } extra = 2").document()
    assert _Lua('return { ["key"] = { 1, 2, }, -- comment\n flag = false }').document() == {
        "key": [1.0, 2.0],
        "flag": False,
    }


def test_bad_files_raise(tmp_path: Path) -> None:
    broken = tmp_path / "broken.xmp"
    broken.write_text("<x:xmpmeta", "utf-8")
    with pytest.raises(ValueError):
        load_lightroom_preset(broken)
    with pytest.raises(ValueError):
        load_lightroom_preset(tmp_path / "missing.xmp")
    other = tmp_path / "look.cube"
    other.write_text("x", "utf-8")
    with pytest.raises(ValueError):
        load_lightroom_preset(other)


def test_list_and_resolve_presets(tmp_path: Path) -> None:
    (tmp_path / "Film").mkdir()
    (tmp_path / "Film" / "a.xmp").write_text(XMP.read_text("utf-8"), "utf-8")
    (tmp_path / "b.lrtemplate").write_text(LRT.read_text("utf-8"), "utf-8")
    (tmp_path / "notes.txt").write_text("x", "utf-8")
    assert list_presets([tmp_path]) == ["b.lrtemplate", "Film/a.xmp"]
    assert resolve_preset("Film/a.xmp", dirs=[tmp_path]) == (tmp_path / "Film" / "a.xmp").resolve()
    with pytest.raises(ValueError):
        resolve_preset("../outside.xmp", dirs=[tmp_path])
    assert resolve_preset("ignored", path=str(XMP), dirs=[tmp_path]) == XMP


def test_tone_curve_from_preset_lifts_blacks() -> None:
    preset = load_lightroom_preset(XMP)
    params = merge_params_json({}, params_to_json({"tone_curve": preset.params["tone_curve"]}))
    black = torch.zeros(1, 4, 4, 3)
    out = apply_colorlab_pipeline(black, params)
    assert out.mean().item() == pytest.approx(18.0 / 255.0, abs=2e-3)
    white = torch.ones(1, 4, 4, 3)
    assert apply_colorlab_pipeline(white, params).mean().item() == pytest.approx(
        250 / 255, abs=2e-3
    )


def test_tone_curve_is_monotone() -> None:
    params = merge_params_json(
        {}, json.dumps({"tone_curve": [[0, 0], [60, 110], [128, 120], [255, 255]]})
    )
    ramp = torch.linspace(0, 1, 256).view(1, 1, 256, 1).expand(1, 1, 256, 3).contiguous()
    out = apply_colorlab_pipeline(ramp, params)[0, 0, :, 0]
    assert (out[1:] - out[:-1] >= -1e-6).all()


def test_bad_curve_rejected() -> None:
    with pytest.raises(ValueError):
        merge_params_json({}, json.dumps({"tone_curve": [[0, 0]]}))
    with pytest.raises(ValueError):
        merge_params_json({}, json.dumps({"tone_curve": [[0, 0], [300, 10]]}))


def test_unknown_crs_keys_are_reported_not_dropped() -> None:
    preset = to_colorlab({"Exposure2012": "0.5", "FutureSlider": "12", "LuminanceSmoothing": "0"})
    assert preset.params == {"basic_exposure": 0.5}
    assert preset.ignored == ["FutureSlider"]  # zero-valued keys are not noise in the report


def test_parametric_curve_composes_with_point_curve() -> None:
    props = {
        "ParametricDarks": "-34",
        "ParametricLights": "+35",
        "ToneCurvePV2012": ["0, 36", "128, 131", "255, 239"],
    }
    preset = to_colorlab(props)
    curve = preset.params["tone_curve"]
    assert len(curve) == 33 and "ParametricCurve" in preset.applied
    assert curve[0] == [0.0, 36.0]  # faded black point kept
    plain = to_colorlab({"ToneCurvePV2012": props["ToneCurvePV2012"]}).params["tone_curve"]
    plain_at_64 = plain[0][1] + (plain[1][1] - plain[0][1]) * 64 / plain[1][0]  # linear part
    composed_at_64 = next(y for x, y in curve if x >= 63)
    assert composed_at_64 < plain_at_64 - 3.0  # darks pulled down by the parametric part


def test_parametric_alone_is_s_curve_and_monotone() -> None:
    curve = to_colorlab({"ParametricDarks": "-40", "ParametricLights": "+40"}).params["tone_curve"]
    ys = [y for _, y in curve]
    assert all(b >= a for a, b in zip(ys, ys[1:], strict=False))
    assert ys[8] < 64.0 and ys[24] > 191.0  # x=64 darker, x=191 brighter


def test_calibration_maps_and_changes_colour() -> None:
    preset = to_colorlab({"RedSaturation": "+5", "BlueHue": "-3", "ShadowTint": "+10"})
    assert preset.params == {
        "calib_red_sat": 5.0,
        "calib_blue_hue": -3.0,
        "calib_shadow_tint": 10.0,
    }
    image = torch.rand(1, 8, 8, 3) * 0.3
    out = apply_colorlab_pipeline(image, preset.params)
    assert not torch.allclose(out, image, atol=1e-4)
    grey = torch.full((1, 4, 4, 3), 0.5)
    only_matrix = {k: v for k, v in preset.params.items() if k != "calib_shadow_tint"}
    assert torch.allclose(apply_colorlab_pipeline(grey, only_matrix), grey, atol=2e-3)  # white kept


def test_kelvin_white_balance_against_as_shot() -> None:
    warm = to_colorlab(
        {
            "WhiteBalance": "Custom",
            "Temperature": "10000",
            "Tint": "+20",
            "AsShotTemperature": "4450",
            "AsShotTint": "24",
        }
    )
    assert warm.params["wb_temperature"] == pytest.approx(99.8, abs=0.1)  # ~125 mired warmer
    assert warm.params["wb_tint"] == -4.0
    assert warm.notes and "approximated" in warm.notes[0]
    cool = to_colorlab({"WhiteBalance": "Tungsten", "AsShotTemperature": "5500"})
    assert cool.params["wb_temperature"] < -50.0
    lone = to_colorlab({"Temperature": "5000"})
    assert "wb_temperature" not in lone.params and "not applied" in lone.notes[0]


def test_default_profile_and_empty_point_colors_are_neutral() -> None:
    xmp = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"'
        ' crs:Exposure2012="+0.2">'
        "<crs:PointColors><rdf:Seq><rdf:li>-1.000000, -1.000000, -1.000000</rdf:li>"
        "</rdf:Seq></crs:PointColors>"
        '<crs:Look><rdf:Description crs:Name="Adobe Color" crs:Amount="1" crs:UUID="X"/></crs:Look>'
        "</rdf:Description></rdf:RDF></x:xmpmeta>"
    )
    title, props = parse_xmp(xmp)
    assert title is None and props["Look"]["Name"] == "Adobe Color"
    assert "Amount" not in props and "UUID" not in props  # nested profile fields stay nested
    preset = to_colorlab(props)
    assert preset.params == {"basic_exposure": 0.2} and preset.ignored == []


def test_grain_is_deterministic_and_textures_midtones() -> None:
    grey = torch.full((1, 64, 64, 3), 0.5)
    params = {"grain_amount": 40.0, "grain_size": 25.0, "grain_frequency": 50.0}
    a = apply_colorlab_pipeline(grey, params)
    b = apply_colorlab_pipeline(grey, params)
    assert torch.equal(a, b)
    assert a.std().item() > 0.01
    assert (a[..., 0] == a[..., 1]).all()  # luminance grain, no colour noise
    rough = apply_colorlab_pipeline(grey, {**params, "grain_frequency": 100.0})
    neighbour_diff = lambda x: (x[:, :, 1:] - x[:, :, :-1]).abs().mean()  # noqa: E731
    assert neighbour_diff(rough) > neighbour_diff(a)  # rougher = finer grain


def test_vignette_darkens_corners_only() -> None:
    grey = torch.full((1, 60, 90, 3), 0.6)
    out = apply_colorlab_pipeline(grey, {"vignette_amount": -60.0})
    assert out[0, 30, 45, 0].item() == pytest.approx(0.6, abs=1e-3)
    assert out[0, 0, 0, 0].item() < 0.45
    light = apply_colorlab_pipeline(grey, {"vignette_amount": 60.0})
    assert light[0, 0, 0, 0].item() > 0.7


def test_gray_magenta_only_when_set() -> None:
    torch.manual_seed(2)
    image = torch.rand(1, 16, 16, 3)
    base = {"gray_enable": True, "gray_red": 20.0}
    assert torch.equal(
        apply_colorlab_pipeline(image, base),
        apply_colorlab_pipeline(image, {**base, "gray_magenta": 0.0}),
    )
    assert not torch.equal(
        apply_colorlab_pipeline(image, base),
        apply_colorlab_pipeline(image, {**base, "gray_magenta": 60.0}),
    )


def test_effect_shapes_dropped_without_amount() -> None:
    preset = to_colorlab(
        {"GrainSize": "20", "GrainFrequency": "45", "PostCropVignetteMidpoint": "50"}
    )
    assert preset.params == {}
