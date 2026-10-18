"""Culling score core (``core/cull.py``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageFilter

from core.cull import cull_scores, select

_ROOT = Path(__file__).resolve().parents[1] / "workflows"


def _load(name: str, blur: float = 0.0) -> torch.Tensor:
    image = Image.open(_ROOT / name).convert("RGB").resize((300, 450))
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)


def _batch() -> torch.Tensor:
    portrait = _load("sample_portrait.jpg")
    return torch.stack(
        [
            portrait,  # 0 sharp
            _load("sample_portrait.jpg", blur=5),  # 1 blurred
            (portrait * 2.5).clamp(0, 1),  # 2 blown out
            _load("sample_cafe.jpg"),  # 3 another scene
        ]
    )


def test_sharp_beats_blurred_and_blown_out() -> None:
    rows = cull_scores(_batch(), check_eyes=False, use_faces=False)
    assert rows[0]["sharpness"] > rows[1]["sharpness"] + 0.3
    assert rows[2]["exposure"] < 0.3 < rows[0]["exposure"]
    assert rows[0]["score"] > rows[1]["score"] and rows[0]["score"] > rows[2]["score"]


def test_duplicates_keep_the_best_of_a_scene() -> None:
    rows = cull_scores(_batch(), check_eyes=False, use_faces=False)
    assert rows[0]["duplicate_of"] is None
    assert rows[1]["duplicate_of"] == 0 and rows[2]["duplicate_of"] == 0
    assert rows[3]["duplicate_of"] is None  # different scene
    assert sorted(select(rows, 0, 0.0)) == [0, 3]
    assert len(select(rows, 1, 0.0)) == 1


def test_scene_relative_sharpness_does_not_punish_smooth_scenes() -> None:
    rows = cull_scores(_batch(), check_eyes=False, use_faces=False)
    assert rows[0]["sharpness"] > 0.5  # smooth skin, but the sharpest shot of its scene


def test_scores_are_size_independent() -> None:
    small = cull_scores(_batch(), check_eyes=False, use_faces=False)
    big_batch = torch.nn.functional.interpolate(
        _batch().permute(0, 3, 1, 2), scale_factor=2, mode="bilinear"
    ).permute(0, 2, 3, 1)
    big = cull_scores(big_batch, check_eyes=False, use_faces=False)
    assert [r["duplicate_of"] for r in small] == [r["duplicate_of"] for r in big]
    assert abs(small[2]["exposure"] - big[2]["exposure"]) < 0.05


@pytest.mark.parametrize(("args", "match"), [((-1, 0.5), "top_k"), ((0, 1.5), "min_score")])
def test_select_validates(args: tuple, match: str) -> None:
    rows = cull_scores(_batch()[:2], check_eyes=False, use_faces=False)
    with pytest.raises(ValueError, match=match):
        select(rows, *args)


def test_eye_aspect_ratio_open_vs_closed() -> None:
    from core.cull import _EAR_CLOSED, _LEFT_EYE, _ear

    points = np.zeros((468, 2), dtype=np.float32)
    p1, p2, p3, p4, p5, p6 = _LEFT_EYE
    points[p1], points[p4] = (0, 0), (30, 0)
    for lid in (8.0, 1.0):  # open, closed
        points[p2], points[p3] = (10, -lid), (20, -lid)
        points[p6], points[p5] = (10, lid), (20, lid)
        ear = _ear(points, _LEFT_EYE)
        assert (ear < _EAR_CLOSED) == (lid < 2)


def test_blurred_photo_alone_is_not_kept() -> None:
    blurred = _load("sample_portrait.jpg", blur=8)[None]
    rows = cull_scores(blurred, check_eyes=False, use_faces=False)
    assert rows[0]["sharpness"] < 0.4
    assert select(rows, 0, 0.5) == []


def test_drifting_burst_is_one_group() -> None:
    from core.cull import _groups

    rows = [{"hash": 0}, {"hash": (1 << 8) - 1}, {"hash": (1 << 16) - 1}]  # A-B 8, B-C 8, A-C 16
    assert len(set(_groups(rows, 10))) == 1


def test_tiny_images_rejected() -> None:
    with pytest.raises(ValueError, match="at least"):
        cull_scores(torch.rand(1, 2, 2, 3))
