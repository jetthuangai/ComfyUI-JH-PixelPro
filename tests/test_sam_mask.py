"""Tests for core.sam_mask: prompt parsing (no model) and SAM runs (skipped w/o checkpoints)."""

from __future__ import annotations

import hashlib
import json
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from core import sam_mask as sm

_ROOT = Path(__file__).resolve().parents[1]
_SIZE = (512, 768)  # (W, H) — keeps CPU runs of mobile_sam around a second

needs_mobile = pytest.mark.skipif(
    not sm.checkpoint_path("mobile_sam").exists(), reason="mobile_sam.pt not in models/sams"
)
needs_vit_b_cuda = pytest.mark.skipif(
    not (sm.checkpoint_path("vit_b").exists() and torch.cuda.is_available()),
    reason="needs sam_vit_b checkpoint and CUDA",
)


@pytest.fixture(scope="module")
def portrait() -> torch.Tensor:
    img = Image.open(_ROOT / "workflows" / "sample_portrait.jpg").convert("RGB").resize(_SIZE)
    return torch.from_numpy(np.array(img)).float()[None] / 255.0


@pytest.fixture(scope="module")
def truth() -> torch.Tensor:
    m = Image.open(_ROOT / "workflows" / "sample_binary_mask.png").convert("L").resize(_SIZE)
    return torch.from_numpy(np.array(m)).float()[None] / 255.0 > 0.5


def _iou(mask: torch.Tensor, truth: torch.Tensor) -> float:
    m = mask > 0.5
    return float((m & truth).sum() / (m | truth).sum())


# ----------------------------------------------------------------- parsing (no model)


def test_points_pixels_and_normalised() -> None:
    assert sm.parse_points_json("[[10, 20], [30.5, 40]]", 100, 100) == [(10, 20), (30.5, 40)]
    assert sm.parse_points_json("[[0.5, 0.25]]", 101, 201) == [(50.0, 50.0)]


def test_points_empty_and_bare_pair() -> None:
    assert sm.parse_points_json("", 10, 10) == []
    assert sm.parse_points_json("  ", 10, 10) == []
    assert sm.parse_points_json("[]", 10, 10) == []
    assert sm.parse_points_json("[5, 6]", 10, 10) == [(5, 6)]


@pytest.mark.parametrize(
    "text",
    ["nope", '{"a": 1}', "[[1]]", "[[1, 2, 3]]", '[["a", "b"]]', "[[true, 1]]", "[[500, 5]]"],
)
def test_points_invalid(text: str) -> None:
    with pytest.raises(ValueError):
        sm.parse_points_json(text, 100, 100)


def test_points_non_str() -> None:
    with pytest.raises(TypeError):
        sm.parse_points_json(5, 10, 10)  # type: ignore[arg-type]


FACE = [{"x": 10, "y": 20, "w": 30, "h": 40, "conf": 0.9, "batch_index": 0}]


def test_bbox_face_detect_format() -> None:
    assert sm.parse_bbox_json(json.dumps(FACE), 100, 100) == (10, 20, 40, 60)


def test_bbox_batch_index_matching() -> None:
    boxes = [
        {"x": 1, "y": 1, "w": 10, "h": 10, "batch_index": 0},
        {"x": 50, "y": 50, "w": 20, "h": 20, "batch_index": 1},
    ]
    text = json.dumps(boxes)
    assert sm.parse_bbox_json(text, 100, 100, 1) == (50, 50, 70, 70)
    assert sm.parse_bbox_json(text, 100, 100, 2) is None


def test_bbox_without_batch_index_uses_first_and_clamps() -> None:
    text = json.dumps([{"x": -5, "y": 90, "w": 30, "h": 50}, {"x": 0, "y": 0, "w": 5, "h": 5}])
    assert sm.parse_bbox_json(text, 100, 100, 3) == (0, 90, 25, 100)


@pytest.mark.parametrize("text", ["x", "[1, 2]", '[{"x": 1}]', '[{"x":1,"y":1,"w":0,"h":5}]'])
def test_bbox_invalid(text: str) -> None:
    with pytest.raises(ValueError):
        sm.parse_bbox_json(text, 100, 100)


def test_bbox_empty() -> None:
    assert sm.parse_bbox_json("", 10, 10) is None
    assert sm.parse_bbox_json("[]", 10, 10) is None


def test_mask_hint_prompt_box_and_inside_point() -> None:
    m = torch.zeros(20, 20)
    m[5:10, 4:12] = 1.0
    box, pt = sm.mask_hint_prompt(m)
    assert box == (4.0, 5.0, 12.0, 10.0)
    assert m[int(pt[1]), int(pt[0])] == 1.0
    assert sm.mask_hint_prompt(torch.zeros(5, 5)) is None


def test_mask_hint_point_snaps_inside_ring() -> None:
    m = torch.zeros(21, 21)
    m[2:19, 2:19] = 1.0
    m[6:15, 6:15] = 0.0  # centroid sits in the hole
    _, pt = sm.mask_hint_prompt(m)
    assert m[int(pt[1]), int(pt[0])] == 1.0


def test_checkpoint_names() -> None:
    assert sm.checkpoint_path("vit_b").name == "sam_vit_b_01ec64.pth"
    assert sm.checkpoint_path("vit_l").name == "sam_vit_l_0b3195.pth"
    assert sm.checkpoint_path("vit_h").name == "sam_vit_h_4b8939.pth"
    assert sm.checkpoint_path("mobile_sam").name == "mobile_sam.pt"
    with pytest.raises(ValueError):
        sm.checkpoint_path("vit_x")


def test_sam_mask_validation_before_model() -> None:
    img = torch.rand(1, 16, 16, 3)
    with pytest.raises(TypeError):
        sm.sam_mask("x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sm.sam_mask(torch.rand(1, 16, 16, 4))
    with pytest.raises(ValueError):
        sm.sam_mask(img, model_type="vit_x")
    with pytest.raises(TypeError):
        sm.sam_mask(img, multimask=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sm.sam_mask(img, mask_threshold=99.0)
    with pytest.raises(ValueError):
        sm.sam_mask(img, points_json="[[1,")
    with pytest.raises(ValueError):
        sm.sam_mask(img, mask_hint=torch.rand(3, 16, 16))


# ----------------------------------------------------------------- real model runs


@needs_mobile
def test_cpu_fallback_point_on_subject(portrait) -> None:
    mask, score = sm.sam_mask(
        portrait, model_type="mobile_sam", points_json="[[0.5, 0.45]]", device="cpu"
    )
    assert mask.shape == (1, 768, 512) and mask.dtype == torch.float32
    assert score.shape == (1,) and 0.0 < float(score) < 1.5
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert mask.mean() > 0.02  # found an object


@needs_mobile
def test_cpu_bbox_json_from_face_detect_format(portrait, truth) -> None:
    box = json.dumps([{"x": 60, "y": 40, "w": 400, "h": 700, "conf": 0.9, "batch_index": 0}])
    mask, _ = sm.sam_mask(portrait, model_type="mobile_sam", bbox_json=box, device="cpu")
    assert _iou(mask, truth) > 0.5  # measured ~0.69
    # a box that belongs to another image index gives no box -> centre-point fallback still runs
    other = json.dumps([{"x": 60, "y": 40, "w": 400, "h": 700, "batch_index": 5}])
    m2, _ = sm.sam_mask(portrait, model_type="mobile_sam", bbox_json=other, device="cpu")
    assert m2.shape == mask.shape


@needs_mobile
def test_cpu_mask_hint_path(portrait, truth) -> None:
    hint = torch.zeros(1, 768, 512)
    ys, xs = torch.nonzero(truth[0], as_tuple=True)
    hint[0, int(ys.min()) : int(ys.max()), int(xs.min()) : int(xs.max())] = 1.0
    mask, _ = sm.sam_mask(portrait, model_type="mobile_sam", mask_hint=hint, device="cpu")
    assert _iou(mask, truth) > 0.8  # measured ~0.96


@needs_mobile
def test_cpu_negative_point_and_batch(portrait) -> None:
    batch = torch.cat([portrait, portrait.flip(2)], dim=0)
    mask, scores = sm.sam_mask(
        batch,
        model_type="mobile_sam",
        points_json="[[256, 380]]",
        negative_points_json="[[256, 100]]",
        multimask=False,
        device="cpu",
    )
    assert mask.shape == (2, 768, 512) and scores.shape == (2,)


@needs_vit_b_cuda
def test_vit_b_point_on_subject(portrait, truth) -> None:
    mask, score = sm.sam_mask(portrait, model_type="vit_b", points_json="[[0.5, 0.45]]")
    assert mask.device == portrait.device
    assert _iou(mask, truth) > 0.7  # measured ~0.84
    assert float(score) > 0.8


@needs_vit_b_cuda
def test_vit_b_cuda_peak_memory(portrait) -> None:
    sm.sam_mask(portrait, model_type="vit_b")  # load once
    torch.cuda.reset_peak_memory_stats()
    sm.sam_mask(portrait, model_type="vit_b")
    assert torch.cuda.max_memory_allocated() < 6 * 1024**3


# ----------------------------------------------------------------- v1.6.0 review fixes (no model)


def test_points_ambiguous_unit_values_are_pixels() -> None:
    assert sm.parse_points_json("[[1, 1]]", 100, 100) == [(1, 1)]
    assert sm.parse_points_json("[[0, 0]]", 100, 100) == [(0, 0)]
    assert sm.parse_points_json("[[1.0, 1.0]]", 100, 100) == [(1, 1)]
    # one real fraction makes the whole list normalised
    assert sm.parse_points_json("[[1, 0.5]]", 101, 101) == [(100.0, 50.0)]


def test_points_reject_off_by_one_edge() -> None:
    assert sm.parse_points_json("[[99, 49]]", 100, 50) == [(99, 49)]
    for text in ("[[100, 10]]", "[[10, 50]]", "[[-1, 5]]"):
        with pytest.raises(ValueError, match="outside"):
            sm.parse_points_json(text, 100, 50)


def test_sams_dir_lookup_does_not_create(tmp_path, monkeypatch) -> None:
    fake = types.SimpleNamespace(models_dir=str(tmp_path / "models"), folder_names_and_paths={})
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    path = sm.checkpoint_path("vit_b")
    assert path == tmp_path / "models" / "sams" / "sam_vit_b_01ec64.pth"
    assert not (tmp_path / "models").exists()


def test_sams_dir_honours_registered_folder(tmp_path, monkeypatch) -> None:
    extra = tmp_path / "extra_sams"
    fake = types.SimpleNamespace(
        models_dir=str(tmp_path / "models"),
        folder_names_and_paths={"sams": ([str(extra)], set())},
        get_folder_paths=lambda name: [str(extra)],
    )
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    assert sm.checkpoint_path("mobile_sam") == extra / "mobile_sam.pt"
    extra.mkdir()
    (extra / "mobile_sam.pt").write_bytes(b"x")
    assert sm.checkpoint_path("mobile_sam").exists()


class _FakeResp:
    def __init__(self, data: bytes) -> None:
        self._data, self.headers = data, {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a) -> None:
        return None

    def read(self, n: int) -> bytes:
        out, self._data = self._data[:n], self._data[n:]
        return out


def test_download_verifies_sha256_and_cleans_part(tmp_path, monkeypatch) -> None:
    data = b"checkpoint-bytes"
    monkeypatch.setattr(sm, "_sams_dirs", lambda: [tmp_path / "sams"])
    seen = {}

    def fake_open(url, timeout=None):
        seen["timeout"] = timeout
        return _FakeResp(data)

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    monkeypatch.setitem(sm._SHA256, "mobile_sam", hashlib.sha256(data).hexdigest())
    path = sm._ensure_checkpoint("mobile_sam")
    assert path.read_bytes() == data and seen["timeout"] and not list(path.parent.glob("*.part"))

    path.unlink()
    monkeypatch.setitem(sm._SHA256, "mobile_sam", "0" * 64)
    with pytest.raises(RuntimeError, match="SHA256"):
        sm._ensure_checkpoint("mobile_sam")
    assert not path.exists() and not list(path.parent.glob("*.part"))


class _CtorPrompter:
    def __init__(self, cfg, device=None, dtype=None) -> None:
        self.model = types.SimpleNamespace(eval=lambda: None)


def test_prompter_cache_keeps_one_model(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules, "kornia.contrib.models.sam", types.SimpleNamespace(SamConfig=lambda **k: k)
    )
    monkeypatch.setitem(
        sys.modules,
        "kornia.contrib.visual_prompter",
        types.SimpleNamespace(VisualPrompter=_CtorPrompter),
    )
    monkeypatch.setattr(sm, "_ensure_checkpoint", lambda t: Path(t))
    sm.unload()
    cpu = torch.device("cpu")
    a = sm._get_prompter("vit_b", cpu)
    assert sm._get_prompter("vit_b", cpu) is a
    b = sm._get_prompter("mobile_sam", cpu)
    assert b is not a and list(sm._PROMPTER_CACHE) == [("mobile_sam", "cpu")]
    sm.unload()
    assert not sm._PROMPTER_CACHE


def test_pick_mask_skips_near_full_frame() -> None:
    full = torch.ones(10, 10, dtype=torch.bool)
    small = torch.zeros(10, 10, dtype=torch.bool)
    small[:3, :3] = True
    binary = torch.stack([full, small, small])
    assert sm._pick_mask(binary, torch.tensor([0.99, 0.5, 0.6])) == 2
    assert sm._pick_mask(torch.stack([full, full]), torch.tensor([0.4, 0.9])) == 1  # all huge


class _FakeResult:
    def __init__(self, binary, scores) -> None:
        self.binary_masks, self.scores, self.mask_threshold = binary, scores, 0.0


class _FakePrompter:
    def __init__(self, binary, scores) -> None:
        self._r = _FakeResult(binary, scores)
        self.kwargs: dict = {}

    def set_image(self, img) -> None:
        pass

    def reset_image(self) -> None:
        pass

    def predict(self, **kw):
        self.kwargs = kw
        return self._r


def test_sam_mask_avoids_full_frame_candidate(monkeypatch) -> None:
    full = torch.ones(8, 8, dtype=torch.bool)
    part = torch.zeros(8, 8, dtype=torch.bool)
    part[2:5, 2:5] = True
    fake = _FakePrompter(torch.stack([full, part, part])[None], torch.tensor([[0.99, 0.7, 0.6]]))
    monkeypatch.setattr(sm, "_get_prompter", lambda t, d: fake)
    mask, score = sm.sam_mask(torch.rand(1, 8, 8, 3), points_json="[[3, 3]]", device="cpu")
    assert float(mask.sum()) == 9.0 and float(score) == pytest.approx(0.7)


def test_negative_only_warns(monkeypatch, caplog) -> None:
    part = torch.zeros(8, 8, dtype=torch.bool)
    part[:2] = True
    fake = _FakePrompter(part[None, None], torch.tensor([[0.5]]))
    monkeypatch.setattr(sm, "_get_prompter", lambda t, d: fake)
    with caplog.at_level("WARNING"):
        sm.sam_mask(torch.rand(1, 8, 8, 3), negative_points_json="[[3, 3]]", device="cpu")
    assert "negative" in caplog.text.lower()
    assert "keypoints" in fake.kwargs


def test_bbox_json_without_box_for_item_gives_empty_mask(monkeypatch) -> None:
    part = torch.zeros(8, 8, dtype=torch.bool)
    part[2:5, 2:5] = True
    fake = _FakePrompter(part[None, None], torch.tensor([[0.9]]))
    monkeypatch.setattr(sm, "_get_prompter", lambda t, d: fake)
    bbox = json.dumps([{"batch_index": 0, "x": 1, "y": 1, "w": 4, "h": 4}])
    mask, score = sm.sam_mask(torch.rand(2, 8, 8, 3), bbox_json=bbox, device="cpu")
    assert float(mask[0].sum()) == 9.0
    assert float(mask[1].sum()) == 0.0 and float(score[1]) == 0.0


def test_nan_pixels_do_not_reach_sam(monkeypatch) -> None:
    seen = {}
    part = torch.zeros(8, 8, dtype=torch.bool)
    part[:2] = True
    fake = _FakePrompter(part[None, None], torch.tensor([[0.8]]))
    fake.set_image = lambda img: seen.update(img=img)
    monkeypatch.setattr(sm, "_get_prompter", lambda t, d: fake)
    image = torch.rand(1, 8, 8, 3)
    image[0, 3, 3] = float("nan")
    _, score = sm.sam_mask(image, points_json="[[3, 3]]", device="cpu")
    assert bool(torch.isfinite(seen["img"]).all()) and float(score) == pytest.approx(0.8)
