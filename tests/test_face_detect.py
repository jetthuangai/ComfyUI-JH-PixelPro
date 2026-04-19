from __future__ import annotations

import builtins
import importlib
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from core import face_detect as public_face_detect
from core import facial_align
from core.face_detect import _get_landmarker, _import_mediapipe, _resolve_model_path

WORKFLOW_PORTRAIT = Path(__file__).resolve().parents[1] / "workflows" / "sample_portrait.jpg"
face_detect_module = importlib.import_module("core.face_detect")


def _load_portrait() -> torch.Tensor:
    array = np.array(Image.open(WORKFLOW_PORTRAIT).convert("RGB"), copy=True)
    return torch.from_numpy(array).float() / 255.0


def _make_stub_face(x0: float, y0: float, x1: float, y1: float) -> list[SimpleNamespace]:
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    landmarks = [SimpleNamespace(x=center_x, y=center_y) for _ in range(468)]
    landmarks[0] = SimpleNamespace(x=x0, y=y0)
    landmarks[10] = SimpleNamespace(x=x1, y=y1)
    landmarks[33] = SimpleNamespace(x=x0 + 0.2 * (x1 - x0), y=y0 + 0.3 * (y1 - y0))
    landmarks[263] = SimpleNamespace(x=x0 + 0.8 * (x1 - x0), y=y0 + 0.3 * (y1 - y0))
    landmarks[1] = SimpleNamespace(x=center_x, y=y0 + 0.55 * (y1 - y0))
    landmarks[61] = SimpleNamespace(x=x0 + 0.3 * (x1 - x0), y=y0 + 0.8 * (y1 - y0))
    landmarks[291] = SimpleNamespace(x=x0 + 0.7 * (x1 - x0), y=y0 + 0.8 * (y1 - y0))
    return landmarks


def test_import_mediapipe_ok() -> None:
    mp, _, face_landmarker_cls, _, _ = _import_mediapipe()
    assert mp.__version__ >= "0.10.33"
    assert face_landmarker_cls is not None


def test_single_largest_portrait() -> None:
    portrait = _load_portrait().unsqueeze(0)
    landmarks, boxes, count = public_face_detect(
        portrait,
        mode="single_largest",
        max_faces=1,
        confidence_threshold=0.5,
    )
    assert count == 1
    assert len(landmarks) == 1
    assert len(landmarks[0]) == 5
    assert len(boxes) == 1
    assert boxes[0]["w"] > 0
    assert boxes[0]["h"] > 0


def test_multi_top_k_group(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_faces = [
        _make_stub_face(0.1, 0.1, 0.9, 0.9),
        _make_stub_face(0.2, 0.2, 0.6, 0.6),
        _make_stub_face(0.4, 0.1, 0.7, 0.5),
    ]

    class FakeLandmarker:
        def detect(self, _image):
            return SimpleNamespace(face_landmarks=fake_faces)

    fake_mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB="srgb"),
        Image=lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        face_detect_module,
        "_import_mediapipe",
        lambda: (fake_mp, None, None, None, None),
    )
    monkeypatch.setattr(face_detect_module, "_ensure_model_file", lambda: Path("dummy.task"))
    monkeypatch.setattr(
        face_detect_module,
        "_get_landmarker",
        lambda *_args, **_kwargs: FakeLandmarker(),
    )

    image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    landmarks, boxes, count = face_detect_module.face_detect(
        image,
        mode="multi_top_k",
        max_faces=2,
        confidence_threshold=0.5,
    )
    assert count == 2
    assert len(landmarks) == 2
    areas = [box["w"] * box["h"] for box in boxes]
    assert areas == sorted(areas, reverse=True)


def test_confidence_gate() -> None:
    portrait = _load_portrait().unsqueeze(0)
    landmarks, _, count = public_face_detect(
        portrait,
        mode="single_largest",
        max_faces=1,
        confidence_threshold=0.85,
    )
    assert count == 1
    assert len(landmarks[0]) == 5


def test_no_face_landscape() -> None:
    image = torch.zeros((1, 512, 768, 3), dtype=torch.float32)
    landmarks, boxes, count = public_face_detect(image)
    assert count == 0
    assert landmarks == []
    assert boxes == []


def test_model_auto_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source_model = _resolve_model_path()
    if not source_model.exists():
        pytest.skip("MediaPipe model is not available on this runner.")

    target_model = tmp_path / "mediapipe" / "face_landmarker.task"
    _get_landmarker.cache_clear()
    monkeypatch.setattr(face_detect_module, "_resolve_model_path", lambda: target_model)

    def fake_urlretrieve(_url: str, destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_model, destination)
        return str(destination), None

    monkeypatch.setattr(face_detect_module.urllib.request, "urlretrieve", fake_urlretrieve)
    portrait = _load_portrait().unsqueeze(0)
    _, _, count = public_face_detect(portrait)
    assert count == 1
    assert target_model.exists()
    assert target_model.stat().st_size > 1_000_000
    _get_landmarker.cache_clear()


def test_fallback_error_no_mediapipe(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("mediapipe"):
            raise ImportError("forced failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="pip install mediapipe"):
        _import_mediapipe()


def test_bbox_clamp_to_image(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_faces = [_make_stub_face(-0.5, -0.5, 1.4, 1.3)]

    class FakeLandmarker:
        def detect(self, _image):
            return SimpleNamespace(face_landmarks=fake_faces)

    fake_mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB="srgb"),
        Image=lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        face_detect_module,
        "_import_mediapipe",
        lambda: (fake_mp, None, None, None, None),
    )
    monkeypatch.setattr(face_detect_module, "_ensure_model_file", lambda: Path("dummy.task"))
    monkeypatch.setattr(
        face_detect_module,
        "_get_landmarker",
        lambda *_args, **_kwargs: FakeLandmarker(),
    )

    _, boxes, count = face_detect_module.face_detect(
        torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    )
    assert count == 1
    box = boxes[0]
    assert 0 <= box["x"] <= 63
    assert 0 <= box["y"] <= 63
    assert box["x"] + box["w"] <= 64
    assert box["y"] + box["h"] <= 64


def test_landmarks_compatible_s06() -> None:
    portrait_bhwc = _load_portrait().unsqueeze(0)
    landmarks, _, count = public_face_detect(portrait_bhwc)
    assert count == 1
    portrait_bchw = portrait_bhwc.permute(0, 3, 1, 2)
    aligned, inverse = facial_align(portrait_bchw, landmarks[0], target_size=512, padding=0.2)
    assert aligned.shape == (1, 3, 512, 512)
    assert inverse.shape == (1, 3, 3)


def test_cpu_inference_time() -> None:
    portrait = _load_portrait().unsqueeze(0)
    public_face_detect(portrait)
    start = time.perf_counter()
    _, _, count = public_face_detect(portrait)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert count == 1
    assert elapsed_ms < 500.0
