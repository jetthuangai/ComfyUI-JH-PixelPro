from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from core.face_pipeline import beauty_blend, extract_landmarks, face_warp_delaunay

WORKFLOW_PORTRAIT = Path(__file__).resolve().parents[1] / "workflows" / "sample_portrait.jpg"


def _load_portrait() -> torch.Tensor:
    array = np.array(Image.open(WORKFLOW_PORTRAIT).convert("RGB"), copy=True)
    return torch.from_numpy(array).float() / 255.0


def _grid_landmarks(batch: int = 1) -> torch.Tensor:
    xs = torch.linspace(0.2, 0.8, 18, dtype=torch.float32)
    ys = torch.linspace(0.2, 0.8, 26, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    return points.unsqueeze(0).expand(batch, -1, -1).contiguous()


def _gradient_image(size: int = 96) -> torch.Tensor:
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, 1, size, 1)
    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, size, 1, 1)
    return torch.cat(
        [
            xx.expand(1, size, size, 1),
            yy.expand(1, size, size, 1),
            ((xx + yy) / 2.0).expand(1, size, size, 1),
        ],
        dim=-1,
    )


def _make_face_landmarks(offset_x: float = 0.0, offset_y: float = 0.0) -> list[SimpleNamespace]:
    points = []
    for y in np.linspace(0.2, 0.8, 26):
        for x in np.linspace(0.2, 0.8, 18):
            points.append(SimpleNamespace(x=float(x + offset_x), y=float(y + offset_y), z=0.0))
    return points


def test_extract_landmarks_returns_expected_shape_on_real_face() -> None:
    portrait = _load_portrait().unsqueeze(0)

    landmarks, visibility = extract_landmarks(portrait, max_num_faces=1)

    assert landmarks.shape == (1, 1, 468, 2)
    assert visibility.shape == (1, 1)
    assert torch.isfinite(landmarks[0, 0]).all()
    assert visibility[0, 0].item() == 1.0


def test_extract_landmarks_returns_nan_padding_when_no_face() -> None:
    blank = torch.zeros((1, 256, 256, 3), dtype=torch.float32)

    landmarks, visibility = extract_landmarks(blank, max_num_faces=2)

    assert landmarks.shape == (1, 2, 468, 2)
    assert torch.isnan(landmarks).all()
    assert torch.count_nonzero(visibility) == 0


def test_extract_landmarks_truncates_to_468_points(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_result = SimpleNamespace(face_landmarks=[_make_face_landmarks()])
    fake_mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB="srgb"),
        Image=lambda **_kwargs: object(),
    )

    class FakeLandmarker:
        def detect(self, _image):
            return fake_result

    monkeypatch.setattr("core.face_pipeline._import_face_landmarker", lambda: fake_mp)
    monkeypatch.setattr("core.face_pipeline._ensure_model_file", lambda: Path("dummy.task"))
    monkeypatch.setattr(
        "core.face_pipeline._get_landmarker",
        lambda *_args, **_kwargs: FakeLandmarker(),
    )

    image = torch.zeros((1, 32, 32, 3), dtype=torch.float32)
    landmarks, visibility = extract_landmarks(image, max_num_faces=1)

    assert landmarks.shape == (1, 1, 468, 2)
    assert visibility[0, 0].item() == 1.0


def test_extract_landmarks_import_error_when_mediapipe_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "core.face_pipeline._import_face_landmarker",
        lambda: (_ for _ in ()).throw(ImportError("missing mediapipe")),
    )

    with pytest.raises(ImportError, match="mediapipe"):
        extract_landmarks(torch.zeros((1, 16, 16, 3), dtype=torch.float32))


def test_face_warp_delaunay_identity_invariant() -> None:
    image = _gradient_image()
    landmarks = _grid_landmarks()

    warped = face_warp_delaunay(image, landmarks, landmarks)

    assert warped.shape == image.shape
    assert torch.mean(torch.abs(warped - image)).item() < 1e-3


def test_face_warp_delaunay_changes_image_for_shifted_landmarks() -> None:
    image = _gradient_image()
    src = _grid_landmarks()
    dst = src.clone()
    dst[..., 0] = (dst[..., 0] + 0.05).clamp(0.0, 1.0)

    warped = face_warp_delaunay(image, src, dst)

    assert torch.mean(torch.abs(warped - image)).item() > 1e-3


def test_face_warp_delaunay_import_error_when_cv2_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "core.face_pipeline._import_cv2_scipy",
        lambda: (_ for _ in ()).throw(ImportError("missing cv2")),
    )

    with pytest.raises(ImportError, match="cv2"):
        face_warp_delaunay(_gradient_image(), _grid_landmarks(), _grid_landmarks())


def test_beauty_blend_mask_zero_returns_base() -> None:
    base = _gradient_image()
    retouched = 1.0 - base
    mask = torch.zeros((1, base.shape[1], base.shape[2]), dtype=torch.float32)

    blended = beauty_blend(base, retouched, mask, strength=1.0, feather=8)

    assert torch.allclose(blended, base, atol=1e-6)


def test_beauty_blend_mask_one_returns_retouched() -> None:
    base = _gradient_image()
    retouched = 1.0 - base
    mask = torch.ones((1, base.shape[1], base.shape[2]), dtype=torch.float32)

    blended = beauty_blend(base, retouched, mask, strength=1.0, feather=0)

    assert torch.allclose(blended, retouched, atol=1e-6)


def test_beauty_blend_half_strength_midpoint() -> None:
    base = _gradient_image()
    retouched = 1.0 - base
    mask = torch.ones((1, base.shape[1], base.shape[2]), dtype=torch.float32)

    blended = beauty_blend(base, retouched, mask, strength=0.5, feather=0)

    assert torch.allclose(blended, (base + retouched) / 2.0, atol=1e-6)


def test_beauty_blend_feather_preserves_mask_mass() -> None:
    base = _gradient_image(size=64)
    retouched = 1.0 - base
    mask = torch.zeros((1, 64, 64), dtype=torch.float32)
    mask[:, 16:48, 16:48] = 1.0

    feathered = beauty_blend(base, retouched, mask, strength=1.0, feather=12)
    applied_mask = ((feathered - base) / (retouched - base + 1e-6)).mean(dim=-1).clamp(0.0, 1.0)

    assert abs(applied_mask.sum().item() - mask.sum().item()) / mask.sum().item() < 0.02
