from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as func
from PIL import Image

from core import face_detect_lite as lite_module
from core import facial_align
from core.face_detect_lite import _get_detector, face_detect_lite

_ROOT = Path(__file__).resolve().parents[1]
_PORTRAIT = _ROOT / "workflows" / "sample_portrait.jpg"


@pytest.fixture(scope="module")
def _model() -> None:
    try:
        _get_detector(lite_module._pick_device(), 0.6, 0.3)
    except RuntimeError as exc:
        pytest.skip(f"YuNet weights unavailable: {exc}")


def _load(path: Path) -> torch.Tensor:
    array = np.array(Image.open(path).convert("RGB"), copy=True)
    return (torch.from_numpy(array).float() / 255.0).unsqueeze(0)


def _resized(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    out = func.interpolate(image.permute(0, 3, 1, 2), size=size, mode="bilinear", antialias=True)
    return out.permute(0, 2, 3, 1).contiguous()


def _small_portrait() -> torch.Tensor:
    return _resized(_load(_PORTRAIT), (512, 341))


def _portrait_at(height: int) -> torch.Tensor:
    return _resized(_load(_PORTRAIT), (height, round(height * 2333 / 3500)))


def test_validation_errors() -> None:
    img = torch.zeros(1, 32, 32, 3)
    with pytest.raises(TypeError):
        face_detect_lite("x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        face_detect_lite(torch.zeros(1, 3, 32, 32))
    with pytest.raises(ValueError):
        face_detect_lite(torch.zeros(1, 32, 32, 3, dtype=torch.float64))
    with pytest.raises(ValueError):
        face_detect_lite(img, mode="bogus")
    with pytest.raises(ValueError):
        face_detect_lite(img, max_faces=0)
    with pytest.raises(ValueError):
        face_detect_lite(img, confidence_threshold=0.99)
    with pytest.raises(ValueError):
        face_detect_lite(img, nms_threshold=0.0)
    with pytest.raises(ValueError):
        face_detect_lite(img, nms_threshold=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        face_detect_lite(img, detect_size=100)
    with pytest.raises(ValueError):
        face_detect_lite(img, detect_size=1056)


def test_weights_failure_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import kornia.contrib

    def boom(*_args, **_kwargs):
        raise OSError("no network")

    monkeypatch.setattr(kornia.contrib, "FaceDetector", boom)
    monkeypatch.setattr(lite_module, "_DETECTOR_CACHE", {})
    with pytest.raises(RuntimeError, match="yunet_final.pth"):
        face_detect_lite(torch.zeros(1, 32, 32, 3))


@pytest.mark.usefixtures("_model")
def test_portrait_one_face_and_mask() -> None:
    landmarks, boxes, count, mask = face_detect_lite(_small_portrait())
    assert count == 1 and len(landmarks) == 1 and len(landmarks[0]) == 5
    box = boxes[0]
    assert 0.6 <= box["conf"] <= 1.0
    assert mask.shape == (1, 512, 341) and mask.dtype == torch.float32
    assert mask.sum().item() == box["w"] * box["h"]
    assert mask[0, box["y"], box["x"]].item() == 1.0
    assert landmarks[0][0][0] < landmarks[0][1][0]


@pytest.mark.usefixtures("_model")
def test_blank_image_no_faces() -> None:
    landmarks, boxes, count, mask = face_detect_lite(torch.full((1, 256, 320, 3), 0.5))
    assert (landmarks, boxes, count) == ([], [], 0)
    assert mask.shape == (1, 256, 320) and mask.abs().sum().item() == 0.0


@pytest.mark.usefixtures("_model")
def test_multi_top_k_two_faces() -> None:
    face = _portrait_at(384)  # 384 x 256
    collage = torch.cat([face, face], dim=2)
    _, boxes, count, mask = face_detect_lite(collage, mode="multi_top_k", max_faces=2)
    assert count == 2
    assert sorted(b["x"] < 256 for b in boxes) == [False, True]
    assert face_detect_lite(collage, mode="single_largest")[2] == 1
    assert face_detect_lite(collage, mode="multi_top_k", max_faces=1)[2] == 1
    assert mask.sum().item() > 0


@pytest.mark.usefixtures("_model")
@pytest.mark.parametrize("shape", [(16, 16), (48, 48), (63, 63), (1, 200), (200, 1), (1, 1)])
def test_tiny_images_do_not_crash(shape: tuple[int, int]) -> None:
    for fill in (0.5, 0.0):
        _, boxes, count, mask = face_detect_lite(torch.full((1, *shape, 3), fill))
        assert mask.shape == (1, *shape)
        for box in boxes:
            assert box["x"] + box["w"] <= shape[1] and box["y"] + box["h"] <= shape[0]
        assert count == len(boxes)


def test_candidate_edge_box_covers_last_column_and_row() -> None:
    row = [10.2, 20.7, 99.0, 79.5] + [50.0] * 10 + [0.9]
    _, box = lite_module._to_candidate(row, width=100, height=80, batch_index=0)
    assert (box["x"], box["y"]) == (10, 20)
    assert box["x"] + box["w"] == 99 and box["y"] + box["h"] == 80
    row[2], row[3] = 140.0, 120.0  # beyond the image: clamp to exclusive bounds
    _, box = lite_module._to_candidate(row, width=100, height=80, batch_index=0)
    assert box["x"] + box["w"] == 100 and box["y"] + box["h"] == 80
    assert all(isinstance(box[k], int) for k in ("x", "y", "w", "h"))


def test_detector_cache_is_per_device_and_does_not_touch_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lite_module, "_DETECTOR_CACHE", {})
    device = lite_module._pick_device()
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    try:
        first = _get_detector(device, 0.6, 0.3)
    except RuntimeError as exc:
        pytest.skip(f"YuNet weights unavailable: {exc}")
    assert torch.equal(torch.rand(3), expected)
    second = _get_detector(device, 0.8, 0.5)
    assert second is first and len(lite_module._DETECTOR_CACHE) == 1
    assert (first.confidence_threshold, first.nms_threshold) == (0.8, 0.5)


@pytest.mark.usefixtures("_model")
def test_batch_indices_and_mask_shape() -> None:
    image = _small_portrait()
    batch = torch.cat([image, torch.full_like(image, 0.5), image], dim=0)
    landmarks, boxes, count, mask = face_detect_lite(batch)
    assert count == 2 and len(landmarks) == 2
    assert [b["batch_index"] for b in boxes] == [0, 2]
    assert mask.shape[0] == 3 and mask[1].sum().item() == 0.0 and mask[2].sum().item() > 0


@pytest.mark.usefixtures("_model")
def test_matches_n10_mediapipe_reference() -> None:
    image = _portrait_at(512)
    try:
        from core.face_detect import face_detect

        mp_landmarks, _, mp_count = face_detect(image)
    except Exception as exc:
        pytest.skip(f"MediaPipe unavailable: {exc}")
    if mp_count != 1:
        pytest.skip("MediaPipe found no face on reference image")
    landmarks, _, count, _ = face_detect_lite(image)
    assert count == 1
    mp_pts = np.array(mp_landmarks[0])
    pts = np.array(landmarks[0])
    iod = np.linalg.norm(mp_pts[0] - mp_pts[1])
    # YuNet gives eye centres, MediaPipe 33/263 are outer corners and its mouth/nose sit on the
    # mesh: on the sample portrait the worst point is ~0.26 of inter-ocular; allow 0.35.
    assert (np.linalg.norm(pts - mp_pts, axis=1) / iod).max() < 0.35


@pytest.mark.usefixtures("_model")
def test_landmarks_feed_facial_align() -> None:
    image = _small_portrait()
    landmarks, _, count, _ = face_detect_lite(image)
    assert count == 1
    aligned, inverse = facial_align(
        image.permute(0, 3, 1, 2), landmarks[0], target_size=512, padding=0.2
    )
    assert aligned.shape == (1, 3, 512, 512) and inverse.shape == (1, 3, 3)


@pytest.mark.usefixtures("_model")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpu_gpu_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    image = _small_portrait()
    gpu_lm, gpu_boxes, _, _ = face_detect_lite(image)
    monkeypatch.setattr(lite_module, "_pick_device", lambda: torch.device("cpu"))
    cpu_lm, cpu_boxes, _, _ = face_detect_lite(image)
    assert len(gpu_lm) == len(cpu_lm) == 1
    assert np.abs(np.array(gpu_lm) - np.array(cpu_lm)).max() < 3.0
    assert abs(gpu_boxes[0]["w"] - cpu_boxes[0]["w"]) <= 4
