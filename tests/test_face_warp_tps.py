from __future__ import annotations

import pytest
import torch

from core.face_pipeline import face_warp_delaunay
from core.face_warp_tps import _subsample, face_warp_tps

SIZE = 128


def _grid_landmarks_px(size: int = SIZE, batch: int = 1) -> torch.Tensor:
    """468 pixel-coordinate landmarks on a regular grid in the central region."""
    cols, rows = 18, 26
    xs = torch.linspace(0.25, 0.75, cols) * (size - 1)
    ys = torch.linspace(0.25, 0.75, rows) * (size - 1)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    return points.unsqueeze(0).expand(batch, -1, -1).clone()


def _dot_image(cx: float, cy: float, size: int = SIZE, sigma: float = 2.0) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    dot = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
    return dot[None, :, :, None].expand(1, size, size, 3).contiguous()


def _centroid(image: torch.Tensor) -> tuple[float, float]:
    gray = image[0, ..., 0]
    total = gray.sum()
    ys, xs = torch.meshgrid(
        torch.arange(gray.shape[0], dtype=torch.float32),
        torch.arange(gray.shape[1], dtype=torch.float32),
        indexing="ij",
    )
    return float((gray * xs).sum() / total), float((gray * ys).sum() / total)


def test_identity_returns_input() -> None:
    torch.manual_seed(0)
    image = torch.rand(1, SIZE, SIZE, 3)
    lm = _grid_landmarks_px()
    out = face_warp_tps(image, lm, lm.clone())
    assert out.shape == image.shape and out.dtype == torch.float32
    assert torch.allclose(out, image, atol=1e-4)


def test_translation_moves_dot_to_destination() -> None:
    src = _grid_landmarks_px()
    dst = src + torch.tensor([10.0, 6.0])
    image = _dot_image(64.0, 64.0)
    for anchor in (False, True):
        out = face_warp_tps(image, src, dst, anchor_border=anchor)
        cx, cy = _centroid(out)
        assert abs(cx - 74.0) < 1.0 and abs(cy - 70.0) < 1.0, (anchor, cx, cy)


def test_dot_at_single_control_point_lands_at_destination() -> None:
    """Local move of one control point: the dot must end up at that point's destination."""
    src = _grid_landmarks_px()
    dst = src.clone()
    # pick the landmark nearest the image centre and move it, plus its neighbours, together
    centre = torch.tensor([63.5, 63.5])
    near = (src[0] - centre).norm(dim=-1) < 12.0
    dst[0, near] += torch.tensor([10.0, 6.0])
    cx0, cy0 = (src[0, near].mean(dim=0)).tolist()
    out = face_warp_tps(_dot_image(cx0, cy0), src, dst)
    cx, cy = _centroid(out)
    assert abs(cx - (cx0 + 10.0)) < 1.0 and abs(cy - (cy0 + 6.0)) < 1.0, (cx, cy)


def test_normalised_landmarks_match_pixel_landmarks() -> None:
    src = _grid_landmarks_px()
    dst = src + torch.tensor([8.0, -4.0])
    image = _dot_image(64.0, 64.0)
    a = face_warp_tps(image, src, dst)
    b = face_warp_tps(image, src / (SIZE - 1), dst / (SIZE - 1))
    assert torch.allclose(a, b, atol=1e-4)


def test_border_anchors_keep_corners_fixed() -> None:
    torch.manual_seed(1)
    image = torch.rand(1, SIZE, SIZE, 3)
    src = _grid_landmarks_px()
    dst = src + torch.tensor([12.0, 9.0])
    out = face_warp_tps(image, src, dst, anchor_border=True)
    for y, x in [(0, 0), (0, SIZE - 1), (SIZE - 1, 0), (SIZE - 1, SIZE - 1)]:
        assert torch.allclose(out[0, y, x], image[0, y, x], atol=1e-3)
    # edge midpoint is pinned with anchors but dragged along without them
    mid = SIZE // 2
    free = face_warp_tps(image, src, dst, anchor_border=False)
    assert torch.allclose(out[0, 0, mid], image[0, 0, mid], atol=1e-2)
    assert not torch.allclose(free[0, 0, mid], image[0, 0, mid], atol=1e-3)


def test_smooth_ramp_stays_continuous() -> None:
    """A seam-free warp of a gentle ramp cannot introduce jumps."""
    ramp = torch.linspace(0, 1, SIZE).view(1, 1, SIZE, 1).expand(1, SIZE, SIZE, 3).contiguous()
    src = _grid_landmarks_px()
    torch.manual_seed(2)
    dst = src + torch.randn_like(src) * 1.5
    tps = face_warp_tps(ramp, src, dst)
    assert torch.isfinite(tps).all() and tps.min() >= 0 and tps.max() <= 1
    assert (tps[:, :, 1:] - tps[:, :, :-1]).abs().max().item() < 0.1
    delaunay = face_warp_delaunay(ramp, src, dst, device="cpu")
    assert delaunay.shape == tps.shape


def test_max_points_subsamples_and_keeps_result_close() -> None:
    src = _grid_landmarks_px()
    dst = src + torch.tensor([10.0, 6.0])
    out = face_warp_tps(_dot_image(64.0, 64.0), src, dst, max_points=16)
    cx, cy = _centroid(out)
    assert abs(cx - 74.0) < 1.5 and abs(cy - 70.0) < 1.5


def _oval_landmarks_px(size: int = 256) -> torch.Tensor:
    """468 points on concentric ellipses; the outermost ring (36 points) plays the jawline."""
    rings = [(0.45, 36)] + [(0.45 * k / 12, 36) for k in range(1, 12)]
    pts = []
    for radius, n in rings:
        t = torch.linspace(0, 2 * torch.pi, n + 1)[:-1]
        pts.append(
            torch.stack(
                [size / 2 + radius * size * 0.8 * t.cos(), size / 2 + radius * size * t.sin()],
                dim=-1,
            )
        )
    pts_t = torch.cat(pts)
    extra = 468 - pts_t.shape[0]
    filler = torch.stack([torch.linspace(0.3, 0.7, extra) * size] * 2, dim=-1)
    return torch.cat([pts_t, filler]).unsqueeze(0)


def test_subsample_keeps_every_moved_pair() -> None:
    src = _oval_landmarks_px()[0]
    dst = src.clone()
    dst[:36] += 6.0
    out_src, out_dst = _subsample(src.double(), dst.double(), 64)
    assert out_src.shape[0] == 64
    moved_kept = ((out_src - out_dst).norm(dim=-1) > 0.5).sum().item()
    assert moved_kept == 36
    # moved set larger than the budget: all moved pairs still kept
    out_src, _ = _subsample(src.double(), dst.double(), 16)
    assert out_src.shape[0] == 36


def test_jaw_edit_with_small_budget_lands_on_destination() -> None:
    size = 256
    src = _oval_landmarks_px(size)
    dst = src.clone()
    dst[0, :36] = src[0, :36] * 0.94 + src[0].mean(dim=0) * 0.06
    jaw_src, jaw_dst = src[0, 9], dst[0, 9]
    out = face_warp_tps(
        _dot_image(float(jaw_src[0]), float(jaw_src[1]), size=size), src, dst, max_points=32
    )
    cx, cy = _centroid(out)
    assert abs(cx - float(jaw_dst[0])) < 1.5 and abs(cy - float(jaw_dst[1])) < 1.5, (cx, cy)


def test_control_points_land_within_half_pixel_on_large_image() -> None:
    size = 512
    src = _oval_landmarks_px(size)
    dst = src.clone()
    dst[0, :36] = src[0, :36] * 0.94 + src[0].mean(dim=0) * 0.06
    for index in (0, 9, 18, 27):
        out = face_warp_tps(
            _dot_image(float(src[0, index, 0]), float(src[0, index, 1]), size=size, sigma=3.0),
            src,
            dst,
        )
        cx, cy = _centroid(out)
        assert abs(cx - float(dst[0, index, 0])) < 0.5, (index, cx)
        assert abs(cy - float(dst[0, index, 1])) < 0.5, (index, cy)


def test_one_side_without_face_returns_unchanged() -> None:
    image = torch.rand(1, 32, 32, 3)
    lm = _grid_landmarks_px(32)
    nan = torch.full_like(lm, float("nan"))
    assert torch.equal(face_warp_tps(image, nan, lm), image)
    assert torch.equal(face_warp_tps(image, lm, nan), image)
    assert torch.equal(face_warp_tps(image, nan, nan), image)


def test_fewer_than_three_valid_points_returns_unchanged() -> None:
    image = torch.rand(1, 32, 32, 3)
    src = torch.full((1, 468, 2), float("nan"))
    src[0, :2] = torch.tensor([[5.0, 5.0], [9.0, 9.0]])
    out = face_warp_tps(image, src, src + 3.0)
    assert torch.equal(out, image)


def test_non_finite_points_are_dropped() -> None:
    src = _grid_landmarks_px()
    dst = src + torch.tensor([10.0, 6.0])
    dst[0, 5] = float("nan")
    src[0, 9, 0] = float("inf")
    out = face_warp_tps(_dot_image(64.0, 64.0), src, dst)
    assert torch.isfinite(out).all()
    cx, cy = _centroid(out)
    assert abs(cx - 74.0) < 1.0 and abs(cy - 70.0) < 1.0


def test_batch_and_broadcast_landmarks() -> None:
    torch.manual_seed(3)
    image = torch.rand(3, 64, 64, 3)
    src = _grid_landmarks_px(64)
    dst = src + torch.tensor([4.0, 2.0])
    batched = face_warp_tps(image, src, dst)
    single = face_warp_tps(image[1:2], src, dst)
    assert batched.shape == image.shape
    assert torch.allclose(batched[1:2], single, atol=1e-5)


def test_non_square_image() -> None:
    image = torch.rand(1, 48, 96, 3)
    lm = _grid_landmarks_px(48)
    lm[..., 0] *= 2.0
    out = face_warp_tps(image, lm, lm.clone())
    assert out.shape == image.shape
    assert torch.allclose(out, image, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_matches_cpu() -> None:
    torch.manual_seed(4)
    image = torch.linspace(0, 1, SIZE).view(1, 1, SIZE, 1).expand(1, SIZE, SIZE, 3).contiguous()
    src = _grid_landmarks_px()
    dst = src + torch.randn_like(src)
    cpu = face_warp_tps(image, src, dst)
    gpu = face_warp_tps(image.cuda(), src, dst)
    assert gpu.device.type == "cuda"
    assert torch.allclose(cpu, gpu.cpu(), atol=1e-3)


def test_validation_errors() -> None:
    image = torch.rand(1, 32, 32, 3)
    lm = _grid_landmarks_px(32)
    with pytest.raises(TypeError):
        face_warp_tps("x", lm, lm)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        face_warp_tps(image[..., :1], lm, lm)
    with pytest.raises(ValueError):
        face_warp_tps(image.to(torch.uint8), lm, lm)
    with pytest.raises(ValueError):
        face_warp_tps(image, lm[:, :100], lm)
    with pytest.raises(ValueError):
        face_warp_tps(image, lm.expand(2, -1, -1), lm)
    for bad in (3, 469, True, 10.5):
        with pytest.raises(ValueError):
            face_warp_tps(image, lm, lm, max_points=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        face_warp_tps(image, lm, lm, anchor_border=1)  # type: ignore[arg-type]


def test_subsample_prefers_unmoved_points_near_the_edit() -> None:
    from core.face_warp_tps import _subsample

    src = torch.stack([torch.arange(100.0), torch.zeros(100)], dim=-1)
    dst = src.clone()
    dst[0, 1] = 5.0  # only point 0 moves
    kept_src, _ = _subsample(src, dst, 11)
    xs = set(kept_src[:, 0].tolist())
    assert 0.0 in xs and len(xs) == 11
    assert {1.0, 2.0, 3.0, 4.0, 5.0} <= xs  # nearest neighbours of the edit are pinned
    assert max(xs) > 90.0  # the far end stays pinned as well
