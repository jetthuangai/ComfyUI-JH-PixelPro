"""Auto Upright: straighten horizons and fix converging verticals from the photo's lines.

1. Line segments are found with OpenCV's LSD on a grey proxy (long edge <= 1024 px).
2. ``level``: the roll angle is the length-weighted median deviation of near-horizontal
   and near-vertical segments; the photo is rotated about its centre.
3. ``vertical``/``full``: vanishing points are found by RANSAC over segment pairs and
   refined as the smallest eigenvector of ``sum(w * l l^T)`` (homogeneous lines ``l``).
   The camera is virtually rotated so the vertical vanishing point goes to infinity
   (verticals become parallel); ``full`` also sends the main horizontal vanishing point to
   infinity (that facade faces the camera). The warp is ``H = K R K^-1``; the focal length
   comes from the two vanishing points when they are orthogonal, else a default.
4. A confidence gate (enough lines, enough total length, enough agreement) returns the
   photo untouched with confidence 0 when the evidence is weak. Corrections larger than
   ``max_angle`` are applied only up to ``max_angle`` (a steep look-up shot is
   straightened part of the way instead of being stretched beyond recognition).

All geometry is float64 numpy; the warp runs in torch (``kornia.geometry``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real

import cv2
import numpy as np
import torch
from kornia.geometry.transform import warp_perspective

MODES = ("auto", "level", "vertical", "full")
CROPS = ("none", "valid")
PROXY_SIDE = 1024
_MIN_LENGTH = 0.03  # segment length as a fraction of the proxy's long edge
_CLASS_ANGLE = 30.0  # degrees from vertical/horizontal for a segment to count as such
_LEVEL_TOLERANCE = 1.5  # degrees: segments within this of the median agree
_VP_TOLERANCE = 1.5  # degrees between a segment and the direction to its vanishing point
_RANSAC_ITERATIONS = 400
_MIN_SEGMENTS = 3
_MIN_TOTAL_LENGTH = 0.25  # of the long edge
_MIN_VP_SEGMENTS = 5  # a vanishing point needs more evidence than a level angle
_MIN_VP_LENGTH = 0.5  # inlier length, of the long edge
_MAX_VERTICAL_ROLL = 15.0  # degrees: "verticals" leaning more than this are not walls
_MIN_CONSENSUS = 0.6
_DEFAULT_FOCAL = 1.0  # times the long edge (about a 36 mm lens on full frame)
_FOCAL_RANGE = (0.4, 4.0)


@dataclass
class _Segments:
    p1: np.ndarray  # (N,2) full-resolution pixel coordinates
    p2: np.ndarray
    length: np.ndarray  # (N,)
    angle: np.ndarray  # (N,) degrees in (-90, 90], 0 = horizontal, y down


def _check(image: torch.Tensor, mode: str, strength: float, max_angle: float, crop: str) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")
    if not torch.is_floating_point(image):
        raise ValueError(f"image must be floating point, got {image.dtype}.")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")
    if crop not in CROPS:
        raise ValueError(f"crop must be one of {CROPS}, got {crop!r}.")
    for name, value, upper in (("strength", strength, 1.0), ("max_angle", max_angle, 45.0)):
        if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= upper:
            raise ValueError(f"{name} must be in [0, {upper}].")


def detect_segments(frame: torch.Tensor) -> _Segments:
    """LSD segments of one ``(H,W,3)`` frame, in full-resolution coordinates."""
    height, width = frame.shape[:2]
    scale = min(1.0, PROXY_SIDE / max(height, width))
    grey = frame.float().cpu().clamp(0, 1) @ frame.new_tensor([0.299, 0.587, 0.114]).cpu()
    grey_np = (grey.numpy() * 255.0).astype(np.uint8)
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        grey_np = cv2.resize(grey_np, size, interpolation=cv2.INTER_AREA)
    found = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(grey_np)[0]
    lines = np.zeros((0, 4)) if found is None else found.reshape(-1, 4).astype(np.float64)
    lines = (lines + 0.5) / scale - 0.5
    p1, p2 = lines[:, :2], lines[:, 2:]
    delta = p2 - p1
    length = np.hypot(delta[:, 0], delta[:, 1])
    keep = length >= _MIN_LENGTH * max(height, width)
    p1, p2, delta, length = p1[keep], p2[keep], delta[keep], length[keep]
    angle = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
    angle = np.where(angle <= -90.0, angle + 180.0, np.where(angle > 90.0, angle - 180.0, angle))
    return _Segments(p1, p2, length, angle)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _gate(count: int, total: float, agree: float, long_edge: int) -> bool:
    return (
        count >= _MIN_SEGMENTS
        and total >= _MIN_TOTAL_LENGTH * long_edge
        and agree >= _MIN_CONSENSUS * total
    )


def estimate_roll(segments: _Segments, long_edge: int) -> tuple[float, float]:
    """Roll in degrees (positive = the photo is rotated clockwise) and confidence 0..1."""
    horizontal = np.abs(segments.angle) < _CLASS_ANGLE
    vertical = np.abs(segments.angle) > 90.0 - _CLASS_ANGLE
    deviation = np.where(vertical, segments.angle - np.sign(segments.angle) * 90.0, segments.angle)
    use = horizontal | vertical
    if not use.any():
        return 0.0, 0.0
    dev, weight = deviation[use], segments.length[use]
    roll = _weighted_median(dev, weight)
    agree = float(weight[np.abs(dev - roll) <= _LEVEL_TOLERANCE].sum())
    total = float(weight.sum())
    if not _gate(int(use.sum()), total, agree, long_edge):
        return 0.0, 0.0
    # Refine with the mean of the agreeing segments.
    inside = np.abs(dev - roll) <= _LEVEL_TOLERANCE
    roll = float(np.average(dev[inside], weights=weight[inside]))
    return roll, agree / total


def _homogeneous_lines(segments: _Segments, select: np.ndarray, norm: np.ndarray) -> np.ndarray:
    """Lines ``l`` (N,3) in normalised coordinates with ``(l0, l1)`` of unit length."""
    ones = np.ones((int(select.sum()), 1))
    a = np.hstack([(segments.p1[select] - norm[:2]) / norm[2], ones])
    b = np.hstack([(segments.p2[select] - norm[:2]) / norm[2], ones])
    lines = np.cross(a, b)
    return lines / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)


def _segment_vp_error(segments: _Segments, select: np.ndarray, norm: np.ndarray, vp: np.ndarray):
    """Angle (degrees) between each segment and the direction from its middle to ``vp``."""
    mid = ((segments.p1[select] + segments.p2[select]) / 2.0 - norm[:2]) / norm[2]
    direction = segments.p2[select] - segments.p1[select]
    to_vp = vp[:2][None, :] - mid * vp[2]  # works for finite and infinite vanishing points
    cos = np.abs((direction * to_vp).sum(1)) / (
        np.linalg.norm(direction, axis=1) * np.linalg.norm(to_vp, axis=1) + 1e-12
    )
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))


def find_vanishing_point(
    segments: _Segments, select: np.ndarray, norm: np.ndarray, long_edge: int, seed: int = 0
) -> tuple[np.ndarray | None, float]:
    """RANSAC vanishing point (homogeneous, normalised coordinates) and confidence 0..1."""
    count = int(select.sum())
    if count < _MIN_SEGMENTS:
        return None, 0.0
    lines = _homogeneous_lines(segments, select, norm)
    weight = segments.length[select]
    rng = np.random.default_rng(seed)
    best, best_score = None, -1.0
    pairs = rng.integers(0, count, size=(_RANSAC_ITERATIONS, 2))
    for i, j in pairs:
        if i == j:
            continue
        vp = np.cross(lines[i], lines[j])
        if np.linalg.norm(vp) < 1e-12:
            continue
        vp = vp / np.linalg.norm(vp)
        inliers = _segment_vp_error(segments, select, norm, vp) <= _VP_TOLERANCE
        score = float(weight[inliers].sum())
        if score > best_score:
            best, best_score = vp, score
    if best is None:
        return None, 0.0
    inliers = _segment_vp_error(segments, select, norm, best) <= _VP_TOLERANCE
    if (
        int(inliers.sum()) < _MIN_VP_SEGMENTS
        or float(weight[inliers].sum()) < _MIN_VP_LENGTH * long_edge
    ):
        return None, 0.0
    # Refine: the point closest to all inlier lines (smallest eigenvector of sum w l l^T).
    moment = (lines[inliers] * weight[inliers, None]).T @ lines[inliers]
    vp = np.linalg.eigh(moment)[1][:, 0]
    total = float(weight.sum())
    agree = float(weight[_segment_vp_error(segments, select, norm, vp) <= _VP_TOLERANCE].sum())
    if not _gate(int(inliers.sum()), total, agree, long_edge):
        return None, 0.0
    return vp, agree / total


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector ``a`` to unit vector ``b`` (Rodrigues)."""
    axis = np.cross(a, b)
    sin, cos = np.linalg.norm(axis), float(np.dot(a, b))
    if sin < 1e-12:
        return np.eye(3)
    k = axis / sin
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    angle = math.atan2(sin, cos)
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def _scale_rotation(rotation: np.ndarray, amount: float) -> tuple[np.ndarray, float]:
    """``rotation`` raised to ``amount`` (axis-angle scaling) and its full angle in degrees."""
    vector, _ = cv2.Rodrigues(rotation)
    angle = float(np.degrees(np.linalg.norm(vector)))
    scaled, _ = cv2.Rodrigues(vector * amount)
    return scaled, angle


def _vp_direction(vp: np.ndarray, norm: np.ndarray, focal: float) -> np.ndarray:
    """Camera-space direction of a vanishing point given in normalised coordinates."""
    d = np.array([vp[0] * norm[2], vp[1] * norm[2], vp[2] * focal])
    return d / np.linalg.norm(d)


def _focal_from_vps(v1: np.ndarray, v2: np.ndarray, norm: np.ndarray, long_edge: int) -> float:
    """Focal from two orthogonal finite vanishing points, else the default."""
    default = _DEFAULT_FOCAL * long_edge
    if abs(v1[2]) < 1e-6 or abs(v2[2]) < 1e-6:
        return default
    a = v1[:2] / v1[2] * norm[2]
    b = v2[:2] / v2[2] * norm[2]
    f2 = -float(np.dot(a, b))
    if f2 <= 0:
        return default
    focal = math.sqrt(f2)
    lo, hi = _FOCAL_RANGE
    return focal if lo * long_edge <= focal <= hi * long_edge else default


def _vertical_rotation(
    segments: _Segments, norm: np.ndarray, long_edge: int, full: bool
) -> tuple[np.ndarray | None, float, float]:
    """Rotation making verticals parallel (``full``: also the main horizontals), its
    confidence and the focal length used."""
    focal = _DEFAULT_FOCAL * long_edge
    vertical = np.abs(segments.angle) > 90.0 - _CLASS_ANGLE
    vp_v, confidence = find_vanishing_point(segments, vertical, norm, long_edge)
    if vp_v is None:
        return None, 0.0, focal
    vp_h, conf_h = None, 0.0
    if full:
        horizontal = np.abs(segments.angle) < _CLASS_ANGLE
        vp_h, conf_h = find_vanishing_point(segments, horizontal, norm, long_edge, seed=1)
        if vp_h is not None:
            focal = _focal_from_vps(vp_v, vp_h, norm, long_edge)
    d_v = _vp_direction(vp_v, norm, focal)
    if d_v[1] < 0:
        d_v = -d_v
    if abs(math.degrees(math.atan2(d_v[0], d_v[1]))) > _MAX_VERTICAL_ROLL:
        return None, 0.0, focal  # lines leaning that much are not the walls of a building
    rotation = _rotation_between(d_v, np.array([0.0, 1.0, 0.0]))
    if vp_h is not None:
        d_h = rotation @ _vp_direction(vp_h, norm, focal)
        d_h[1] = 0.0  # verticals are done; only turn about the vertical axis
        if np.linalg.norm(d_h) > 1e-9:
            d_h /= np.linalg.norm(d_h)
            if d_h[0] < 0:
                d_h = -d_h
            rotation = _rotation_between(d_h, np.array([1.0, 0.0, 0.0])) @ rotation
            confidence = min(confidence, conf_h)
    return rotation, confidence, focal


def _roll_rotation(roll: float) -> np.ndarray:
    angle = math.radians(roll)
    return np.array(
        [
            [math.cos(angle), math.sin(angle), 0.0],
            [-math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def upright_matrix(
    frame: torch.Tensor, mode: str, strength: float, max_angle: float
) -> tuple[np.ndarray, float]:
    """3x3 pixel homography straightening ``frame`` (H,W,3) and its confidence 0..1.

    This is the raw camera rotation ``K R K^-1``; ``reframe`` places it in the output.
    """
    height, width = frame.shape[:2]
    long_edge = max(height, width)
    segments = detect_segments(frame)
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    norm = np.array([center[0], center[1], float(long_edge)])

    rotation, confidence, focal = None, 0.0, _DEFAULT_FOCAL * long_edge
    if mode != "level":
        rotation, confidence, focal = _vertical_rotation(
            segments, norm, long_edge, full=mode == "full"
        )
    if rotation is None and mode in ("auto", "level"):
        roll, confidence = estimate_roll(segments, long_edge)
        rotation = _roll_rotation(roll) if confidence > 0 else None
    if rotation is None:
        return np.eye(3), 0.0
    _, angle = _scale_rotation(rotation, 1.0)
    amount = float(strength)
    if angle * amount > max_angle:
        amount = max_angle / angle
    rotation, _ = _scale_rotation(rotation, amount)
    k = np.array([[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]])
    homography = k @ rotation @ np.linalg.inv(k)
    return homography / homography[2, 2], confidence


def _apply(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    mapped = np.hstack([points, np.ones((len(points), 1))]) @ homography.T
    return mapped[:, :2] / mapped[:, 2:]


def _inside(points: np.ndarray, quad: np.ndarray) -> bool:
    """True if every point lies inside the convex quad (corners in order)."""
    x, y = quad[:, 0], quad[:, 1]
    sign = 1.0 if float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) > 0 else -1.0
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        cross = (b[0] - a[0]) * (points[:, 1] - a[1]) - (b[1] - a[1]) * (points[:, 0] - a[0])
        if np.any(cross * sign < -1e-6):
            return False
    return True


def _largest_rect(quad: np.ndarray, aspect: float) -> tuple[float, float, float] | None:
    """Centre and half-height of the largest ``aspect`` (w/h) rectangle inside ``quad``."""
    lo_xy, hi_xy = quad.min(0), quad.max(0)
    unit = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * np.array([aspect, 1.0])
    best = None
    for cx in np.linspace(lo_xy[0], hi_xy[0], 11)[1:-1]:
        for cy in np.linspace(lo_xy[1], hi_xy[1], 11)[1:-1]:
            if not _inside(np.array([[cx, cy]]), quad):
                continue
            lo, hi = 0.0, float(hi_xy[1] - lo_xy[1])
            for _ in range(30):
                mid = (lo + hi) / 2.0
                if _inside(unit * mid + [cx, cy], quad):
                    lo = mid
                else:
                    hi = mid
            if best is None or lo > best[2]:
                best = (float(cx), float(cy), lo)
    return best


def reframe(homography: np.ndarray, height: int, width: int, crop: str) -> np.ndarray:
    """Place the straightened photo in the output frame.

    ``none``: the original centre stays in the centre at the same magnification (empty
    corners are black). ``valid``: zoom to the largest rectangle of the original aspect
    that holds only image pixels.
    """
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    if crop == "valid":
        corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        mapped = np.hstack([corners, np.ones((4, 1))]) @ homography.T
        rect = None
        if np.all(mapped[:, 2] > 0):
            quad = mapped[:, :2] / mapped[:, 2:]
            rect = _largest_rect(quad, (width - 1) / max(height - 1, 1))
        if rect is not None and rect[2] > 0:
            rx, ry, half_h = rect
            zoom = (height - 1) / (2.0 * half_h)
            place = np.array([[zoom, 0.0, cx - zoom * rx], [0.0, zoom, cy - zoom * ry], [0, 0, 1]])
            return place @ homography
        # No clean rectangle (extreme warp): fall back to the centred placement below.
    # Keep the local magnification at the centre (area scale of H there) and re-centre.
    step = 0.005 * min(height, width)
    probe = _apply(homography, np.array([[cx, cy], [cx + step, cy], [cx, cy + step]]))
    u, v = probe[1] - probe[0], probe[2] - probe[0]
    area = abs(u[0] * v[1] - u[1] * v[0]) / step**2
    scale = 1.0 / math.sqrt(max(area, 1e-9))
    place = np.array(
        [[scale, 0.0, cx - scale * probe[0, 0]], [0.0, scale, cy - scale * probe[0, 1]], [0, 0, 1]]
    )
    return place @ homography


def auto_upright(
    image: torch.Tensor,
    *,
    mode: str = "auto",
    strength: float = 1.0,
    max_angle: float = 25.0,
    crop: str = "none",
) -> tuple[torch.Tensor, list[float], list[list[list[float]]]]:
    """Straighten each image of a batch.

    Args:
        image: IMAGE ``(B,H,W,3)``.
        mode: ``auto`` (verticals if found, else level), ``level`` (rotation only),
            ``vertical`` (parallel verticals), ``full`` (verticals and the main
            horizontal lines).
        strength: 0..1 fraction of the correction.
        max_angle: Largest camera rotation in degrees; bigger corrections stop there.
        crop: ``none`` (empty corners are black) or ``valid`` (zoom in until none remain).

    Returns:
        ``(images, confidences, matrices)``: corrected batch, per-image confidence (0 =
        untouched) and the 3x3 pixel homographies applied.
    """
    _check(image, mode, strength, max_angle, crop)
    outputs, confidences, matrices = [], [], []
    height, width = image.shape[1:3]
    for frame in image:
        homography, confidence = upright_matrix(frame, mode, float(strength), float(max_angle))
        if confidence <= 0 or np.allclose(homography, np.eye(3), atol=1e-9):
            outputs.append(frame.float().clamp(0.0, 1.0))
            confidences.append(0.0)
            matrices.append(np.eye(3).tolist())
            continue
        homography = reframe(homography, height, width, crop)
        homography = homography / homography[2, 2]
        src = frame.float().permute(2, 0, 1).unsqueeze(0)
        matrix = torch.from_numpy(homography).to(device=src.device, dtype=torch.float32)
        warped = warp_perspective(
            src,
            matrix.unsqueeze(0),
            (height, width),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        outputs.append(warped[0].permute(1, 2, 0).clamp(0.0, 1.0))
        confidences.append(round(confidence, 3))
        matrices.append(homography.tolist())
    return torch.stack(outputs), confidences, matrices


def matrices_to_json(matrices: list[list[list[float]]]) -> str:
    return json.dumps([[[round(v, 6) for v in row] for row in m] for m in matrices])
