"""Feature-based automatic image alignment (SIFT / LoFTR + RANSAC + warp)."""

from __future__ import annotations

import logging
from numbers import Integral, Real
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from kornia.geometry.ransac import RANSAC
from kornia.geometry.transform import warp_perspective

logger = logging.getLogger(__name__)

_METHODS = ("sift", "loftr")
_MODELS = ("homography", "affine")
_LOFTR_CKPT = "loftr_outdoor.ckpt"
_RANSAC_SEED = 1234
_AFFINE_ITERS = 2000
_SIFT_MATCH_RATIO = 0.9

_MODEL_CACHE: dict[tuple[str, str], object] = {}


def loftr_weights_path() -> Path:
    """Return the expected location of the cached LoFTR (outdoor) checkpoint."""

    return Path(torch.hub.get_dir()) / "checkpoints" / _LOFTR_CKPT


def _validate_image(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (B,H,W,3), got {tuple(value.shape)}.")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must be a floating point tensor, got {value.dtype}.")
    if value.shape[0] < 1 or value.shape[1] < 8 or value.shape[2] < 8:
        raise ValueError(f"{name} must have batch >= 1 and H, W >= 8, got {tuple(value.shape)}.")
    return value.to(dtype=torch.float32).clamp(0.0, 1.0)


def _validate_float(name: str, value: float, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    value_float = float(value)
    if not lower <= value_float <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_float


def _validate_int(name: str, value: int, *, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}].")
    value_int = int(value)
    if not lower <= value_int <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_int


def _feature_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fork_rng(device: torch.device):
    """Fork torch RNG so model construction does not advance the global stream."""

    return torch.random.fork_rng(devices=[device] if device.type == "cuda" else [])


def _get_sift(device: torch.device):
    key = ("sift", str(device))
    if key not in _MODEL_CACHE:
        from kornia.feature import SIFTFeature

        with _fork_rng(device):
            _MODEL_CACHE[key] = SIFTFeature(num_features=4000, device=device).eval()
    return _MODEL_CACHE[key]


def _get_loftr(device: torch.device):
    key = ("loftr", str(device))
    if key not in _MODEL_CACHE:
        from kornia.feature import LoFTR

        with _fork_rng(device):
            _MODEL_CACHE[key] = LoFTR(pretrained="outdoor").to(device).eval()
    return _MODEL_CACHE[key]


# Grey-level std below ~0.25/255 means no usable structure to match.
_MIN_TEXTURE_STD = 1e-3


def _gray_small(
    image: torch.Tensor, max_side: int, multiple: int
) -> tuple[torch.Tensor, float, float]:
    """Return grayscale ``(1,1,h,w)`` plus (sx, sy) = small/full scale factors.

    The downscale is an area resample, so pixel centres map as ``x_s = (x + 0.5) * s - 0.5``
    (see ``_scale_matrix``).
    """

    height, width = image.shape[0], image.shape[1]
    scale = min(1.0, max_side / max(height, width))
    new_h = max(multiple, int(round(height * scale / multiple)) * multiple)
    new_w = max(multiple, int(round(width * scale / multiple)) * multiple)
    chw = image.permute(2, 0, 1).unsqueeze(0)
    gray = (0.299 * chw[:, 0:1] + 0.587 * chw[:, 1:2] + 0.114 * chw[:, 2:3]).contiguous()
    if (new_h, new_w) != (height, width):
        gray = F.interpolate(gray, size=(new_h, new_w), mode="area")
    return gray, new_w / width, new_h / height


def _sift_features(gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model = _get_sift(gray.device)
    lafs, _, descs = model(gray)
    return lafs[0, :, :, 2], descs[0]


def _match_sift(
    feats0: tuple[torch.Tensor, torch.Tensor], feats1: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    from kornia.feature import match_smnn

    kp0, d0 = feats0
    kp1, d1 = feats1
    if d0.shape[0] < 2 or d1.shape[0] < 2:
        empty = kp0.new_zeros((0, 2))
        return empty, empty
    _, idx = match_smnn(d0, d1, _SIFT_MATCH_RATIO)
    return kp0[idx[:, 0]], kp1[idx[:, 1]]


def _match_loftr(gray0: torch.Tensor, gray1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model = _get_loftr(gray0.device)
    out = model({"image0": gray0, "image1": gray1})
    return out["keypoints0"], out["keypoints1"]


def _fit_homography(pts0: torch.Tensor, pts1: torch.Tensor, threshold: float):
    # Kornia compares the SQUARED transfer error with inl_th.
    ransac = RANSAC(model_type="homography", inl_th=threshold**2, batch_size=512, max_iter=10)
    with torch.random.fork_rng(devices=[pts0.device] if pts0.is_cuda else []):
        torch.manual_seed(_RANSAC_SEED)
        model, inliers = ransac(pts0, pts1)
    mask = inliers.detach().cpu().numpy().astype(bool)
    count = int(mask.sum())
    matrix = model.detach().to("cpu", torch.float64).numpy()
    if count >= 4:  # least-squares refit on all inliers in float64 (cv2 method 0: no RNG)
        src = pts0.detach().cpu().numpy().astype(np.float64)[mask]
        dst = pts1.detach().cpu().numpy().astype(np.float64)[mask]
        refit, _ = cv2.findHomography(src, dst, 0)
        if refit is not None and np.isfinite(refit).all():
            matrix = refit
    if not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-12:
        return None, 0
    return matrix / matrix[2, 2], count


def _solve_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Least-squares 2x3 affine mapping ``src`` -> ``dst`` (float64)."""

    design = np.hstack([src, np.ones((src.shape[0], 1))])
    sol, *_ = np.linalg.lstsq(design, dst, rcond=None)
    return sol.T if np.isfinite(sol).all() else None


def _fit_affine(pts0: torch.Tensor, pts1: torch.Tensor, threshold: float):
    """Seeded numpy RANSAC (private RNG: no OpenCV/torch/numpy global state) + LSQ refit."""

    src = pts0.detach().cpu().numpy().astype(np.float64)
    dst = pts1.detach().cpu().numpy().astype(np.float64)
    n = src.shape[0]
    homog = np.hstack([src, np.ones((n, 1))])
    rng = np.random.default_rng(_RANSAC_SEED)
    idx = rng.integers(0, n, size=(_AFFINE_ITERS, 3))  # duplicates are singular -> dropped below
    design = homog[idx]  # (T,3,3)
    ok = np.abs(np.linalg.det(design)) > 1e-9
    if not ok.any():
        return None, 0
    models = np.linalg.solve(design[ok], dst[idx][ok])  # (T',3,2): dst = [x y 1] @ M
    err = np.linalg.norm(np.einsum("nk,tkc->tnc", homog, models) - dst[None], axis=2)
    inl = err <= threshold
    best = inl[int(inl.sum(axis=1).argmax())]
    affine = None
    for _ in range(3):  # refit on inliers, re-evaluate, until stable
        if best.sum() < 3:
            return None, 0
        affine = _solve_affine(src[best], dst[best])
        if affine is None:
            return None, 0
        new = np.linalg.norm(homog @ affine.T - dst, axis=1) <= threshold
        if new.sum() < 3 or (new == best).all():
            break
        best = new
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2] = affine
    return matrix, int(best.sum())


def _scale_matrix(sx: float, sy: float) -> np.ndarray:
    """Full -> resampled pixel coords for an area resample: ``x_s = (x + 0.5) * s - 0.5``."""

    return np.array([[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5], [0.0, 0.0, 1.0]])


def auto_align(
    image: torch.Tensor,
    reference: torch.Tensor,
    *,
    method: str = "sift",
    model: str = "homography",
    max_side: int = 1024,
    ransac_threshold: float = 3.0,
    min_inliers: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, list, list[int]]:
    """Align ``image`` onto ``reference`` using feature matching, RANSAC and a warp.

    Args:
        image: IMAGE tensor ``(B,H,W,3)`` float in ``[0,1]`` to be warped.
        reference: IMAGE tensor ``(1 or B,Hr,Wr,3)``; the output has this size.
        method: ``"sift"`` (Kornia SIFT + mutual-NN matching, no weights) or ``"loftr"``
            (LoFTR outdoor, downloads ~45 MB on first use).
        model: ``"homography"`` (8 DoF) or ``"affine"`` (6 DoF, robust to wide baselines).
        max_side: Longest side used for feature extraction, in ``[64, 4096]``.
        ransac_threshold: Inlier reprojection threshold in matching-resolution pixels.
        min_inliers: Minimum inliers to accept the fit; otherwise a pure-scale fallback
            (image resized to reference size) is returned with a warning.

    Returns:
        ``(aligned, valid_mask, matrices, inliers)``: aligned ``(B,Hr,Wr,3)`` float32,
        valid_mask ``(B,Hr,Wr)`` float32 (warped coverage), per-item 3x3 matrices (nested
        lists, image -> reference pixel coordinates at full resolution), inlier counts.

    Raises:
        TypeError: If inputs are not tensors.
        ValueError: If shapes, dtypes, batches or parameters are invalid.
    """

    img = _validate_image("image", image)
    ref = _validate_image("reference", reference)
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}.")
    if model not in _MODELS:
        raise ValueError(f"model must be one of {_MODELS}, got {model!r}.")
    max_side = _validate_int("max_side", max_side, lower=64, upper=4096)
    threshold = _validate_float("ransac_threshold", ransac_threshold, lower=0.1, upper=100.0)
    min_inliers = _validate_int("min_inliers", min_inliers, lower=4, upper=10000)
    batch = img.shape[0]
    if ref.shape[0] not in (1, batch):
        raise ValueError("reference batch must match image batch or be 1.")

    out_device = img.device
    ref_h, ref_w = ref.shape[1], ref.shape[2]
    feat_device = _feature_device()
    multiple = 8 if method == "loftr" else 1
    min_pts = 4 if model == "homography" else 3

    grays_ref: dict[int, tuple[torch.Tensor, float, float]] = {}
    feats_ref: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    matrices: list[np.ndarray] = []
    counts: list[int] = []
    fallbacks: list[bool] = []

    with torch.no_grad():
        for b in range(batch):
            rb = 0 if ref.shape[0] == 1 else b
            matrix, count, fallback = None, 0, False
            try:
                if rb not in grays_ref:
                    g, sxr, syr = _gray_small(ref[rb].to(feat_device), max_side, multiple)
                    if method == "sift":
                        feats_ref[rb] = _sift_features(g)
                    grays_ref[rb] = (g, sxr, syr)
                g_ref, sxr, syr = grays_ref[rb]
                g_img, sxi, syi = _gray_small(img[b].to(feat_device), max_side, multiple)
                if min(float(g_img.std()), float(g_ref.std())) < _MIN_TEXTURE_STD:
                    # Featureless input: detectors still return keypoints on rounding noise
                    # (platform dependent), which RANSAC can "fit". Treat as no matches.
                    pts0 = pts1 = torch.empty(0, 2)
                elif method == "sift":
                    pts0, pts1 = _match_sift(_sift_features(g_img), feats_ref[rb])
                else:
                    pts0, pts1 = _match_loftr(g_img, g_ref)
                if pts0.shape[0] >= max(min_pts, min_inliers):
                    fit = _fit_homography if model == "homography" else _fit_affine
                    small, count = fit(pts0.float(), pts1.float(), threshold)
                    if small is not None and count >= min_inliers:
                        matrix = (
                            np.linalg.inv(_scale_matrix(sxr, syr)) @ small @ _scale_matrix(sxi, syi)
                        )
                else:
                    count = int(pts0.shape[0])
            except torch.OutOfMemoryError:
                raise  # a resize fallback would silently hide a resource problem
            except (RuntimeError, cv2.error, ValueError) as exc:
                logger.warning("auto_align: item %d feature/RANSAC step failed: %s", b, exc)
                matrix = None

            if matrix is None:
                logger.warning(
                    "auto_align: item %d has too few reliable matches (%d < %d); "
                    "falling back to resize.",
                    b,
                    count,
                    min_inliers,
                )
                # Same pixel-centre convention as the bilinear (align_corners=False) resize below.
                matrix = _scale_matrix(ref_w / img.shape[2], ref_h / img.shape[1])
                fallback = True
            matrices.append(matrix)
            counts.append(count)
            fallbacks.append(fallback)

        mats = torch.from_numpy(np.stack(matrices)).to(device=out_device, dtype=torch.float32)
        src = img.permute(0, 3, 1, 2)
        aligned = warp_perspective(
            src, mats, (ref_h, ref_w), mode="bilinear", padding_mode="zeros", align_corners=True
        )
        coverage = warp_perspective(
            torch.ones_like(src[:, :1]),
            mats,
            (ref_h, ref_w),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0]
        for b, is_fallback in enumerate(fallbacks):
            if is_fallback:  # plain resize: no border bleed from the zero-padded warp
                aligned[b] = F.interpolate(
                    src[b : b + 1], size=(ref_h, ref_w), mode="bilinear", align_corners=False
                )[0]
                coverage[b] = 1.0

    return (
        aligned.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0),
        coverage.clamp(0.0, 1.0),
        [m.tolist() for m in matrices],
        counts,
    )
