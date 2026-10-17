"""Monocular depth estimation with Depth Anything V2 (ONNX, run directly by onnxruntime).

The model predicts *relative inverse depth*: larger value = nearer to the camera. The output of
:func:`estimate_depth` is normalised per image to ``[0, 1]`` with ``1`` = nearest. The
normalisation is per frame, so a video can flicker: the same distance may get different values in
consecutive frames.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import urllib.request
from numbers import Integral
from pathlib import Path

import torch
import torch.nn.functional as tnf

logger = logging.getLogger(__name__)

MODEL_SIZES = ("small", "base", "large")
PATCH = 14
RESOLUTION_MIN = 266
RESOLUTION_MAX = 1036
FOLDER_NAME = "depth_anything"
# Pinned Hugging Face commits (onnx-community/depth-anything-v2-<size>), not "main".
_REVISIONS = {
    "small": "4472b7362082ad9968fee890ca0f1e5aca36b93d",
    "base": "dd4557d492cd7b563738ac8d9ccff9094620983c",
    "large": "1fa1591c7b080e98da9655827c3a33a3972a4a83",
}
# SHA256 of onnx/model.onnx at the pinned revision. base / large: revision pin only.
_SHA256 = {
    "small": "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c",
}
_MODEL_URL = (
    "https://huggingface.co/onnx-community/depth-anything-v2-{size}/resolve/{revision}"
    "/onnx/model.onnx"
)
_DOWNLOAD_TIMEOUT = 60  # seconds per socket operation
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_SESSIONS: dict[tuple[str, str], object] = {}  # at most one entry
_SESSION_LOCK = threading.Lock()


def _validate_image(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")
    if not torch.is_floating_point(image):
        raise ValueError(f"image must be a floating point tensor, got {image.dtype}.")
    return torch.nan_to_num(image.detach().to(dtype=torch.float32)).clamp(0.0, 1.0)


def _validate_resolution(resolution: int) -> int:
    if isinstance(resolution, bool) or not isinstance(resolution, Integral):
        raise ValueError(f"resolution must be an integer in [{RESOLUTION_MIN}, {RESOLUTION_MAX}].")
    value = int(resolution)
    if not RESOLUTION_MIN <= value <= RESOLUTION_MAX:
        raise ValueError(f"resolution must be in [{RESOLUTION_MIN}, {RESOLUTION_MAX}].")
    return value


def network_size(height: int, width: int, resolution: int) -> tuple[int, int]:
    """Network input size: both sides multiples of 14, long side ~= ``resolution``."""
    scale = resolution / max(height, width)
    new_h = max(PATCH, int(round(height * scale / PATCH)) * PATCH)
    new_w = max(PATCH, int(round(width * scale / PATCH)) * PATCH)
    return new_h, new_w


def model_dirs() -> list[Path]:
    """Candidate model folders, best first. Never creates anything.

    Uses ``folder_paths.get_folder_paths("depth_anything")`` when that folder name is registered
    (honours ``extra_model_paths.yaml``), else ``<models_dir>/depth_anything``; without
    ``folder_paths`` it is ``<ComfyUI>/models/depth_anything`` next to the custom_nodes folder.
    """
    try:
        import folder_paths  # type: ignore
    except ImportError:
        folder_paths = None
    if folder_paths is not None:
        try:
            registered = [Path(p) for p in folder_paths.get_folder_paths(FOLDER_NAME)]
            if registered:
                return registered
        except Exception:  # noqa: BLE001 - folder name not registered (KeyError) or API differs
            pass
        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            return [Path(models_dir) / FOLDER_NAME]
    return [Path(__file__).resolve().parents[3] / "models" / FOLDER_NAME]


def model_dir() -> Path:
    """Primary model folder (where a download goes); not created here."""
    return model_dirs()[0]


def model_path(size: str) -> Path:
    """Path of the ONNX file: the first existing one across the model folders, else the primary."""
    if size not in MODEL_SIZES:
        raise ValueError(f"model must be one of {MODEL_SIZES}, got {size!r}.")
    name = f"depth-anything-v2-{size}.onnx"
    for folder in model_dirs():
        if (folder / name).is_file():
            return folder / name
    return model_dir() / name


def _download(url: str, part: Path, expected_sha256: str | None) -> None:
    """Stream ``url`` to ``part`` (socket timeout per read), checking size and SHA256."""
    digest = hashlib.sha256()
    written = 0
    with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as response:  # noqa: S310
        expected_size = response.headers.get("Content-Length")
        with open(part, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                written += len(chunk)
    if expected_size is not None and written != int(expected_size):
        raise OSError(f"incomplete download ({written} of {expected_size} bytes)")
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise OSError("SHA256 mismatch, the file is corrupt or has been changed")


def _ensure_model_file(size: str) -> Path:
    path = model_path(size)
    if path.is_file():
        return path
    url = _MODEL_URL.format(size=size, revision=_REVISIONS[size])
    part = path.with_name(path.name + ".part")
    logger.warning("Downloading Depth Anything V2 %s from %s", size, url)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)  # only right before a download
        _download(url, part, _SHA256.get(size))
        part.replace(path)
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {url} ({exc}). Download it manually and save it as {path}."
        ) from exc
    return path


def _import_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Depth Anything needs onnxruntime. Install it with 'pip install onnxruntime-gpu' "
            "(or 'pip install onnxruntime' for CPU only)."
        ) from exc
    return ort


def unload() -> None:
    """Drop the cached ONNX session (frees the CUDA arena)."""
    with _SESSION_LOCK:
        _SESSIONS.clear()


def _get_session(path: Path):
    ort = _import_onnxruntime()
    use_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"
    key = (str(path), provider)
    with _SESSION_LOCK:
        session = _SESSIONS.get(key)
        if session is not None:
            return session
        _SESSIONS.clear()  # keep at most one session (and one CUDA arena) alive
        if use_cuda:
            # kSameAsRequested stops the arena from growing in large power-of-two steps.
            cuda = ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})
            try:
                session = ort.InferenceSession(str(path), providers=[cuda, "CPUExecutionProvider"])
            except Exception as exc:  # noqa: BLE001 - any CUDA init failure -> CPU
                logger.warning("Depth Anything: CUDA provider failed (%s); using CPU.", exc)
        if session is None:
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        _SESSIONS[key] = session  # a CPU fallback is cached under the requested key too
        return session


def estimate_depth(
    image: torch.Tensor,
    model: str = "small",
    resolution: int = 518,
) -> torch.Tensor:
    """Estimate relative depth with Depth Anything V2.

    Args:
        image: IMAGE tensor ``(B,H,W,3)`` float in ``[0,1]``.
        model: ``"small"`` (Apache-2.0), ``"base"`` or ``"large"`` (CC-BY-NC-4.0).
        resolution: Long-side network size in ``[266, 1036]``; both sides are rounded to
            multiples of 14.

    Returns:
        MASK tensor ``(B,H,W)`` float32 in ``[0,1]`` on the input device, ``1`` = nearest,
        normalised per image (min->0, max->1; a constant image gives zeros). Because the
        normalisation is per frame, consecutive video frames can flicker.

    Raises:
        TypeError: If ``image`` is not a tensor.
        ValueError: On invalid shapes, dtypes or parameters.
        RuntimeError: If onnxruntime is missing or the model cannot be obtained.
    """
    prepared = _validate_image(image)
    resolution = _validate_resolution(resolution)
    if model not in MODEL_SIZES:
        raise ValueError(f"model must be one of {MODEL_SIZES}, got {model!r}.")
    _import_onnxruntime()  # fail early, before any download
    session = _get_session(_ensure_model_file(model))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    batch, height, width, _ = prepared.shape
    net_h, net_w = network_size(height, width, resolution)
    device = prepared.device
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)

    results = []
    for index in range(batch):
        nchw = prepared[index : index + 1].permute(0, 3, 1, 2)
        resized = tnf.interpolate(nchw, size=(net_h, net_w), mode="bicubic", align_corners=False)
        tensor = ((resized.clamp(0.0, 1.0) - mean) / std).contiguous()
        raw = session.run([output_name], {input_name: tensor.cpu().numpy()})[0]
        depth = torch.from_numpy(raw).to(device=device, dtype=torch.float32)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        depth = tnf.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
        low, high = depth.amin(), depth.amax()
        span = high - low
        depth = (depth - low) / span if span > 1e-8 else torch.zeros_like(depth)
        results.append(depth[:, 0].clamp(0.0, 1.0))
    return torch.cat(results, dim=0)
