"""SAM One-Click: Segment Anything (Kornia) driven by points, a box or a rough mask.

ComfyUI has no click widget, so the "click" is text: positive / negative points as JSON,
a box from Face Detect's ``bbox_json``, or a rough MASK (its bounding box plus a point
inside it). The prompts are turned into Kornia ``VisualPrompter`` inputs and the best of
SAM's candidate masks (by predicted IoU) is returned.

Checkpoints live in ``<ComfyUI>/models/sams/`` (or wherever ``folder_paths`` maps ``sams``)
and are downloaded on first use. Only one model is kept loaded; see ``unload()``.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import urllib.request
from numbers import Real
from pathlib import Path

import torch
from kornia.geometry.boxes import Boxes
from torch.nn import functional as nnf

logger = logging.getLogger(__name__)

MODEL_TYPES: tuple[str, ...] = ("vit_b", "mobile_sam", "vit_l", "vit_h")

# type -> (file name, download URL). Same names/URLs as kornia's SamConfig(pretrained=True).
_CHECKPOINTS: dict[str, tuple[str, str]] = {
    "vit_b": (
        "sam_vit_b_01ec64.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    ),
    "vit_l": (
        "sam_vit_l_0b3195.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    ),
    "vit_h": (
        "sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    ),
    "mobile_sam": (
        "mobile_sam.pt",
        "https://github.com/ChaoningZhang/MobileSAM/raw/"
        "a509aac54fdd7af59f843135f2f7cee307283c88/weights/mobile_sam.pt",
    ),
}

# SHA256 of the official checkpoints we could verify locally. vit_l / vit_h are not listed: the
# upstream repo publishes no SHA256 (the hex suffix in the file name is not a SHA256 prefix), so
# those two downloads are only checked for completeness (HTTP length), not for content.
_SHA256: dict[str, str] = {
    "vit_b": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    "mobile_sam": "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
}

_DOWNLOAD_TIMEOUT_S = 60.0  # per socket operation (connect / each read), not for the whole file
_CHUNK = 1 << 20

# Only ONE model stays loaded: SAM vit_h alone needs ~2.5 GB of VRAM, so keeping several
# alive would grow without bound. Key = (model_type, device string).
_PROMPTER_CACHE: dict[tuple[str, str], object] = {}


# --------------------------------------------------------------------------- model files


def _folder_paths():  # noqa: ANN202 — optional ComfyUI module
    try:
        import folder_paths  # type: ignore
    except ImportError:
        return None
    return folder_paths


def _sams_dirs() -> list[Path]:
    """Folders searched for SAM checkpoints, first = where downloads go. Creates nothing.

    Uses ComfyUI's ``folder_paths`` (so ``extra_model_paths.yaml`` is honoured) when the
    ``sams`` folder type is registered (other packs register it), else ``<models_dir>/sams``.
    Without ComfyUI (tests, scripts) it is ``<ComfyUI root>/models/sams``.
    """
    fp = _folder_paths()
    if fp is None:
        return [Path(__file__).resolve().parents[3] / "models" / "sams"]
    if "sams" in getattr(fp, "folder_names_and_paths", {}):
        paths = [Path(p) for p in fp.get_folder_paths("sams")]
        if paths:
            return paths
    return [Path(fp.models_dir) / "sams"]


def checkpoint_path(model_type: str) -> Path:
    """Local path of the checkpoint for ``model_type`` (may not exist yet). Creates nothing."""
    if model_type not in _CHECKPOINTS:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}.")
    name = _CHECKPOINTS[model_type][0]
    dirs = _sams_dirs()
    return next((d / name for d in dirs if (d / name).exists()), dirs[0] / name)


def _download(url: str, part: Path, expected_sha256: str | None) -> None:
    """Stream ``url`` into ``part`` with a socket timeout, verifying length and SHA256."""
    digest = hashlib.sha256()
    written = 0
    with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as resp, open(part, "wb") as out:  # noqa: S310
        total = int(resp.headers.get("Content-Length") or 0)
        while chunk := resp.read(_CHUNK):
            out.write(chunk)
            digest.update(chunk)
            written += len(chunk)
    if total and written != total:
        raise OSError(f"incomplete download ({written} of {total} bytes)")
    if expected_sha256 and digest.hexdigest() != expected_sha256:
        raise OSError(f"SHA256 mismatch (got {digest.hexdigest()}, expected {expected_sha256})")


def _ensure_checkpoint(model_type: str) -> Path:
    path = checkpoint_path(model_type)
    if path.exists():
        return path
    url = _CHECKPOINTS[model_type][1]
    part = path.with_name(path.name + ".part")
    logger.info("[SAM Mask] Downloading %s to %s ...", path.name, path.parent)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)  # only now that we really download
        _download(url, part, _SHA256.get(model_type))
        part.replace(path)
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {path.name} from {url} ({exc}). "
            f"Download it manually into {path.parent}."
        ) from exc
    return path


def unload() -> None:
    """Drop the cached SAM model and free its GPU memory."""
    had_model = bool(_PROMPTER_CACHE)
    _PROMPTER_CACHE.clear()
    gc.collect()
    if had_model and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_prompter(model_type: str, device: torch.device) -> object:
    key = (model_type, str(device))
    if key in _PROMPTER_CACHE:
        return _PROMPTER_CACHE[key]
    unload()  # switching model type or device evicts the previous model first
    from kornia.contrib.models.sam import SamConfig
    from kornia.contrib.visual_prompter import VisualPrompter

    path = _ensure_checkpoint(model_type)
    prompter = VisualPrompter(
        SamConfig(model_type=model_type, checkpoint=str(path)),
        device=device,
        dtype=torch.float32,
    )
    prompter.model.eval()
    _PROMPTER_CACHE[key] = prompter
    return prompter


_get_prompter.cache_clear = unload  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- prompt parsing


def _load_json(text: str, name: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {exc.msg}.") from exc


def parse_points_json(text: str, width: int, height: int, name: str = "points_json") -> list:
    """Parse ``"[[x, y], ...]"`` into a list of ``(x, y)`` pixel tuples.

    Normalised vs pixels rule: the points are read as normalised (0..1, scaled to
    ``size - 1``) only when EVERY value is within [0, 1] AND at least one value is strictly
    between 0 and 1 (e.g. ``0.5``). Anything else is pixels, so ``[[1, 1]]``, ``[[0, 0]]`` and
    ``[[1.0, 1.0]]`` mean the pixel (1, 1) / (0, 0). Pixel coordinates must satisfy
    ``0 <= x <= width - 1`` and ``0 <= y <= height - 1``. Empty text gives an empty list.

    Raises:
        TypeError: ``text`` is not a str.
        ValueError: bad JSON, wrong structure, non-finite numbers, or points outside the image.
    """
    if not isinstance(text, str):
        raise TypeError(f"{name} must be a str.")
    if not text.strip():
        return []
    data = _load_json(text, name)
    if isinstance(data, list) and len(data) == 2 and all(_is_number(v) for v in data):
        data = [data]  # tolerate a single bare [x, y]
    if not isinstance(data, list):
        raise ValueError(f'{name} must look like "[[x, y], [x, y]]".')
    pts: list[tuple[float, float]] = []
    for item in data:
        if not (
            isinstance(item, (list, tuple)) and len(item) == 2 and all(_is_number(v) for v in item)
        ):
            raise ValueError(f"{name}: every point must be [x, y] with numbers, got {item!r}.")
        x, y = float(item[0]), float(item[1])
        if x != x or y != y or abs(x) == float("inf") or abs(y) == float("inf"):
            raise ValueError(f"{name}: coordinates must be finite.")
        pts.append((x, y))
    if not pts:
        return []
    in_unit = all(0.0 <= v <= 1.0 for p in pts for v in p)
    normalised = in_unit and any(0.0 < v < 1.0 for p in pts for v in p)
    out: list[tuple[float, float]] = []
    for x, y in pts:
        if normalised:
            x, y = x * (width - 1), y * (height - 1)
        if not (0.0 <= x <= width - 1 and 0.0 <= y <= height - 1):
            raise ValueError(
                f"{name}: point ({x:g}, {y:g}) is outside the {width}x{height} image "
                f"(valid pixels are 0..{width - 1} and 0..{height - 1})."
            )
        out.append((x, y))
    return out


def _is_number(v: object) -> bool:
    return isinstance(v, Real) and not isinstance(v, bool)


def parse_bbox_json(
    text: str, width: int, height: int, batch_index: int = 0
) -> tuple[float, float, float, float] | None:
    """Pick a box ``(x0, y0, x1, y1)`` from Face Detect's ``bbox_json``.

    The format is a list of ``{x, y, w, h, ...}`` (a single dict is accepted too). An entry
    whose ``batch_index`` equals ``batch_index`` wins. If no entry carries ``batch_index`` the
    first one is used; if some do but none match, there is no box for this image (None).
    Empty text or an empty list gives None.

    Raises:
        TypeError: ``text`` is not a str.
        ValueError: bad JSON, entries without numeric x/y/w/h, or a degenerate box.
    """
    if not isinstance(text, str):
        raise TypeError("bbox_json must be a str.")
    if not text.strip():
        return None
    data = _load_json(text, "bbox_json")
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("bbox_json must be a list of {x, y, w, h} objects.")
    if not data:
        return None
    for entry in data:
        if not isinstance(entry, dict) or not all(
            _is_number(entry.get(k)) for k in ("x", "y", "w", "h")
        ):
            raise ValueError(f"bbox_json: each entry needs numeric x, y, w, h, got {entry!r}.")
    chosen = next((e for e in data if e.get("batch_index") == batch_index), None)
    if chosen is None:
        if any("batch_index" in e for e in data):
            return None
        chosen = data[0]
    x0 = min(max(float(chosen["x"]), 0.0), width - 1.0)
    y0 = min(max(float(chosen["y"]), 0.0), height - 1.0)
    x1 = min(max(float(chosen["x"]) + float(chosen["w"]), 0.0), float(width))
    y1 = min(max(float(chosen["y"]) + float(chosen["h"]), 0.0), float(height))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        raise ValueError(f"bbox_json: box is empty after clamping to the image: {chosen!r}.")
    return x0, y0, x1, y1


def mask_hint_prompt(
    mask_hw: torch.Tensor, threshold: float = 0.5
) -> tuple[tuple[float, float, float, float], tuple[float, float]] | None:
    """Bounding box ``(x0, y0, x1, y1)`` and an inside point from a rough (H, W) mask.

    The point is the centroid, snapped to the nearest mask pixel when the centroid falls
    outside the mask (C or ring shapes). Returns None when the mask is empty.
    """
    inside = mask_hw > threshold
    if not bool(inside.any()):
        return None
    ys, xs = torch.nonzero(inside, as_tuple=True)
    box = (
        float(xs.min()),
        float(ys.min()),
        float(xs.max()) + 1.0,
        float(ys.max()) + 1.0,
    )
    cx, cy = xs.float().mean(), ys.float().mean()
    ix, iy = int(round(float(cx))), int(round(float(cy)))
    if not bool(inside[min(iy, inside.shape[0] - 1), min(ix, inside.shape[1] - 1)]):
        d2 = (xs.float() - cx) ** 2 + (ys.float() - cy) ** 2
        k = int(torch.argmin(d2))
        ix, iy = int(xs[k]), int(ys[k])
    return box, (float(ix), float(iy))


# --------------------------------------------------------------------------- main entry


_MAX_COVERAGE = 0.9


def _pick_mask(binary: torch.Tensor, scores: torch.Tensor) -> int:
    """Index of the best candidate: highest score among masks covering <= 90% of the image.

    SAM sometimes returns a near full-frame mask with a high score for a centre click. If every
    candidate is that large the plain best score wins. ``binary`` is (C,H,W), ``scores`` (C,).
    """
    coverage = binary.flatten(1).float().mean(dim=1)
    ok = coverage <= _MAX_COVERAGE
    if bool(ok.any()):
        scores = scores.masked_fill(~ok, float("-inf"))
    return int(torch.argmax(scores))


def _validate_image(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must be (B,H,W,3), got {tuple(image.shape)}.")
    if not image.is_floating_point():
        raise ValueError(f"image must be a float tensor, got {image.dtype}.")
    if image.shape[0] < 1 or image.shape[1] < 2 or image.shape[2] < 2:
        raise ValueError("image must be at least 2x2 with batch >= 1.")


def _validate_mask_hint(mask: torch.Tensor | None, batch: int) -> None:
    if mask is None:
        return
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask_hint must be a torch.Tensor or None.")
    if mask.ndim == 2:
        return
    if mask.ndim != 3:
        raise ValueError(f"mask_hint must be (B,H,W), got {tuple(mask.shape)}.")
    if mask.shape[0] not in (1, batch):
        raise ValueError(f"mask_hint batch {mask.shape[0]} does not match image batch {batch}.")


def sam_mask(
    image: torch.Tensor,
    *,
    model_type: str = "vit_b",
    points_json: str = "",
    negative_points_json: str = "",
    bbox_json: str = "",
    mask_hint: torch.Tensor | None = None,
    multimask: bool = True,
    mask_threshold: float = 0.0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Segment the object under the prompts with Segment Anything.

    Args:
        image: float (B,H,W,3) in [0,1] (clamped).
        model_type: one of ``MODEL_TYPES``.
        points_json: positive points ``"[[x,y],...]"`` (pixels, or 0..1 normalised).
        negative_points_json: points that must NOT be in the mask, same format.
        bbox_json: Face Detect / Face Detect Lite ``bbox_json`` (``batch_index`` aware).
        mask_hint: optional float (B,H,W) or (1,H,W) rough mask: its bounding box is the box
            prompt (unless ``bbox_json`` gives one) and a point inside it a positive point.
        multimask: let SAM return 3 candidates and keep the best by predicted IoU; False asks
            for a single mask (better for precise prompts such as a box plus several points).
        mask_threshold: logit threshold for the binary mask (0 = SAM default).
        device: default CUDA when available, else CPU.

    Returns:
        ``(mask, score)``: float32 (B,H,W) in {0,1} on the image device, and float32 (B,) of the
        predicted IoU of the chosen mask. With no prompt at all the image centre is used as a
        positive point.

    Raises:
        TypeError / ValueError: invalid arguments or prompts.
        RuntimeError: checkpoint missing and download failed.
    """
    _validate_image(image)
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}.")
    if not isinstance(multimask, bool):
        raise TypeError("multimask must be a bool.")
    if isinstance(mask_threshold, bool) or not _is_number(mask_threshold):
        raise TypeError("mask_threshold must be a number.")
    if not -20.0 <= float(mask_threshold) <= 20.0:
        raise ValueError("mask_threshold must be in [-20, 20].")
    batch, height, width, _ = image.shape
    _validate_mask_hint(mask_hint, batch)

    pos = parse_points_json(points_json, width, height, "points_json")
    neg = parse_points_json(negative_points_json, width, height, "negative_points_json")
    # Validate bbox JSON once up front so errors surface even for item 0 with a mismatch.
    parse_bbox_json(bbox_json, width, height, 0)

    run_device = torch.device(device) if device is not None else None
    if run_device is None:
        run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompter = _get_prompter(model_type, run_device)

    out_masks: list[torch.Tensor] = []
    out_scores: list[float] = []
    for i in range(batch):
        img = torch.nan_to_num(image[i].detach(), nan=0.0, posinf=1.0, neginf=0.0)
        img = img.clamp(0.0, 1.0).permute(2, 0, 1).contiguous()
        img = img.to(device=run_device, dtype=torch.float32)

        pts = list(pos)
        box = parse_bbox_json(bbox_json, width, height, i)
        if mask_hint is not None:
            hint = (
                mask_hint if mask_hint.ndim == 2 else mask_hint[0 if mask_hint.shape[0] == 1 else i]
            )
            if tuple(hint.shape) != (height, width):
                hint = nnf.interpolate(
                    hint[None, None].float(), size=(height, width), mode="nearest"
                )[0, 0]
            prompt = mask_hint_prompt(hint.detach().float())
            if prompt is not None:
                if box is None:
                    box = prompt[0]
                pts.append(prompt[1])
        if not pts and box is None and not neg:
            if bbox_json.strip():
                # A detector ran but found nothing in this image: no object, empty mask.
                out_masks.append(torch.zeros(height, width, dtype=torch.float32, device=run_device))
                out_scores.append(0.0)
                continue
            pts = [((width - 1) / 2.0, (height - 1) / 2.0)]

        if not pts and box is None and neg:
            logger.warning(
                "[SAM Mask] Only negative points given: SAM has nothing to select, so the "
                "result is unreliable. Add a positive point, a box or a mask_hint."
            )
        coords = pts + neg
        labels = [1.0] * len(pts) + [0.0] * len(neg)
        kwargs: dict[str, object] = {}
        if coords:
            kwargs["keypoints"] = torch.tensor([coords], dtype=torch.float32, device=run_device)
            kwargs["keypoints_labels"] = torch.tensor(
                [labels], dtype=torch.float32, device=run_device
            )
        if box is not None:
            # Kornia's raw-tensor box path is broken (expects corners); build a Boxes object.
            kwargs["boxes"] = Boxes.from_tensor(
                torch.tensor([box], dtype=torch.float32, device=run_device), mode="xyxy"
            )

        with torch.no_grad():
            prompter.set_image(img)
            result = prompter.predict(multimask_output=multimask, **kwargs)
            result.mask_threshold = float(mask_threshold)
            binary = result.binary_masks  # (1, C, H, W) bool at original resolution
            scores = result.scores  # (1, C)
            best = _pick_mask(binary[0], scores[0])
            out_masks.append(binary[0, best].to(torch.float32))
            out_scores.append(float(scores[0, best]))
            prompter.reset_image()

    mask = torch.stack(out_masks).to(image.device)
    return mask, torch.tensor(out_scores, dtype=torch.float32, device=image.device)
