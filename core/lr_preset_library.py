"""Preset library for the Lightroom Preset browser: thumbnails, favourites, recents.

- Thumbnails show each preset on the image the node last processed (kept small in
  memory per node), or on a generated colour chart before the first run. They are
  rendered on the CPU (a workflow may be using the GPU) and cached by preset file,
  source image and size, so a page of a large library renders once.
- Favourites and recents live in ``.jh_presets.json`` inside the preset folder, so
  every node and workflow shares them and copying the folder moves them too.
- Collections: "All", "Favorites", "Recent" and one per sub-folder (Lightroom's
  preset groups).
"""

from __future__ import annotations

import io
import json
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from .color_lab import apply_colorlab_pipeline, merge_params_json, params_to_json
from .lr_preset import (
    LightroomPreset,
    list_presets,
    load_lightroom_preset,
    preset_dirs,
    resolve_preset,
)

ALL, FAVORITES, RECENT = "All", "★ Favorites", "Recent"
SOURCE_SIDE = 384
MAX_RECENT = 12
_THUMB_CACHE_ITEMS = 1500
_META_NAME = ".jh_presets.json"

_lock = threading.Lock()
_sources: dict[str, tuple[int, torch.Tensor]] = {}
_source_counter = 0
_thumbs: OrderedDict[tuple, bytes] = OrderedDict()
_parsed: dict[tuple[str, int], LightroomPreset] = {}


# --- source images -------------------------------------------------------------------


def sample_chart(width: int = SOURCE_SIDE, height: int = 256) -> torch.Tensor:
    """Generated test chart (1,H,W,3): grey ramp, skin, sky, foliage, primaries, steps."""

    img = np.zeros((height, width, 3), dtype=np.float32)
    ramp = np.linspace(0.02, 0.98, width, dtype=np.float32)
    img[: height // 4] = ramp[None, :, None]
    patches = [
        (0.86, 0.66, 0.55),  # light skin
        (0.55, 0.36, 0.26),  # dark skin
        (0.36, 0.56, 0.86),  # sky
        (0.27, 0.45, 0.20),  # foliage
        (0.80, 0.18, 0.16),  # red
        (0.92, 0.78, 0.20),  # yellow
    ]
    top, bottom = height // 4, height * 3 // 4
    step = width // len(patches)
    for i, colour in enumerate(patches):
        img[top:bottom, i * step : (i + 1) * step] = colour
    steps = np.linspace(0.04, 0.96, 8, dtype=np.float32)
    for i, value in enumerate(steps):
        img[bottom:, i * width // 8 : (i + 1) * width // 8] = value
    return torch.from_numpy(img).unsqueeze(0)


def _fit(image: torch.Tensor, side: int) -> torch.Tensor:
    """Downscale (1,H,W,3) so the long edge is at most ``side`` (area average)."""

    height, width = image.shape[1:3]
    scale = min(1.0, side / max(height, width))
    if scale >= 1.0:
        return image
    size = (max(1, round(height * scale)), max(1, round(width * scale)))
    bchw = functional.interpolate(image.permute(0, 3, 1, 2), size=size, mode="area")
    return bchw.permute(0, 2, 3, 1)


def remember_source(node_id: str, image: torch.Tensor) -> int:
    """Keep a small CPU copy of the first image a node processed; returns its version."""

    global _source_counter
    small = _fit(image[:1].detach().float().cpu().clamp(0.0, 1.0), SOURCE_SIDE).contiguous()
    with _lock:
        _source_counter += 1
        _sources[str(node_id)] = (_source_counter, small)
        return _source_counter


def source_for(node_id: str | None, version: int | None = None) -> tuple[int, torch.Tensor]:
    """(version, image) for a node; version 0 is the sample chart.

    With ``version`` the stored image is used only if it is that exact version, so a
    node of another workflow that happens to share the id never shows this image.
    """

    with _lock:
        stored = _sources.get(str(node_id)) if node_id is not None else None
    if stored is not None and (version is None or stored[0] == version):
        return stored
    return 0, sample_chart()


# --- presets -------------------------------------------------------------------------


def _load(path: Path) -> LightroomPreset:
    key = (str(path), path.stat().st_mtime_ns)
    with _lock:
        cached = _parsed.get(key)
    if cached is None:
        cached = load_lightroom_preset(path)
        with _lock:
            _parsed[key] = cached
    return cached


def preset_params(preset: LightroomPreset, tone_engine: str) -> dict[str, object]:
    """ColorLab parameters for a preset, as the Lightroom Preset node renders it."""

    base = params_to_json({**preset.params, "tone_engine": tone_engine, "exposure_mode": "linear"})
    return merge_params_json({}, base)


def group_of(name: str) -> str:
    """Collection (sub-folder) of a listed preset name; "" for the folder root."""

    return name.rsplit("/", 1)[0] if "/" in name else ""


def summaries(dirs: list[Path] | None = None) -> list[dict[str, object]]:
    """Every preset with its group, title and what is not supported (for the browser)."""

    out = []
    for name in list_presets(dirs):
        entry: dict[str, object] = {"name": name, "group": group_of(name)}
        try:
            preset = _load(resolve_preset(name, dirs=dirs))
            entry.update(title=preset.title, unsupported=preset.ignored, notes=preset.notes)
        except (ValueError, OSError) as exc:
            entry.update(title=Path(name).stem, error=str(exc))
        out.append(entry)
    return out


def collections(dirs: list[Path] | None = None) -> list[str]:
    groups = sorted({group_of(name) for name in list_presets(dirs)} - {""}, key=str.lower)
    return [ALL, FAVORITES, RECENT, *groups]


def filter_presets(
    names: list[str], collection: str, search: str = "", meta: dict | None = None
) -> list[str]:
    """Presets of a collection whose name contains every word of ``search``."""

    meta = meta if meta is not None else load_meta()
    if collection == FAVORITES:
        picked = [n for n in meta["favorites"] if n in names]
    elif collection == RECENT:
        picked = [n for n in meta["recent"] if n in names]
    elif collection == ALL:
        picked = list(names)
    else:
        picked = [n for n in names if n == collection or n.startswith(collection + "/")]
    words = search.lower().split()
    return [n for n in picked if all(word in n.lower() for word in words)]


# --- favourites / recents -------------------------------------------------------------


def _meta_path(dirs: list[Path] | None = None) -> Path:
    return (dirs if dirs is not None else preset_dirs())[0] / _META_NAME


def load_meta(dirs: list[Path] | None = None) -> dict[str, list[str]]:
    try:
        data = json.loads(_meta_path(dirs).read_text("utf-8"))
    except (OSError, ValueError):
        data = {}
    return {
        "favorites": [str(n) for n in data.get("favorites", []) if isinstance(n, str)],
        "recent": [str(n) for n in data.get("recent", []) if isinstance(n, str)],
    }


def _save_meta(meta: dict[str, list[str]], dirs: list[Path] | None = None) -> None:
    path = _meta_path(dirs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), "utf-8")


def set_favorite(name: str, favorite: bool, dirs: list[Path] | None = None) -> list[str]:
    if name not in list_presets(dirs):
        raise ValueError(f"unknown preset {name!r}")
    with _lock:
        meta = load_meta(dirs)
        favs = [n for n in meta["favorites"] if n != name]
        if favorite:
            favs.append(name)
        meta["favorites"] = sorted(favs, key=str.lower)
        _save_meta(meta, dirs)
        return meta["favorites"]


def add_recent(name: str, dirs: list[Path] | None = None) -> None:
    with _lock:
        meta = load_meta(dirs)
        meta["recent"] = [name, *[n for n in meta["recent"] if n != name]][:MAX_RECENT]
        _save_meta(meta, dirs)


# --- thumbnails ----------------------------------------------------------------------


def render_thumbnail(
    name: str,
    node_id: str | None = None,
    size: int = 192,
    tone_engine: str = "local",
    dirs: list[Path] | None = None,
    version: int | None = None,
) -> bytes:
    """JPEG of preset ``name`` applied to the node's image ``version`` (or the chart)."""

    size = max(64, min(int(size), 384))
    path = resolve_preset(name, dirs=dirs)
    version, source = source_for(node_id, version)
    key = (str(path), path.stat().st_mtime_ns, str(node_id), version, size, tone_engine)
    with _lock:
        if key in _thumbs:
            _thumbs.move_to_end(key)
            return _thumbs[key]
    image = _fit(source, size)
    out = apply_colorlab_pipeline(image, preset_params(_load(path), tone_engine), prefer_gpu=False)
    pixels = (out[0].clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=85)
    data = buffer.getvalue()
    with _lock:
        _thumbs[key] = data
        while len(_thumbs) > _THUMB_CACHE_ITEMS:
            _thumbs.popitem(last=False)
    return data


# --- contact sheet -------------------------------------------------------------------


def _font(size: int):  # noqa: ANN202 - PIL font object
    from PIL import ImageFont

    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "Arial Unicode.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_contact_sheet(
    image: torch.Tensor,
    names: list[str],
    *,
    columns: int = 4,
    thumb_size: int = 320,
    tone_engine: str = "local",
    dirs: list[Path] | None = None,
) -> torch.Tensor:
    """Grid of ``image`` (first frame) with each preset applied, labelled; original first."""

    from PIL import ImageDraw

    small = _fit(image[:1].float().clamp(0.0, 1.0), thumb_size)
    tiles: list[tuple[str, torch.Tensor]] = [("Original", small)]
    for name in names:
        try:
            preset = _load(resolve_preset(name, dirs=dirs))
        except (ValueError, OSError):
            continue
        label = Path(name).stem + (" [!]" if preset.ignored else "")
        tiles.append((label, apply_colorlab_pipeline(small, preset_params(preset, tone_engine))))

    height, width = small.shape[1:3]
    label_h, gap = max(18, thumb_size // 14), 6
    columns = max(1, min(columns, len(tiles)))
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * (width + gap) + gap, rows * (height + label_h + gap) + gap),
        (28, 28, 28),
    )
    draw, font = ImageDraw.Draw(sheet), _font(max(11, label_h - 6))
    for index, (label, tile) in enumerate(tiles):
        row, col = divmod(index, columns)
        x = gap + col * (width + gap)
        y = gap + row * (height + label_h + gap)
        pixels = (tile[0].detach().cpu().clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)
        sheet.paste(Image.fromarray(pixels), (x, y + label_h))
        draw.text((x + 2, y + 1), label[:48], fill=(230, 230, 230), font=font)
    return torch.from_numpy(np.asarray(sheet, dtype=np.float32) / 255.0).unsqueeze(0)
