"""Identity HALD generator + Adobe Cube 1.0 (.cube) 3D-LUT exporter.

Pure ``torch.arange`` reshape + stdlib file I/O. No Kornia, no bench-critical
GPU-parallel primitive. ImageMagick HALD convention for the identity image,
blue-outer-loop ordering for the ``.cube`` body (DaVinci Resolve / Premiere /
OCIO compatible).
"""

from __future__ import annotations

import os
from numbers import Real
from pathlib import Path

import torch
import torch.nn.functional as functional


def identity_hald(
    level: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate an identity HALD image for 3D LUT round-trip (ImageMagick convention).

    Args:
        level: HALD level L in [2, 16]. Cube grid N = L**2. Image side = L**3.
        device: torch device for output. CPU default — HALD generation is cheap.
        dtype: Output dtype. Pack downstream assumes float32 BHWC in [0, 1].

    Returns:
        Tensor shape ``(1, L**3, L**3, 3)`` in ``[0, 1]``. Pixel ``(y, x)`` encodes::

            r = (x mod N) / (N - 1)
            g = (y mod N) / (N - 1)
            b = ((x // N) + (y // N) * L) / (N - 1)

        with ``N = L**2``.

    Raises:
        ValueError: ``level`` not an int in ``[2, 16]``.
    """
    if not isinstance(level, int) or level < 2 or level > 16:
        raise ValueError(f"HALD level must be int in [2, 16], got {level!r}")

    n = level * level
    side = level**3  # n * level == level ** 3
    x = torch.arange(side, device=device).view(1, side).expand(side, side)
    y = torch.arange(side, device=device).view(side, 1).expand(side, side)
    r = (x % n).to(dtype) / (n - 1)
    g = (y % n).to(dtype) / (n - 1)
    b = ((x // n) + (y // n) * level).to(dtype) / (n - 1)
    hald = torch.stack([r, g, b], dim=-1)  # (side, side, 3)
    return hald.unsqueeze(0)  # (1, side, side, 3)


def export_cube(
    hald: torch.Tensor,
    level: int,
    path: str,
    *,
    title: str = "JHPixelPro LUT",
) -> str:
    """Export a graded HALD image as an Adobe Cube 1.0 (``.cube``) 3D-LUT file.

    Args:
        hald: Graded HALD tensor ``(B, H, W, 3)`` or ``(H, W, 3)`` float in ``[0, 1]``.
            First batch taken. ``H == W == level**3`` required (must match the
            identity HALD output for the same level).
        level: HALD level L used when the identity HALD was generated. Cube
            ``N = L**2``. Must match ``hald`` spatial dims.
        path: Output file path (absolute or relative). Parent directory must exist.
        title: TITLE header string (Adobe Cube metadata).

    Returns:
        Absolute resolved path of the written file.

    Raises:
        ValueError: ``level`` out of ``[2, 16]``, ``hald`` shape mismatch, or
            non-finite values in ``hald``.
        OSError: Parent directory missing or write permission denied.
    """
    if not isinstance(level, int) or level < 2 or level > 16:
        raise ValueError(f"HALD level must be int in [2, 16], got {level!r}")

    n = level * level
    side = level**3

    if hald.ndim == 4:
        hald = hald[0]
    if hald.shape != (side, side, 3):
        raise ValueError(
            f"hald shape must be ({side},{side},3) for level={level}, got {tuple(hald.shape)}"
        )
    if not torch.isfinite(hald).all():
        raise ValueError("hald contains non-finite values (NaN/Inf) — clamp or filter upstream")

    samples = hald.clamp(0.0, 1.0).detach().to("cpu", torch.float32).numpy()

    resolved = os.path.abspath(path)
    parent = os.path.dirname(resolved)
    if parent and not os.path.isdir(parent):
        raise OSError(f"Parent directory missing: {parent}")

    lines = [
        f'TITLE "{title}"',
        f"LUT_3D_SIZE {n}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
        "",
    ]
    # Blue-outer / green-middle / red-innermost (Adobe Cube 1.0 standard).
    for b_idx in range(n):
        by = b_idx // level
        bx = b_idx % level
        for g_idx in range(n):
            y = g_idx + by * n
            for r_idx in range(n):
                x = r_idx + bx * n
                r, g, b = samples[y, x]
                lines.append(f"{r:.6f} {g:.6f} {b:.6f}")

    with open(resolved, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return resolved


def _parse_float_triplet(parts: list[str], *, line_num: int) -> tuple[float, float, float]:
    if len(parts) != 3:
        raise ValueError(f"line {line_num}: expected 3 float values, got {len(parts)}.")
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError as exc:
        raise ValueError(f"line {line_num}: malformed numeric values: {' '.join(parts)}") from exc


def parse_cube(path: str | Path) -> dict:
    """Parse an Adobe Cube 1.0 3D LUT file into a tensor grid."""

    cube_path = Path(path)
    if not cube_path.is_file():
        raise FileNotFoundError(f".cube file not found: {cube_path}")

    title = ""
    size: int | None = None
    domain_min = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    domain_max = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    body: list[tuple[float, float, float]] = []

    for line_num, raw_line in enumerate(
        cube_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split()
        keyword = parts[0]
        if keyword == "TITLE":
            title_value = stripped[len("TITLE") :].strip()
            if not title_value:
                raise ValueError(f"line {line_num}: TITLE header missing value.")
            if len(title_value) >= 2 and title_value[0] == '"' and title_value[-1] == '"':
                title = title_value[1:-1]
            else:
                title = title_value
            continue

        if keyword == "LUT_3D_SIZE":
            if len(parts) != 2:
                raise ValueError(f"line {line_num}: LUT_3D_SIZE expects a single integer value.")
            try:
                size = int(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"line {line_num}: invalid LUT_3D_SIZE value {parts[1]!r}."
                ) from exc
            if size < 2 or size > 256:
                raise ValueError(f"line {line_num}: LUT_3D_SIZE must be in [2, 256], got {size}.")
            continue

        if keyword == "DOMAIN_MIN":
            domain_min = torch.tensor(
                _parse_float_triplet(parts[1:], line_num=line_num), dtype=torch.float32
            )
            continue

        if keyword == "DOMAIN_MAX":
            domain_max = torch.tensor(
                _parse_float_triplet(parts[1:], line_num=line_num), dtype=torch.float32
            )
            continue

        body.append(_parse_float_triplet(parts, line_num=line_num))

    if size is None:
        raise ValueError("missing LUT_3D_SIZE header.")

    expected_body = size**3
    if len(body) != expected_body:
        raise ValueError(f"expected {expected_body} body lines, got {len(body)}.")

    lut = torch.tensor(body, dtype=torch.float32).view(size, size, size, 3)
    return {
        "lut": lut,
        "size": size,
        "title": title,
        "domain_min": domain_min,
        "domain_max": domain_max,
    }


def _prepare_domain_tensor(
    name: str,
    value: torch.Tensor | None,
    *,
    default: tuple[float, float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        return torch.tensor(default, dtype=dtype, device=device)

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None.")
    if value.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {tuple(value.shape)}.")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must be floating point, got {value.dtype}.")
    return value.to(device=device, dtype=dtype)


def _prepare_mask(
    mask: torch.Tensor | None,
    *,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if mask is None:
        return None

    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor or None.")

    if mask.ndim == 3:
        prepared = mask
    elif mask.ndim == 4 and mask.shape[1] == 1:
        prepared = mask.squeeze(1)
    else:
        raise ValueError(f"mask must have shape (B,H,W) or (B,1,H,W), got {tuple(mask.shape)}.")

    if prepared.shape[-2:] != (height, width):
        raise ValueError(
            f"mask HxW {tuple(prepared.shape[-2:])} must match image HxW {(height, width)}."
        )
    if prepared.shape[0] not in (1, batch):
        raise ValueError(
            f"mask batch ({prepared.shape[0]}) must be 1 or equal to image batch ({batch})."
        )
    if not torch.is_floating_point(prepared) and prepared.dtype is not torch.bool:
        raise ValueError(f"mask must be float or bool, got {prepared.dtype}.")

    prepared = prepared.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    if prepared.shape[0] == 1 and batch > 1:
        prepared = prepared.expand(batch, -1, -1)
    return prepared


def apply_lut_3d(
    image_bhwc: torch.Tensor,
    lut_grid: torch.Tensor,
    *,
    strength: float = 1.0,
    mask: torch.Tensor | None = None,
    domain_min: torch.Tensor | None = None,
    domain_max: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a 3D LUT to a BHWC RGB tensor via trilinear sampling."""

    if not isinstance(image_bhwc, torch.Tensor):
        raise TypeError("image_bhwc must be a torch.Tensor.")
    if image_bhwc.ndim != 4 or image_bhwc.shape[-1] != 3:
        raise ValueError(f"image_bhwc must have shape (B,H,W,3), got {tuple(image_bhwc.shape)}.")
    if not torch.is_floating_point(image_bhwc):
        raise ValueError(f"image_bhwc must be floating point, got {image_bhwc.dtype}.")

    if not isinstance(lut_grid, torch.Tensor):
        raise TypeError("lut_grid must be a torch.Tensor.")
    if lut_grid.ndim != 4 or lut_grid.shape[-1] != 3:
        raise ValueError(f"lut_grid must have shape (N,N,N,3), got {tuple(lut_grid.shape)}.")
    if lut_grid.shape[0] != lut_grid.shape[1] or lut_grid.shape[1] != lut_grid.shape[2]:
        raise ValueError(
            f"lut_grid must be cubic along the first 3 axes, got {tuple(lut_grid.shape)}."
        )
    if not torch.is_floating_point(lut_grid):
        raise ValueError(f"lut_grid must be floating point, got {lut_grid.dtype}.")

    if isinstance(strength, bool) or not isinstance(strength, Real):
        raise ValueError("strength must be in [0.0, 1.0].")
    strength_float = float(strength)
    if not 0.0 <= strength_float <= 1.0:
        raise ValueError("strength must be in [0.0, 1.0].")

    compute_dtype = (
        image_bhwc.dtype if image_bhwc.dtype in {torch.float32, torch.float64} else torch.float32
    )
    image = image_bhwc.to(dtype=compute_dtype)
    lut = lut_grid.to(device=image.device, dtype=compute_dtype)
    batch, height, width, _ = image.shape

    domain_min_tensor = _prepare_domain_tensor(
        "domain_min",
        domain_min,
        default=(0.0, 0.0, 0.0),
        device=image.device,
        dtype=compute_dtype,
    )
    domain_max_tensor = _prepare_domain_tensor(
        "domain_max",
        domain_max,
        default=(1.0, 1.0, 1.0),
        device=image.device,
        dtype=compute_dtype,
    )
    if torch.any(domain_max_tensor <= domain_min_tensor):
        raise ValueError("domain_max must be strictly greater than domain_min for all channels.")

    mask_bhw = _prepare_mask(
        mask,
        batch=batch,
        height=height,
        width=width,
        device=image.device,
        dtype=compute_dtype,
    )

    input_clamped = image.clamp(
        min=domain_min_tensor.view(1, 1, 1, 3),
        max=domain_max_tensor.view(1, 1, 1, 3),
    )
    domain_range = (domain_max_tensor - domain_min_tensor).clamp_min(1e-6)
    normed = (input_clamped - domain_min_tensor.view(1, 1, 1, 3)) / domain_range.view(1, 1, 1, 3)
    grid = normed.mul(2.0).sub(1.0).unsqueeze(1)

    volume = lut.permute(3, 0, 1, 2).unsqueeze(0).expand(batch, -1, -1, -1, -1)
    lut_output = functional.grid_sample(
        volume,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    lut_output = lut_output.squeeze(2).permute(0, 2, 3, 1)

    blend = torch.full(
        (batch, height, width, 1), strength_float, dtype=compute_dtype, device=image.device
    )
    if mask_bhw is not None:
        blend = blend * mask_bhw.unsqueeze(-1)
    output = input_clamped * (1.0 - blend) + lut_output * blend
    return output.to(dtype=image_bhwc.dtype)
