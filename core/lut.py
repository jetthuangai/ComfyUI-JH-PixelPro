"""Identity HALD generator + Adobe Cube 1.0 (.cube) 3D-LUT exporter.

Pure ``torch.arange`` reshape + stdlib file I/O. No Kornia, no bench-critical
GPU-parallel primitive. ImageMagick HALD convention for the identity image,
blue-outer-loop ordering for the ``.cube`` body (DaVinci Resolve / Premiere /
OCIO compatible).
"""

from __future__ import annotations

import os

import torch


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
    side = level ** 3  # n * level == level ** 3
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
    side = level ** 3

    if hald.ndim == 4:
        hald = hald[0]
    if hald.shape != (side, side, 3):
        raise ValueError(
            f"hald shape must be ({side},{side},3) for level={level}, "
            f"got {tuple(hald.shape)}"
        )
    if not torch.isfinite(hald).all():
        raise ValueError(
            "hald contains non-finite values (NaN/Inf) — clamp or filter upstream"
        )

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
