"""Pack-bundled LUT preset discovery and loading."""

from __future__ import annotations

from pathlib import Path

from .lut import parse_cube

PRESETS_DIR = Path(__file__).resolve().parents[1] / "presets"


def list_presets() -> list[str]:
    """Return bundled ``.cube`` preset names without extension."""
    if not PRESETS_DIR.is_dir():
        return []
    return sorted(path.stem for path in PRESETS_DIR.glob("*.cube") if path.is_file())


def preset_path(name: str) -> Path:
    """Resolve a bundled preset name to a safe path inside ``presets/``."""
    preset_name = name.strip()
    if not preset_name or Path(preset_name).name != preset_name:
        raise ValueError(f"invalid preset name: {name!r}")
    path = PRESETS_DIR / f"{Path(preset_name).stem}.cube"
    if not path.is_file():
        raise FileNotFoundError(f"LUT preset not found: {preset_name}")
    return path


def load_preset(name: str) -> dict:
    """Parse a bundled ``.cube`` LUT preset."""
    return parse_cube(preset_path(name))
