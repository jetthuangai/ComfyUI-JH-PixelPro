"""Read Lightroom / Camera Raw presets (.xmp, .lrtemplate) as ColorLab parameters.

Formats:
- ``.xmp``: an XMP packet whose ``crs:`` (camera-raw-settings) properties are
  attributes of ``rdf:Description`` or child elements (sequences such as
  ``crs:ToneCurvePV2012`` hold ``rdf:li`` items "x, y").
- ``.lrtemplate``: a Lua table literal ``s = { title = ..., value = { settings =
  { Exposure2012 = 0.3, ... } } }``. It is parsed as data only; no Lua runs.

Mapping notes (Lightroom Classic 10+): Shadows / Highlights colour-grading hue and
saturation are still stored as ``SplitToning*``; ``ColorGrade*`` holds midtones,
luminance and the global wheel. Grain, post-crop vignette, calibration and point /
parametric curves map to ColorLab's params_json-only settings; absolute Kelvin white
balance is approximated against the as-shot white (noted in the report). Anything
ColorLab cannot reproduce (creative profiles, local masks, lens and detail settings,
...) is listed in the report instead of being silently dropped.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .color_lab import _monotone_lut

PRESET_SUFFIXES = (".xmp", ".lrtemplate")
MAX_PRESET_BYTES = 4 * 1024 * 1024
_CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"
_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

_LR_COLORS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
_GRAY_COLORS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple")

_DIRECT = {
    "Exposure2012": "basic_exposure",
    "Contrast2012": "basic_contrast",
    "Highlights2012": "basic_highlights",
    "Shadows2012": "basic_shadows",
    "Whites2012": "basic_whites",
    "Blacks2012": "basic_blacks",
    "Texture": "basic_texture",
    "Clarity2012": "basic_clarity",
    "Dehaze": "basic_dehaze",
    "Vibrance": "basic_vibrance",
    "Saturation": "basic_saturation",
    "IncrementalTemperature": "wb_temperature",
    "IncrementalTint": "wb_tint",
    "SplitToningShadowHue": "grade_shadow_hue",
    "SplitToningShadowSaturation": "grade_shadow_sat",
    "SplitToningHighlightHue": "grade_highlight_hue",
    "SplitToningHighlightSaturation": "grade_highlight_sat",
    "ColorGradeShadowLum": "grade_shadow_lum",
    "ColorGradeMidtoneHue": "grade_mid_hue",
    "ColorGradeMidtoneSat": "grade_mid_sat",
    "ColorGradeMidtoneLum": "grade_mid_lum",
    "ColorGradeHighlightLum": "grade_highlight_lum",
}
for _color in _LR_COLORS:
    _DIRECT[f"HueAdjustment{_color}"] = f"hsl_{_color.lower()}_hue"
    _DIRECT[f"SaturationAdjustment{_color}"] = f"hsl_{_color.lower()}_sat"
    _DIRECT[f"LuminanceAdjustment{_color}"] = f"hsl_{_color.lower()}_lum"
for _color in _GRAY_COLORS:
    _DIRECT[f"GrayMixer{_color}"] = f"gray_{_color.lower()}"
for _color in ("Red", "Green", "Blue"):
    _DIRECT[f"{_color}Hue"] = f"calib_{_color.lower()}_hue"
    _DIRECT[f"{_color}Saturation"] = f"calib_{_color.lower()}_sat"
_DIRECT["ShadowTint"] = "calib_shadow_tint"
_DIRECT["GrayMixerMagenta"] = "gray_magenta"
_DIRECT.update(
    {
        "GrainAmount": "grain_amount",
        "GrainSize": "grain_size",
        "GrainFrequency": "grain_frequency",
        "PostCropVignetteAmount": "vignette_amount",
        "PostCropVignetteMidpoint": "vignette_midpoint",
        "PostCropVignetteFeather": "vignette_feather",
        "PostCropVignetteRoundness": "vignette_roundness",
    }
)
# Shape settings only matter when their effect's amount is set.
_EFFECT_SHAPES = {
    "grain_amount": ("grain_size", "grain_frequency"),
    "vignette_amount": ("vignette_midpoint", "vignette_feather", "vignette_roundness"),
}
# Profiles that are Lightroom's neutral starting point, not a creative look.
_DEFAULT_PROFILES = {"adobe color", "adobe standard", "adobe default"}

_PARAMETRIC = ("ParametricShadows", "ParametricDarks", "ParametricLights", "ParametricHighlights")
_SPLITS = (
    ("ParametricShadowSplit", 25.0),
    ("ParametricMidtoneSplit", 50.0),
    ("ParametricHighlightSplit", 75.0),
)
# Lightroom white-balance presets (Kelvin); tint is left as shot.
_WB_PRESETS = {
    "Daylight": 5500.0,
    "Cloudy": 6500.0,
    "Shade": 7500.0,
    "Tungsten": 2850.0,
    "Fluorescent": 3800.0,
    "Flash": 5500.0,
}
# Relative Temp units per mired: ColorLab's +100 (R x1.3, B x0.7) is ~125 mired warmer.
_TEMP_PER_MIRED = 0.8
# Detail / optics settings: not part of a colour look, listed separately in the report.
_NOT_COLOUR = re.compile(
    r"^(Sharpen|Sharpness|LuminanceSmoothing|LuminanceNoise|ColorNoise|Lens|AutoLateralCA|"
    r"Defringe|Perspective|Upright|Crop|Vignette|ChromaticAberration|EnableLens)"
)

# Pre-2012 process names, used only when the 2012 key is absent.
_FALLBACK = {"Clarity": "Clarity2012", "Exposure": "Exposure2012"}
_CURVES = {
    "ToneCurvePV2012": "tone_curve",
    "ToneCurvePV2012Red": "tone_curve_red",
    "ToneCurvePV2012Green": "tone_curve_green",
    "ToneCurvePV2012Blue": "tone_curve_blue",
}
# Bookkeeping keys that carry no look.
_METADATA = {
    "Version",
    "ProcessVersion",
    "PresetType",
    "Cluster",
    "UUID",
    "SupportsAmount",
    "SupportsAmount2",
    "SupportsColor",
    "SupportsMonochrome",
    "SupportsHighDynamicRange",
    "SupportsNormalDynamicRange",
    "SupportsSceneReferred",
    "SupportsOutputReferred",
    "CameraModelRestriction",
    "Copyright",
    "ContactInfo",
    "Name",
    "ShortName",
    "SortName",
    "Group",
    "Description",
    "HasSettings",
    "AlreadyApplied",
    "ToneCurveName2012",
    "ToneCurveName",
    "WhiteBalance",
    "CameraProfile",
    "CameraProfileDigest",
    "LookTable",
    "ConvertToGrayscale",
    "SplitToningBalance",
    "ColorGradeBlending",
    "ColorGradeBalance",
    "OverrideLookVignette",
    "HDREditMode",
    "AsShotTemperature",
    "AsShotTint",
    "Temperature",
    "Tint",
    "Stubbed",
    "Amount",
    "CurveRefineSaturation",
    "PostCropVignetteStyle",
    *_PARAMETRIC,
    *(name for name, _ in _SPLITS),
}


@dataclass
class LightroomPreset:
    """A preset translated to ColorLab parameters, with what could not be carried over."""

    title: str
    params: dict[str, object]
    applied: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            f"Preset: {self.title}",
            f"Applied ({len(self.applied)}): {', '.join(self.applied) or '-'}",
        ]
        if self.ignored:
            lines.append(f"Not supported ({len(self.ignored)}): {', '.join(self.ignored)}")
        if self.skipped:
            lines.append(
                f"Detail/optics, not colour ({len(self.skipped)}): {', '.join(self.skipped)}"
            )
        lines.extend(f"Note: {note}" for note in self.notes)
        return "\n".join(lines)


# --- reading -------------------------------------------------------------------------


def _local(tag: str) -> tuple[str, str]:
    if tag.startswith("{"):
        ns, name = tag[1:].split("}", 1)
        return ns, name
    return "", tag


def parse_xmp(text: str) -> tuple[str | None, dict[str, object]]:
    """``crs:`` properties of an XMP preset (attributes and element forms) and its title."""

    try:
        root = ElementTree.fromstring(text.encode("utf-8") if isinstance(text, str) else text)
    except ElementTree.ParseError as exc:
        raise ValueError(f"not a valid XMP file: {exc}") from exc
    props: dict[str, object] = {}
    # A profile ("crs:Look") nests its own Name/Amount/UUID: keep them out of the settings.
    looks = [el for el in root.iter() if _local(el.tag) == (_CRS, "Look")]
    nested = {id(child) for look in looks for child in look.iter() if child is not look}
    for look in looks:
        description = next((el for el in look.iter() if el is not look and el.attrib), None)
        if description is not None:
            props["Look"] = {_local(k)[1]: v for k, v in description.attrib.items()}
    for element in root.iter():
        if id(element) in nested or _local(element.tag) == (_CRS, "Look"):
            continue
        for attr, value in element.attrib.items():
            ns, name = _local(attr)
            if ns == _CRS:
                props[name] = value
        ns, name = _local(element.tag)
        if ns != _CRS:
            continue
        items = [li.text or "" for li in element.iter(f"{{{_RDF}}}li")]
        if items:
            props[name] = items if name not in ("Name", "Group", "Description") else items[0]
        elif element.text and element.text.strip() and len(element) == 0:
            props[name] = element.text.strip()
    title = props.get("Name")
    return (str(title) if title else None), props


class _Lua:
    """Minimal reader for Lua table literals (strings, numbers, booleans, nested tables)."""

    _TOKEN = re.compile(
        r"""\s+|--\[(?P<eq>=*)\[.*?\](?P=eq)\]|--[^\n]*|"""
        r"""(?P<str>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|"""
        r"""(?P<long>\[(?P<leq>=*)\[.*?\](?P=leq)\])|"""
        r"""(?P<num>-?(?:0[xX][0-9a-fA-F]+|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))|"""
        r"""(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<sym>[{}=,;\[\]])""",
        re.S,
    )

    _ESCAPE = re.compile(r"\\(.)", re.S)
    _ESCAPES = {"n": "\n", "t": "\t", "r": "\r"}

    @classmethod
    def _unescape(cls, match: re.Match[str]) -> str:
        return cls._ESCAPES.get(match.group(1), match.group(1))

    def __init__(self, text: str) -> None:
        self.tokens: list[tuple[str, object]] = []
        pos = 0
        while pos < len(text):
            match = self._TOKEN.match(text, pos)
            if not match:
                raise ValueError(f"unexpected character {text[pos]!r} in .lrtemplate at {pos}")
            pos = match.end()
            if match.group("str") is not None:
                raw = match.group("str")[1:-1]
                self.tokens.append(("val", self._ESCAPE.sub(self._unescape, raw)))
            elif match.group("long") is not None:
                body = match.group("long")
                start = body.index("[", 1) + 1
                self.tokens.append(("val", body[start : len(body) - start]))
            elif match.group("num") is not None:
                num = match.group("num")
                value = int(num, 16) if num.lower().lstrip("-").startswith("0x") else float(num)
                self.tokens.append(("val", value))
            elif match.group("name") is not None:
                word = match.group("name")
                if word in ("true", "false"):
                    self.tokens.append(("val", word == "true"))
                elif word == "nil":
                    self.tokens.append(("val", None))
                else:
                    self.tokens.append(("name", word))
            elif match.group("sym") is not None:
                self.tokens.append(("sym", match.group("sym")))
        self.i = 0

    def _peek(self, offset: int = 0) -> tuple[str, object]:
        index = self.i + offset
        return self.tokens[index] if index < len(self.tokens) else ("eof", None)

    def _take(self, kind: str, value: object = None) -> object:
        token = self._peek()
        if token[0] != kind or (value is not None and token[1] != value):
            raise ValueError(f"malformed .lrtemplate near token {token!r}")
        self.i += 1
        return token[1]

    def value(self) -> object:
        kind, val = self._peek()
        if kind == "val":
            self.i += 1
            return val
        if (kind, val) == ("sym", "{"):
            return self.table()
        raise ValueError(f"malformed .lrtemplate near token {(kind, val)!r}")

    def table(self) -> object:
        self._take("sym", "{")
        keyed: dict[object, object] = {}
        items: list[object] = []
        while self._peek() != ("sym", "}"):
            if self._peek()[0] == "name" and self._peek(1) == ("sym", "="):
                key = self._take("name")
                self._take("sym", "=")
                keyed[key] = self.value()
            elif self._peek() == ("sym", "["):
                self._take("sym", "[")
                key = self.value()
                self._take("sym", "]")
                self._take("sym", "=")
                keyed[key] = self.value()
            else:
                items.append(self.value())
            if self._peek()[0] == "sym" and self._peek()[1] in (",", ";"):
                self.i += 1
        self._take("sym", "}")
        if keyed:
            keyed.update({index + 1: item for index, item in enumerate(items)})
            return keyed
        return items

    def document(self) -> object:
        if self._peek()[0] == "name" and self._peek()[1] == "return":
            self.i += 1
        elif self._peek()[0] == "name" and self._peek(1) == ("sym", "="):
            self.i += 2
        result = self.value()
        if self._peek()[0] != "eof":
            raise ValueError("unexpected content after the .lrtemplate table")
        return result


def _delocalize(text: object) -> str | None:
    """``"$$$/Key/Path=Default text"`` -> ``"Default text"``."""

    if not isinstance(text, str):
        return None
    return text.split("=", 1)[1] if text.startswith("$$$/") and "=" in text else text


def parse_lrtemplate(text: str) -> tuple[str | None, dict[str, object]]:
    """Develop settings of a classic ``.lrtemplate`` (data-only Lua) and its title."""

    doc = _Lua(text).document()
    if not isinstance(doc, dict):
        raise ValueError(".lrtemplate must hold a table")
    value = doc.get("value")
    settings = value.get("settings") if isinstance(value, dict) else None
    if not isinstance(settings, dict):
        raise ValueError(".lrtemplate has no value.settings table")
    return _delocalize(doc.get("title")), {str(k): v for k, v in settings.items()}


def read_preset_file(path: str | Path) -> tuple[str, dict[str, object]]:
    """Read a preset file -> (title, raw ``crs`` properties)."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in PRESET_SUFFIXES:
        raise ValueError(f"unsupported preset type {suffix!r}; use .xmp or .lrtemplate")
    if not path.is_file():
        raise ValueError(f"preset file not found: {path}")
    if path.stat().st_size > MAX_PRESET_BYTES:
        raise ValueError(f"preset file is too large: {path}")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    title, props = parse_xmp(text) if suffix == ".xmp" else parse_lrtemplate(text)
    return title or path.stem, props


# --- mapping -------------------------------------------------------------------------


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _curve(value: object) -> list[list[float]] | None:
    """Points from XMP items ``["0, 0", "255, 255"]`` or a flat Lua list ``[0, 0, 255, 255]``."""

    numbers: list[float] = []
    if isinstance(value, dict):  # Lua table with numeric keys
        value = [value[k] for k in sorted(k for k in value if isinstance(k, int))]
    if isinstance(value, list | tuple):
        for item in value:
            parts = item if isinstance(item, int | float) else str(item).replace(",", " ").split()
            if isinstance(parts, int | float):
                numbers.append(float(parts))
            else:
                numbers.extend(float(p) for p in parts)
    if len(numbers) < 4 or len(numbers) % 2:
        return None
    points = [[numbers[i], numbers[i + 1]] for i in range(0, len(numbers), 2)]
    if all(0.0 <= v <= 255.0 for p in points for v in p):
        return points
    return None


def _is_neutral(value: object) -> bool:
    if isinstance(value, list | tuple):
        # An empty Point Color slot is stored as a row of -1.
        numbers = [_number(v) for item in value for v in str(item).split(",")]
        return bool(numbers) and all(n == -1.0 for n in numbers)
    if isinstance(value, dict):
        return False
    number = _number(value)
    if number is not None:
        return number == 0.0
    return str(value).strip().lower() in ("", "false", "none", "as shot", "0")


def _parametric_curve(props: dict[str, object]) -> np.ndarray | None:
    """Lightroom's parametric (region) curve as a 0..1 LUT, LightCraft's model."""

    amounts = [_number(props.get(name)) or 0.0 for name in _PARAMETRIC]
    if not any(amounts):
        return None
    s1, s2, s3 = ((_number(props.get(name)) or default) / 100.0 for name, default in _SPLITS)
    x = np.linspace(0.0, 1.0, 1024)
    delta = np.zeros_like(x)
    for (a, b), amount in zip(((0.0, s1), (s1, s2), (s2, s3), (s3, 1.0)), amounts, strict=True):
        if amount:
            centre, half = (a + b) / 2.0, (b - a) * 0.75 + 0.05
            t = np.clip((x - centre) / half, -1.0, 1.0)
            delta += amount / 100.0 * 0.22 * (0.5 + 0.5 * np.cos(t * math.pi))
    curve = np.clip(x + delta * 4.0 * x * (1.0 - x), 0.0, 1.0)
    return np.maximum.accumulate(curve)


def _compose_master(points: list[list[float]] | None, parametric: np.ndarray) -> list[list[float]]:
    """``master(parametric(x))`` sampled as a 33-point curve (0..255)."""

    master = _monotone_lut(points) if points else np.linspace(0.0, 1.0, 1024)
    composed = np.interp(parametric, np.linspace(0.0, 1.0, len(master)), master)
    xs = np.linspace(0, 1023, 33).round().astype(int)
    return [[round(float(i / 1023 * 255), 3), round(float(composed[i] * 255), 3)] for i in xs]


def _white_balance(props: dict[str, object]) -> tuple[dict[str, float], str | None]:
    """Absolute Kelvin / named white balance -> relative Temp/Tint against the as-shot white."""

    mode = str(props.get("WhiteBalance", "")).strip()
    shot_temp = _number(props.get("AsShotTemperature"))
    target = _WB_PRESETS.get(mode) or (
        _number(props.get("Temperature")) if mode == "Custom" else None
    )
    if not target or not shot_temp:
        if _number(props.get("Temperature")):
            return {}, "absolute Temperature/Tint without an as-shot white: not applied"
        return {}, None
    temp = max(-100.0, min(100.0, (1e6 / shot_temp - 1e6 / target) * _TEMP_PER_MIRED))
    out = {"wb_temperature": round(temp, 2)} if abs(temp) > 0.5 else {}
    shot_tint, tint = _number(props.get("AsShotTint")), _number(props.get("Tint"))
    if mode == "Custom" and shot_tint is not None and tint is not None and tint != shot_tint:
        out["wb_tint"] = max(-100.0, min(100.0, tint - shot_tint))
    note = (
        f"white balance {mode} {target:.0f} K vs as shot {shot_temp:.0f} K approximated as "
        f"relative Temp {temp:+.0f}"
    )
    return out, note


def to_colorlab(props: dict[str, object], title: str = "Preset") -> LightroomPreset:
    """Translate raw ``crs`` properties into ColorLab parameters and a report."""

    params: dict[str, object] = {}
    applied: list[str] = []
    ignored: list[str] = []
    skipped: list[str] = []
    notes: list[str] = []
    for old, new in _FALLBACK.items():
        if old in props and new not in props:
            props = {**props, new: props[old]}

    for key, value in props.items():
        if key in _DIRECT:
            number = _number(value)
            if number is None:
                ignored.append(key)
            elif number != 0.0:
                params[_DIRECT[key]] = number
                applied.append(key)
        elif key in _CURVES:
            points = _curve(value)
            if points is None:
                ignored.append(key)
            elif any(abs(x - y) > 1e-6 for x, y in points):
                params[_CURVES[key]] = points
                applied.append(key)
        elif key in _METADATA or key in _FALLBACK:
            continue
        elif key == "Look":
            name = str(value.get("Name", "")) if isinstance(value, dict) else str(value)
            if name.strip().lower() not in _DEFAULT_PROFILES and name.strip():
                ignored.append(f"Look ({name})")
        elif not _is_neutral(value):
            (skipped if _NOT_COLOUR.match(key) else ignored).append(key)

    for amount_key, shape_keys in _EFFECT_SHAPES.items():
        if amount_key not in params:
            for shape in shape_keys:
                params.pop(shape, None)
    parametric = _parametric_curve(props)
    if parametric is not None:
        params["tone_curve"] = _compose_master(params.get("tone_curve"), parametric)  # type: ignore[arg-type]
        applied.append("ParametricCurve")
    wb, note = _white_balance(props)
    if wb and not any(key in params for key in ("wb_temperature", "wb_tint")):
        params.update(wb)
        applied.append("WhiteBalance")
    if note:
        notes.append(note)

    if _truthy(props.get("ConvertToGrayscale")):
        params["gray_enable"] = True
        applied.append("ConvertToGrayscale")
    balance = _number(props.get("SplitToningBalance", props.get("ColorGradeBalance")))
    if balance:
        for region in ("shadow", "mid", "highlight"):
            params[f"grade_{region}_bal"] = balance
        applied.append("SplitToningBalance")
    return LightroomPreset(
        title=title,
        params=params,
        applied=applied,
        ignored=sorted(set(ignored)),
        skipped=sorted(set(skipped)),
        notes=notes,
    )


def load_lightroom_preset(path: str | Path) -> LightroomPreset:
    """Read and translate a ``.xmp`` / ``.lrtemplate`` preset file."""

    title, props = read_preset_file(path)
    return to_colorlab(props, title)


# --- preset folder -------------------------------------------------------------------

PRESET_FOLDER = "lightroom_presets"


def _folder_paths():  # noqa: ANN202 - optional ComfyUI module
    try:
        import folder_paths  # type: ignore
    except ImportError:
        return None
    return folder_paths


# Presets bundled with the pack (written for PixelPro, listed after the user's own).
BUNDLED_PRESETS = Path(__file__).resolve().parents[1] / "presets" / "lightroom"


def preset_dirs() -> list[Path]:
    """Folders searched for presets: the user's folder first, then the bundled samples.

    The user's folder is ``<ComfyUI>/models/lightroom_presets`` (or its
    ``extra_model_paths.yaml`` mapping); favourites and recents are stored there.
    """

    fp = _folder_paths()
    if fp is None:
        user = [Path(__file__).resolve().parents[3] / "models" / PRESET_FOLDER]
    elif PRESET_FOLDER in getattr(fp, "folder_names_and_paths", {}):
        user = [Path(p) for p in fp.get_folder_paths(PRESET_FOLDER)] or [
            Path(fp.models_dir) / PRESET_FOLDER
        ]
    else:
        user = [Path(fp.models_dir) / PRESET_FOLDER]
    return [*user, BUNDLED_PRESETS]


def list_presets(dirs: list[Path] | None = None) -> list[str]:
    """Preset files under the preset folders (recursive), as sorted relative paths."""

    names: set[str] = set()
    for root in dirs if dirs is not None else preset_dirs():
        if root.is_dir():
            for path in root.rglob("*"):
                if path.suffix.lower() in PRESET_SUFFIXES and path.is_file():
                    names.add(path.relative_to(root).as_posix())
    return sorted(names, key=str.lower)


def resolve_preset(name: str, path: str = "", dirs: list[Path] | None = None) -> Path:
    """The file for an explicit ``path`` (wins when set) or a listed preset ``name``."""

    if path and path.strip():
        return Path(path.strip().strip('"')).expanduser()
    for root in dirs if dirs is not None else preset_dirs():
        candidate = (root / name).resolve()
        if candidate.is_file() and root.resolve() in candidate.parents:
            return candidate
    raise ValueError(
        f"preset {name!r} not found. Put .xmp / .lrtemplate files in "
        f"{preset_dirs()[0]} or fill preset_path."
    )
