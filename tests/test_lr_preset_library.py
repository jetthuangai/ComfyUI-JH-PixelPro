from __future__ import annotations

import importlib.util
import io
import shutil
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from core import lr_preset_library as lib

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "lr"
_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_lr_library"


@pytest.fixture()
def library(tmp_path: Path) -> list[Path]:
    (tmp_path / "Film").mkdir()
    shutil.copy(FIXTURES / "warm-film.xmp", tmp_path / "Film" / "Warm Film.xmp")
    shutil.copy(FIXTURES / "classic-bw.lrtemplate", tmp_path / "Classic BW.lrtemplate")
    return [tmp_path]


def test_collections_and_summaries(library: list[Path]) -> None:
    assert lib.collections(library) == [lib.ALL, lib.FAVORITES, lib.RECENT, "Film"]
    items = {item["name"]: item for item in lib.summaries(library)}
    assert items["Film/Warm Film.xmp"]["group"] == "Film"
    assert items["Film/Warm Film.xmp"]["title"] == "JH Warm Film"
    assert items["Film/Warm Film.xmp"]["unsupported"] == ["Look (Vintage 06)"]


def test_filter_by_collection_search_favorites_recent(library: list[Path]) -> None:
    names = ["Classic BW.lrtemplate", "Film/Warm Film.xmp"]
    meta = {"favorites": ["Film/Warm Film.xmp"], "recent": ["Classic BW.lrtemplate"]}
    assert lib.filter_presets(names, lib.ALL, meta=meta) == names
    assert lib.filter_presets(names, "Film", meta=meta) == ["Film/Warm Film.xmp"]
    assert lib.filter_presets(names, lib.FAVORITES, meta=meta) == ["Film/Warm Film.xmp"]
    assert lib.filter_presets(names, lib.RECENT, meta=meta) == ["Classic BW.lrtemplate"]
    assert lib.filter_presets(names, lib.ALL, "warm FILM", meta=meta) == ["Film/Warm Film.xmp"]


def test_favorites_and_recent_persist_in_folder(library: list[Path]) -> None:
    assert lib.set_favorite("Film/Warm Film.xmp", True, library) == ["Film/Warm Film.xmp"]
    assert lib.set_favorite("Film/Warm Film.xmp", False, library) == []
    with pytest.raises(ValueError):
        lib.set_favorite("../escape.xmp", True, library)
    for _ in range(3):
        lib.add_recent("Classic BW.lrtemplate", library)
    lib.add_recent("Film/Warm Film.xmp", library)
    meta = lib.load_meta(library)
    assert meta["recent"] == ["Film/Warm Film.xmp", "Classic BW.lrtemplate"]
    assert (library[0] / ".jh_presets.json").is_file()


def test_thumbnail_uses_chart_then_node_image(library: list[Path]) -> None:
    chart = lib.render_thumbnail("Film/Warm Film.xmp", "77", size=96, dirs=library)
    assert Image.open(io.BytesIO(chart)).size[0] == 96
    version = lib.remember_source("77", torch.full((1, 300, 200, 3), 0.5))
    assert version > 0 and lib.source_for("77")[1].shape[1:3] == (300, 200)
    on_image = lib.render_thumbnail("Film/Warm Film.xmp", "77", size=96, dirs=library)
    assert Image.open(io.BytesIO(on_image)).size == (64, 96)  # portrait source, long edge 96
    assert (
        lib.render_thumbnail("Film/Warm Film.xmp", "77", size=96, dirs=library) is on_image
    )  # cached


def test_thumbnail_rejects_unknown_preset(library: list[Path]) -> None:
    with pytest.raises(ValueError):
        lib.render_thumbnail("../../secret.xmp", dirs=library)


def test_contact_sheet(library: list[Path]) -> None:
    image = torch.rand(1, 120, 160, 3)
    sheet = lib.render_contact_sheet(
        image,
        ["Film/Warm Film.xmp", "Classic BW.lrtemplate"],
        columns=3,
        thumb_size=128,
        dirs=library,
    )
    assert sheet.shape[0] == 1 and sheet.shape[-1] == 3
    assert sheet.shape[2] == 3 * (128 + 6) + 6  # three tiles: original + 2 presets


def _load_pack():
    if _PACK_MODULE_NAME in sys.modules:
        return sys.modules[_PACK_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(
        _PACK_MODULE_NAME,
        _PACK_ROOT / "__init__.py",
        submodule_search_locations=[str(_PACK_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACK_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def test_sheet_node_registered_and_runs(
    monkeypatch: pytest.MonkeyPatch, library: list[Path]
) -> None:
    pack = _load_pack()
    cls = pack.NODE_CLASS_MAPPINGS["JHPixelProLightroomPresetSheet"]
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLightroomPresetSheet"] == (
        "Color: Lightroom Preset Contact Sheet"
    )
    module = sys.modules[cls.__module__]
    monkeypatch.setattr(module, "list_presets", lambda: lib.list_presets(library))
    monkeypatch.setattr(
        module,
        "render_contact_sheet",
        lambda image, names, **kw: lib.render_contact_sheet(image, names, dirs=library, **kw),
    )
    sheet, names = cls().render(torch.rand(1, 64, 64, 3), "Film", 2, 128, 10)
    assert names == "Film/Warm Film.xmp"
    assert sheet.shape[2] == 2 * (64 + 6) + 6  # 64 px image is never upscaled


def test_thumbnail_needs_matching_source_version(library: list[Path]) -> None:
    version = lib.remember_source("91", torch.full((1, 300, 200, 3), 0.5))
    same = lib.render_thumbnail("Film/Warm Film.xmp", "91", size=96, dirs=library, version=version)
    other = lib.render_thumbnail("Film/Warm Film.xmp", "91", size=96, dirs=library, version=0)
    assert Image.open(io.BytesIO(same)).size == (64, 96)  # the node's portrait image
    assert Image.open(io.BytesIO(other)).size == (96, 64)  # chart: another workflow's node 91
