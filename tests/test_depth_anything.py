from __future__ import annotations

import builtins
from pathlib import Path

import cv2
import pytest
import torch

from core import depth_anything
from core.depth_anything import estimate_depth, network_size

_SAMPLE = Path(__file__).resolve().parents[1] / "workflows" / "sample_portrait.jpg"


def _has_model() -> bool:
    return depth_anything.model_path("small").is_file()


needs_model = pytest.mark.skipif(not _has_model(), reason="depth-anything-v2-small.onnx missing")


def _portrait() -> torch.Tensor:
    bgr = cv2.imread(str(_SAMPLE))
    assert bgr is not None, "sample_portrait.jpg missing"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).float().div(255.0).unsqueeze(0)


def test_network_size_multiples_of_14() -> None:
    for h, w in [(100, 100), (1080, 1920), (777, 333), (2048, 2048), (5, 900)]:
        nh, nw = network_size(h, w, 518)
        assert nh % 14 == 0 and nw % 14 == 0 and nh >= 14 and nw >= 14
        assert abs(max(nh, nw) - 518) <= 14


def test_network_size_keeps_aspect() -> None:
    nh, nw = network_size(1000, 2000, 518)
    assert (nh, nw) == (252, 518)


def test_model_path_and_validation() -> None:
    assert depth_anything.model_path("base").name == "depth-anything-v2-base.onnx"
    assert depth_anything.model_path("small").parent.name == "depth_anything"
    with pytest.raises(ValueError, match="model"):
        depth_anything.model_path("huge")


def test_validation_errors() -> None:
    img = torch.rand(1, 32, 32, 3)
    with pytest.raises(TypeError):
        estimate_depth("x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        estimate_depth(torch.rand(32, 32, 3))
    with pytest.raises(ValueError, match="shape"):
        estimate_depth(torch.rand(1, 32, 32, 4))
    with pytest.raises(ValueError, match="floating"):
        estimate_depth(torch.zeros(1, 32, 32, 3, dtype=torch.uint8))
    with pytest.raises(ValueError, match="resolution"):
        estimate_depth(img, resolution=100)
    with pytest.raises(ValueError, match="resolution"):
        estimate_depth(img, resolution=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model"):
        estimate_depth(img, model="huge")


def test_missing_onnxruntime_raises_clear_error(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("no onnxruntime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="onnxruntime-gpu"):
        estimate_depth(torch.rand(1, 32, 32, 3))


def _fake_folder_paths(monkeypatch, tmp_path, *, registered):
    import sys
    import types

    module = types.ModuleType("folder_paths")
    module.models_dir = str(tmp_path / "models")

    def get_folder_paths(name):
        if name not in registered:
            raise KeyError(name)
        return [str(p) for p in registered[name]]

    module.get_folder_paths = get_folder_paths
    monkeypatch.setitem(sys.modules, "folder_paths", module)


def test_model_dir_uses_registered_folder_and_never_touches_sys_path(monkeypatch, tmp_path) -> None:
    import sys

    extra = tmp_path / "extra" / "da"
    _fake_folder_paths(monkeypatch, tmp_path, registered={"depth_anything": [extra]})
    before = list(sys.path)
    assert depth_anything.model_dir() == extra
    assert depth_anything.model_path("small") == extra / "depth-anything-v2-small.onnx"
    assert sys.path == before
    assert not extra.exists()


def test_model_dir_falls_back_to_models_dir_then_pack_relative(monkeypatch, tmp_path) -> None:
    import sys

    _fake_folder_paths(monkeypatch, tmp_path, registered={})
    assert depth_anything.model_dir() == tmp_path / "models" / "depth_anything"
    assert not (tmp_path / "models").exists()
    monkeypatch.setitem(sys.modules, "folder_paths", None)  # import raises ImportError
    expected = Path(depth_anything.__file__).resolve().parents[3] / "models" / "depth_anything"
    assert depth_anything.model_dir() == expected


def test_model_path_prefers_existing_file_across_folders(monkeypatch, tmp_path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    second.mkdir()
    (second / "depth-anything-v2-base.onnx").write_bytes(b"x")
    _fake_folder_paths(monkeypatch, tmp_path, registered={"depth_anything": [first, second]})
    assert depth_anything.model_path("base") == second / "depth-anything-v2-base.onnx"
    assert depth_anything.model_path("small") == first / "depth-anything-v2-small.onnx"


class _FakeResponse:
    def __init__(self, data: bytes, length: int | None = None):
        self._data = data
        self.headers = {"Content-Length": str(len(data) if length is None else length)}
        self._pos = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        chunk = self._data[self._pos : self._pos + n]
        self._pos += n
        return chunk


def test_download_is_pinned_creates_folder_late_and_verifies_sha(monkeypatch, tmp_path) -> None:
    import hashlib

    target = tmp_path / "models" / "depth_anything"
    _fake_folder_paths(monkeypatch, tmp_path, registered={"depth_anything": [target]})
    payload = b"fake-onnx" * 100
    monkeypatch.setitem(depth_anything._SHA256, "small", hashlib.sha256(payload).hexdigest())
    seen = {}

    def fake_urlopen(url, timeout=None):
        seen["url"], seen["timeout"] = url, timeout
        return _FakeResponse(payload)

    monkeypatch.setattr(depth_anything.urllib.request, "urlopen", fake_urlopen)
    path = depth_anything._ensure_model_file("small")
    assert path.read_bytes() == payload and not path.with_name(path.name + ".part").exists()
    assert "/resolve/main/" not in seen["url"]
    assert depth_anything._REVISIONS["small"] in seen["url"]
    assert seen["timeout"] and seen["timeout"] > 0


def test_download_rejects_bad_hash_and_truncation(monkeypatch, tmp_path) -> None:
    target = tmp_path / "da"
    _fake_folder_paths(monkeypatch, tmp_path, registered={"depth_anything": [target]})
    monkeypatch.setattr(
        depth_anything.urllib.request, "urlopen", lambda url, timeout=None: _FakeResponse(b"bad")
    )
    with pytest.raises(RuntimeError, match="SHA256"):
        depth_anything._ensure_model_file("small")
    monkeypatch.setattr(
        depth_anything.urllib.request,
        "urlopen",
        lambda url, timeout=None: _FakeResponse(b"abc", length=10),
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        depth_anything._ensure_model_file("base")  # no hash pinned: size check still applies
    assert list(target.iterdir()) == []  # no .part or final file left behind


def test_revisions_are_commit_hashes() -> None:
    for size in depth_anything.MODEL_SIZES:
        rev = depth_anything._REVISIONS[size]
        assert len(rev) == 40 and set(rev) <= set("0123456789abcdef")
        assert rev in depth_anything._MODEL_URL.format(size=size, revision=rev)


class _FakeSession:
    pass


def _fake_ort(monkeypatch, *, cuda: bool, cuda_fails: bool = False):
    calls = []
    providers_available = ["CPUExecutionProvider"]
    if cuda:
        providers_available.insert(0, "CUDAExecutionProvider")

    class Ort:
        @staticmethod
        def get_available_providers():
            return providers_available

        @staticmethod
        def InferenceSession(path, providers):  # noqa: N802 - mirrors onnxruntime API
            calls.append(providers)
            if cuda_fails and isinstance(providers[0], tuple):
                raise RuntimeError("no cuda")
            return _FakeSession()

    monkeypatch.setattr(depth_anything, "_import_onnxruntime", lambda: Ort)
    depth_anything.unload()
    return calls


def test_session_cache_single_entry_arena_option_and_unload(monkeypatch, tmp_path) -> None:
    calls = _fake_ort(monkeypatch, cuda=True)
    a, b = tmp_path / "a.onnx", tmp_path / "b.onnx"
    first = depth_anything._get_session(a)
    assert depth_anything._get_session(a) is first and len(calls) == 1
    name, options = calls[0][0]
    assert name == "CUDAExecutionProvider"
    assert options["arena_extend_strategy"] == "kSameAsRequested"
    depth_anything._get_session(b)
    assert len(depth_anything._SESSIONS) == 1
    depth_anything.unload()
    assert depth_anything._SESSIONS == {}


def test_cuda_failure_falls_back_to_cpu_and_is_cached(monkeypatch, tmp_path) -> None:
    calls = _fake_ort(monkeypatch, cuda=True, cuda_fails=True)
    path = tmp_path / "m.onnx"
    session = depth_anything._get_session(path)
    assert depth_anything._get_session(path) is session
    assert len(calls) == 2  # one failed CUDA attempt + one CPU session, no retry
    assert calls[1] == ["CPUExecutionProvider"]
    depth_anything.unload()


@needs_model
def test_shapes_range_batch() -> None:
    img = torch.rand(2, 70, 90, 3)
    depth = estimate_depth(img, "small", 266)
    assert depth.shape == (2, 70, 90)
    assert depth.dtype == torch.float32
    assert float(depth.min()) >= 0.0 and float(depth.max()) <= 1.0
    assert torch.allclose(depth.amax(dim=(1, 2)), torch.ones(2), atol=1e-5)


@needs_model
def test_portrait_near_shoulder_nearer_than_far_hair_corner() -> None:
    # Tight close-up: the shoulder (bottom-left) is the nearest surface, the hair in the top-left
    # corner is farther. (The face sits in between, so it is not used as the "near" reference.)
    img = _portrait()
    depth = estimate_depth(img, "small", 518)[0]
    h, w = depth.shape
    shoulder = depth[int(0.65 * h) : int(0.85 * h), int(0.1 * w) : int(0.4 * w)].mean()
    hair = depth[: h // 10, : w // 10].mean()
    assert float(shoulder) > float(hair) + 0.2
    assert float(depth.std()) > 0.1


@needs_model
def test_resolution_changes_network_input_only() -> None:
    img = torch.rand(1, 60, 80, 3)
    a = estimate_depth(img, "small", 266)
    b = estimate_depth(img, "small", 336)
    assert a.shape == b.shape == (1, 60, 80)
