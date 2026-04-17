"""ComfyUI-JH-PixelPro — GPU-accelerated pro image pack for ComfyUI.

Entry point được ComfyUI load.
Mỗi node concrete đăng ký qua ``NODE_CLASS_MAPPINGS`` ở module con.
"""

from __future__ import annotations

# Khởi tạo rỗng — Claude Code sẽ fill khi làm T-20260417-03.
NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

# Web extensions (JS/CSS) — chưa dùng ở Phase 1.
WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
