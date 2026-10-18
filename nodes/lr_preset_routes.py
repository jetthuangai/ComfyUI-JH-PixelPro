"""HTTP routes for the Lightroom Preset browser (thumbnails, list, favourites).

Registered on ComfyUI's PromptServer at import time; a no-op outside ComfyUI (tests).
Only preset names from the preset folder listing are accepted, never client paths.
"""

from __future__ import annotations

import asyncio
import logging

from ..core.lr_preset import preset_dirs
from ..core.lr_preset_library import (
    collections,
    load_meta,
    render_thumbnail,
    set_favorite,
    summaries,
)

logger = logging.getLogger(__name__)
_PREFIX = "/jh_pixelpro/lr"


def register_routes() -> bool:
    try:
        from aiohttp import web
        from server import PromptServer  # type: ignore
    except ImportError:
        return False
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return False
    routes = instance.routes

    @routes.get(f"{_PREFIX}/presets")
    async def presets(_request: web.Request) -> web.Response:
        items = await asyncio.get_running_loop().run_in_executor(None, summaries)
        meta = load_meta()
        return web.json_response(
            {
                "presets": items,
                "collections": collections(),
                "favorites": meta["favorites"],
                "recent": meta["recent"],
                "folder": str(preset_dirs()[0]),
            }
        )

    @routes.get(f"{_PREFIX}/thumb")
    async def thumb(request: web.Request) -> web.Response:
        query = request.rel_url.query
        name = query.get("preset", "")
        node = query.get("node") or None
        engine = query.get("engine", "local")
        if engine not in ("local", "legacy"):
            engine = "local"
        try:
            size = int(query.get("size", "192"))
            version = int(query.get("v", "0"))
        except ValueError:
            size, version = 192, 0
        try:
            data = await asyncio.get_running_loop().run_in_executor(
                None, render_thumbnail, name, node, size, engine, None, version
            )
        except (ValueError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=404)
        return web.Response(
            body=data, content_type="image/jpeg", headers={"Cache-Control": "no-cache"}
        )

    @routes.post(f"{_PREFIX}/favorite")
    async def favorite(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            favs = set_favorite(str(body["preset"]), bool(body.get("favorite", True)))
        except (ValueError, KeyError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response({"favorites": favs})

    logger.info("JH PixelPro: Lightroom preset browser routes registered")
    return True
