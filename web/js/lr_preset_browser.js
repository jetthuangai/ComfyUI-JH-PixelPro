// Visual preset browser for "Color: Lightroom Preset (XMP)".
//
// A thumbnail grid of the presets applied to the image this node last processed
// (a generated colour chart before the first run), with collections (All /
// Favorites / Recent / sub-folders), search, paging and favourite stars. Clicking a
// tile only sets the existing "preset" widget, so saved workflows and the API
// prompt are unchanged; every extra widget is drawn-only (serialize: false).
// Thumbnails come from /jh_pixelpro/lr/thumb and are cached server-side.

import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";

const NODE_NAME = "JHPixelProLightroomPreset";
const STATE_KEY = "jh_lr_ui";
const ROUTE = "/jh_pixelpro/lr";
const ROWS = 3;
const MIN_WIDTH = 420;
const ACCENT = "#4c9aff";
const WARN = "#f5a524";
const STAR = "#ffc53d";

function theme() {
  const lg = window.LiteGraph ?? {};
  return {
    bg: lg.WIDGET_BGCOLOR ?? "#222",
    outline: lg.WIDGET_OUTLINE_COLOR ?? "#666",
    text: lg.WIDGET_TEXT_COLOR ?? "#ddd",
    muted: lg.WIDGET_SECONDARY_TEXT_COLOR ?? "#999",
  };
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
  else ctx.rect(x, y, w, h);
}

function state(node) {
  node.properties ??= {};
  const s = (node.properties[STATE_KEY] ??= {});
  s.collection ??= "All";
  s.page ??= 0;
  s.open ??= true;
  return s;
}

function data(node) {
  return (node.__jhLr ??= { presets: [], favorites: [], recent: [], collections: ["All"], version: 0, images: new Map() });
}

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function refreshLayout(node) {
  const open = state(node).open;
  for (const w of node.widgets ?? []) {
    if (w.__jhLrGrid) w.hidden = !open;
  }
  const size = node.computeSize();
  node.setSize([Math.max(node.size[0], size[0], MIN_WIDTH), size[1]]);
  node.setDirtyCanvas?.(true, true);
}

function filtered(node) {
  const d = data(node);
  const s = state(node);
  const names = d.presets.map((p) => p.name);
  let picked;
  if (s.collection === "★ Favorites") picked = d.favorites.filter((n) => names.includes(n));
  else if (s.collection === "Recent") picked = d.recent.filter((n) => names.includes(n));
  else if (s.collection === "All") picked = names;
  else picked = names.filter((n) => n === s.collection || n.startsWith(`${s.collection}/`));
  const words = (widget(node, "jh_lr_search")?.value ?? "").toLowerCase().split(/\s+/).filter(Boolean);
  return picked.filter((n) => words.every((w) => n.toLowerCase().includes(w)));
}

function geometry(node, width) {
  const margin = 10;
  const gap = 8;
  const cols = Math.max(2, Math.floor((width - margin * 2 + gap) / (120 + gap)));
  const cell = (width - margin * 2 - gap * (cols - 1)) / cols;
  const thumbH = Math.round(cell * 0.72);
  const cellH = thumbH + 18;
  return { margin, gap, cols, cell, thumbH, cellH, perPage: cols * ROWS };
}

async function loadPresets(node) {
  try {
    const res = await api.fetchApi(`${ROUTE}/presets`);
    if (!res.ok) return;
    const json = await res.json();
    const d = data(node);
    Object.assign(d, {
      presets: json.presets ?? [],
      favorites: json.favorites ?? [],
      recent: json.recent ?? [],
      collections: json.collections ?? ["All"],
      folder: json.folder,
    });
    const combo = widget(node, "jh_lr_collection");
    if (combo) {
      combo.options.values = d.collections;
      if (!d.collections.includes(combo.value)) combo.value = "All";
    }
    // Keep the preset dropdown in sync with files added since ComfyUI started.
    const presetWidget = widget(node, "preset");
    if (presetWidget && d.presets.length) presetWidget.options.values = d.presets.map((p) => p.name);
    node.setDirtyCanvas?.(true, true);
  } catch (err) {
    console.warn("JH PixelPro: could not load Lightroom presets", err);
  }
}

function thumbnail(node, name, px) {
  const d = data(node);
  const engine = widget(node, "tone_engine")?.value ?? "local";
  const query = new URLSearchParams({ preset: name, node: String(node.id), v: String(d.version), size: String(px), engine });
  const url = api.apiURL(`${ROUTE}/thumb?${query}`);
  let img = d.images.get(url);
  if (!img) {
    img = new Image();
    img.onload = () => node.setDirtyCanvas?.(true, true);
    img.src = url;
    d.images.set(url, img);
    if (d.images.size > 400) d.images.delete(d.images.keys().next().value);
  }
  return img;
}

function selectPreset(node, name) {
  const w = widget(node, "preset");
  if (!w || w.value === name) return;
  const old = w.value;
  w.value = name;
  w.callback?.(name, app.canvas, node);
  node.onWidgetChanged?.("preset", name, old, w);
  node.graph?.incrementVersion?.();
}

async function toggleFavorite(node, name) {
  const d = data(node);
  const favorite = !d.favorites.includes(name);
  try {
    const res = await api.fetchApi(`${ROUTE}/favorite`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preset: name, favorite }),
    });
    if (res.ok) d.favorites = (await res.json()).favorites ?? d.favorites;
  } catch (err) {
    console.warn("JH PixelPro: could not save favourite", err);
  }
  node.setDirtyCanvas?.(true, true);
}

// --- drawn-only widgets --------------------------------------------------------------

function gridWidget() {
  return {
    name: "jh_lr_grid",
    type: "jh_lr_grid",
    value: null,
    serialize: false,
    options: { serialize: false }, // keeps it out of the API prompt too
    __jhLrGrid: true,
    _cells: [],
    computeSize(width) {
      const g = geometry(null, Number.isFinite(width) ? width : MIN_WIDTH);
      return [width ?? MIN_WIDTH, ROWS * g.cellH + (ROWS - 1) * g.gap + 4];
    },
    draw(ctx, node, width, y) {
      const t = theme();
      const s = state(node);
      const d = data(node);
      const g = geometry(node, width);
      const names = filtered(node);
      const pages = Math.max(1, Math.ceil(names.length / g.perPage));
      s.page = Math.min(s.page, pages - 1);
      const selected = widget(node, "preset")?.value;
      const px = Math.min(256, Math.round(g.cell * (window.devicePixelRatio || 1)));
      this._cells = [];
      ctx.save();
      ctx.font = "11px sans-serif";
      ctx.textBaseline = "middle";
      if (!names.length) {
        ctx.fillStyle = t.muted;
        ctx.textAlign = "center";
        const msg = d.presets.length ? "No preset matches" : `No presets yet: add .xmp files to ${d.folder ?? "models/lightroom_presets"}`;
        ctx.fillText(msg, width / 2, y + g.cellH);
        ctx.restore();
        return;
      }
      names.slice(s.page * g.perPage, (s.page + 1) * g.perPage).forEach((name, i) => {
        const col = i % g.cols;
        const row = Math.floor(i / g.cols);
        const x = g.margin + col * (g.cell + g.gap);
        const cy = y + 2 + row * (g.cellH + g.gap);
        roundRect(ctx, x, cy, g.cell, g.thumbH, 5);
        ctx.fillStyle = t.bg;
        ctx.fill();
        const img = thumbnail(node, name, px);
        if (img.complete && img.naturalWidth) {
          const scale = Math.min(g.cell / img.naturalWidth, g.thumbH / img.naturalHeight);
          const w = img.naturalWidth * scale;
          const h = img.naturalHeight * scale;
          ctx.save();
          roundRect(ctx, x, cy, g.cell, g.thumbH, 5);
          ctx.clip();
          ctx.drawImage(img, x + (g.cell - w) / 2, cy + (g.thumbH - h) / 2, w, h);
          ctx.restore();
        }
        const info = d.presets.find((p) => p.name === name) ?? {};
        if (name === selected) {
          ctx.lineWidth = 2.5;
          ctx.strokeStyle = ACCENT;
          roundRect(ctx, x - 1, cy - 1, g.cell + 2, g.thumbH + 2, 6);
          ctx.stroke();
        }
        // Favourite star (top right) and "not fully supported" badge (top left).
        const fav = d.favorites.includes(name);
        ctx.font = "14px sans-serif";
        ctx.textAlign = "center";
        ctx.fillStyle = "#0009";
        ctx.beginPath();
        ctx.arc(x + g.cell - 11, cy + 11, 9, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = fav ? STAR : "#fffa";
        ctx.fillText(fav ? "★" : "☆", x + g.cell - 11, cy + 12);
        if (info.unsupported?.length || info.error) {
          ctx.fillStyle = WARN;
          ctx.beginPath();
          ctx.arc(x + 10, cy + 10, 7, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "#000";
          ctx.font = "bold 11px sans-serif";
          ctx.fillText("!", x + 10, cy + 10.5);
        }
        ctx.font = "11px sans-serif";
        ctx.fillStyle = name === selected ? "#fff" : t.text;
        const label = (info.title ?? name.split("/").pop().replace(/\.(xmp|lrtemplate)$/i, "")).trim();
        let shown = label;
        while (shown.length > 3 && ctx.measureText(shown).width > g.cell - 4) shown = `${shown.slice(0, -2)}…`;
        ctx.fillText(shown, x + g.cell / 2, cy + g.thumbH + 9);
        this._cells.push({ x, y: cy - y, w: g.cell, h: g.thumbH + 16, name });
      });
      ctx.restore();
    },
    mouse(event, pos, node) {
      if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
      const localY = pos[1] - this.y;
      const cell = this._cells.find((c) => pos[0] >= c.x && pos[0] <= c.x + c.w && localY >= c.y && localY <= c.y + c.h);
      if (!cell) return false;
      if (pos[0] >= cell.x + cell.w - 22 && localY <= cell.y + 22) toggleFavorite(node, cell.name);
      else selectPreset(node, cell.name);
      node.setDirtyCanvas?.(true, true);
      return true;
    },
  };
}

function pagerWidget() {
  return {
    name: "jh_lr_pager",
    type: "jh_lr_pager",
    value: null,
    serialize: false,
    options: { serialize: false }, // keeps it out of the API prompt too
    _hits: [],
    computeSize: (width) => [width ?? MIN_WIDTH, 24],
    draw(ctx, node, width, y) {
      const t = theme();
      const s = state(node);
      const g = geometry(node, width);
      const names = filtered(node);
      const pages = Math.max(1, Math.ceil(names.length / g.perPage));
      const selected = data(node).presets.find((p) => p.name === widget(node, "preset")?.value);
      this._hits = [];
      ctx.save();
      ctx.font = "12px sans-serif";
      ctx.textBaseline = "middle";
      const button = (label, x, w, action, enabled = true) => {
        roundRect(ctx, x, y + 2, w, 20, 5);
        ctx.fillStyle = t.bg;
        ctx.fill();
        ctx.strokeStyle = t.outline;
        ctx.stroke();
        ctx.fillStyle = enabled ? t.text : t.outline;
        ctx.textAlign = "center";
        ctx.fillText(label, x + w / 2, y + 12);
        if (enabled) this._hits.push({ x, w, action });
      };
      let x = 10;
      if (s.open) {
        button("◀", x, 26, () => (s.page = Math.max(0, s.page - 1)), s.page > 0);
        x += 30;
        ctx.fillStyle = t.muted;
        ctx.textAlign = "center";
        ctx.fillText(`${s.page + 1}/${pages}`, x + 22, y + 12);
        x += 44;
        button("▶", x, 26, () => (s.page = Math.min(pages - 1, s.page + 1)), s.page < pages - 1);
        x += 34;
      }
      ctx.textAlign = "left";
      ctx.fillStyle = selected?.unsupported?.length ? WARN : t.muted;
      const note = selected
        ? selected.unsupported?.length
          ? `! not supported: ${selected.unsupported.join(", ")}`
          : `${names.length} presets`
        : `${names.length} presets`;
      let shown = note;
      const room = width - x - 120;
      while (shown.length > 4 && ctx.measureText(shown).width > room) shown = `${shown.slice(0, -2)}…`;
      ctx.fillText(shown, x, y + 12);
      button("⟳", width - 108, 26, () => loadPresets(node));
      button(s.open ? "Hide grid" : "Show grid", width - 78, 68, () => {
        s.open = !s.open;
        refreshLayout(node);
      });
      ctx.restore();
    },
    mouse(event, pos, node) {
      if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
      const hit = this._hits.find((h) => pos[0] >= h.x && pos[0] <= h.x + h.w);
      if (!hit) return false;
      hit.action();
      node.setDirtyCanvas?.(true, true);
      return true;
    },
  };
}

// --- wiring --------------------------------------------------------------------------

function placeAfter(node, added, afterName) {
  const list = node.widgets;
  list.splice(list.indexOf(added), 1);
  const at = list.findIndex((w) => w.name === afterName);
  list.splice(at < 0 ? list.length : at + 1, 0, added);
}

function setup(node) {
  if (node.__jhLrBrowser) return;
  node.__jhLrBrowser = true;
  const s = state(node);
  const combo = node.addWidget("combo", "jh_lr_collection", s.collection, (v) => {
    s.collection = v;
    s.page = 0;
    node.setDirtyCanvas?.(true, true);
  }, { values: ["All"], serialize: false });
  combo.label = "Collection";
  combo.serialize = false;
  const search = node.addWidget("text", "jh_lr_search", s.search ?? "", (v) => {
    s.search = v;
    s.page = 0;
    node.setDirtyCanvas?.(true, true);
  }, { serialize: false });
  search.label = "Search";
  search.serialize = false;
  const grid = node.addCustomWidget(gridWidget());
  const pager = node.addCustomWidget(pagerWidget());
  placeAfter(node, combo, "preset");
  placeAfter(node, search, "jh_lr_collection");
  placeAfter(node, grid, "jh_lr_search");
  placeAfter(node, pager, "jh_lr_grid");
  combo.__jhLrGrid = false;
  refreshLayout(node);
  loadPresets(node);
}

app.registerExtension({
  name: "JH.PixelPro.LightroomPresetBrowser",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const vueNodes = () => app.ui?.settings?.getSettingValue?.("Comfy.VueNodes.Enabled") === true;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      if (!vueNodes()) setup(this);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      if (this.__jhLrBrowser) {
        const s = state(this);
        const combo = widget(this, "jh_lr_collection");
        const search = widget(this, "jh_lr_search");
        if (combo) combo.value = s.collection;
        data(this).version = s.source_version ?? 0;
        if (search) search.value = s.search ?? "";
        requestAnimationFrame(() => refreshLayout(this));
      }
      return result;
    };

    // After a run the server holds this node's image: redraw thumbnails on it.
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      onExecuted?.apply(this, arguments);
      const version = output?.jh_lr_source?.[0];
      if (this.__jhLrBrowser && version !== undefined) {
        const d = data(this);
        d.version = version;
        state(this).source_version = version; // reopening the workflow keeps its thumbnails
        d.images.clear();
        loadPresets(this); // recents changed
      }
    };
  },
});
