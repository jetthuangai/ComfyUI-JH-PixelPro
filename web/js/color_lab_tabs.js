// Tabbed layout for "Color: ColorLab (ACR)".
//
// The node has ~60 sliders. This extension only changes how they are shown:
// a tab bar (Basic / HSL / Grading / B&W / Options), an 8-colour picker for HSL
// and three colour wheels for Color Grading, so a tab shows at most ~17 rows.
// Every slider stays a normal widget and is still saved in widgets_values in the
// original order; the extra widgets are drawn-only (serialize: false), so saved
// workflows load and run exactly as before. UI state lives in node.properties.

import { app } from "../../../scripts/app.js";

const NODE_NAME = "JHPixelProColorLab";
const STATE_KEY = "jh_colorlab_ui";

const TABS = [
  { id: "basic", label: "Basic" },
  { id: "hsl", label: "HSL" },
  { id: "grading", label: "Grading" },
  { id: "bw", label: "B&W" },
  { id: "options", label: "Options" },
];

const HSL_COLORS = [
  { id: "red", label: "Red", css: "#e5484d" },
  { id: "orange", label: "Orange", css: "#f76b15" },
  { id: "yellow", label: "Yellow", css: "#ffc53d" },
  { id: "green", label: "Green", css: "#46a758" },
  { id: "aqua", label: "Aqua", css: "#12a594" },
  { id: "blue", label: "Blue", css: "#3e63dd" },
  { id: "purple", label: "Purple", css: "#8e4ec6" },
  { id: "magenta", label: "Magenta", css: "#d6409f" },
];

const GRADE_REGIONS = [
  { id: "shadow", label: "Shadows" },
  { id: "mid", label: "Midtones" },
  { id: "highlight", label: "Highlights" },
];

const BASIC_LABELS = {
  basic_exposure: "Exposure",
  basic_contrast: "Contrast",
  basic_highlights: "Highlights",
  basic_shadows: "Shadows",
  basic_whites: "Whites",
  basic_blacks: "Blacks",
  basic_texture: "Texture",
  basic_clarity: "Clarity",
  basic_dehaze: "Dehaze",
  basic_vibrance: "Vibrance",
  basic_saturation: "Saturation",
  wb_temperature: "Temp",
  wb_tint: "Tint",
  exposure_mode: "Exposure mode",
  tone_engine: "Tone engine",
  gray_enable: "Convert to B&W",
};

// Section titles drawn above the first widget of each group.
const SECTIONS = {
  basic_exposure: "Light",
  basic_texture: "Presence",
  wb_temperature: "White balance",
  gray_red: "Black & white mix",
};

const ROW_H = 26;
const MIN_WIDTH = 340; // room for five tab labels and Reset
const ACCENT = "#4c9aff";

function tabOf(name) {
  if (name.startsWith("basic_") || name.startsWith("wb_")) return "basic";
  if (name.startsWith("hsl_")) return "hsl";
  if (name.startsWith("grade_")) return "grading";
  if (name.startsWith("gray_")) return "bw";
  if (name === "exposure_mode" || name === "tone_engine") return "options";
  return null;
}

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
  s.tab ??= "basic";
  s.hsl ??= "red";
  s.grade ??= "shadow";
  return s;
}

function valueWidgets(node) {
  return (node.widgets ?? []).filter((w) => w.serialize !== false && tabOf(w.name));
}

function isModified(node, w) {
  const def = node.__jhDefaults?.[w.name];
  if (def === undefined) return false;
  if (typeof def === "number") return Math.abs(Number(w.value) - def) > 1e-9;
  return w.value !== def;
}

function tabModified(node, tab) {
  return valueWidgets(node).some((w) => tabOf(w.name) === tab && isModified(node, w));
}

function isShown(node, w) {
  const s = state(node);
  if (w.__jhTab) return w.__jhTab === "*" || w.__jhTab === s.tab;
  const tab = tabOf(w.name ?? "");
  if (!tab) return true; // not ours (e.g. a widget added by another extension)
  if (tab !== s.tab) return false;
  if (tab === "hsl") return w.name.startsWith(`hsl_${s.hsl}_`);
  if (tab === "grading") return w.name.startsWith(`grade_${s.grade}_`);
  return true;
}

function refresh(node) {
  for (const w of node.widgets ?? []) {
    if (w.__jhTab || tabOf(w.name ?? "")) w.hidden = !isShown(node, w);
  }
  const size = node.computeSize();
  node.setSize([Math.max(node.size[0], size[0], MIN_WIDTH), size[1]]);
  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas(true, true);
}

function resetTab(node) {
  const s = state(node);
  for (const w of valueWidgets(node)) {
    if (tabOf(w.name) !== s.tab) continue;
    // HSL / Grading reset only the colour or region on screen.
    if (s.tab === "hsl" && !w.name.startsWith(`hsl_${s.hsl}_`)) continue;
    if (s.tab === "grading" && !w.name.startsWith(`grade_${s.grade}_`)) continue;
    const def = node.__jhDefaults?.[w.name];
    if (def === undefined || w.value === def) continue;
    w.value = def;
    w.callback?.(def, app.canvas, node);
  }
  refresh(node);
}

function isDown(event) {
  return event.type === "pointerdown" || event.type === "mousedown";
}

function openFirstEdited(node) {
  const s = state(node);
  const edited = valueWidgets(node).filter((w) => isModified(node, w));
  const first = TABS.find((t) => edited.some((w) => tabOf(w.name) === t.id));
  if (!first) return;
  s.tab = first.id;
  const hsl = HSL_COLORS.find((c) => edited.some((w) => w.name.startsWith(`hsl_${c.id}_`)));
  if (hsl) s.hsl = hsl.id;
  const grade = GRADE_REGIONS.find((g) => edited.some((w) => w.name.startsWith(`grade_${g.id}_`)));
  if (grade) s.grade = grade.id;
}

// --- drawn-only widgets --------------------------------------------------------------

function tabBarWidget() {
  return {
    name: "jh_tabs",
    type: "jh_colorlab_tabs",
    value: null,
    serialize: false,
    options: { serialize: false }, // keeps it out of the API prompt too
    __jhTab: "*",
    computeSize: (width) => [width, ROW_H + 2],
    _hits: [],
    draw(ctx, node, width, y) {
      const t = theme();
      const s = state(node);
      const margin = 10;
      const resetW = 54;
      const tabsW = width - margin * 2 - resetW - 6;
      const segW = tabsW / TABS.length;
      this._hits = [];
      ctx.save();
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      roundRect(ctx, margin, y + 1, tabsW, ROW_H, 6);
      ctx.fillStyle = t.bg;
      ctx.fill();
      ctx.strokeStyle = t.outline;
      ctx.stroke();
      TABS.forEach((tab, i) => {
        const x = margin + i * segW;
        if (tab.id === s.tab) {
          roundRect(ctx, x + 2, y + 3, segW - 4, ROW_H - 4, 5);
          ctx.fillStyle = ACCENT;
          ctx.fill();
        }
        ctx.fillStyle = tab.id === s.tab ? "#fff" : t.text;
        ctx.fillText(tab.label, x + segW / 2, y + 1 + ROW_H / 2);
        if (tabModified(node, tab.id)) {
          ctx.beginPath();
          ctx.arc(x + segW - 7, y + 7, 2.5, 0, Math.PI * 2);
          ctx.fillStyle = tab.id === s.tab ? "#fff" : ACCENT;
          ctx.fill();
        }
        this._hits.push({ x, w: segW, action: () => (s.tab = tab.id) });
      });
      const rx = margin + tabsW + 6;
      roundRect(ctx, rx, y + 1, resetW, ROW_H, 6);
      ctx.fillStyle = t.bg;
      ctx.fill();
      ctx.strokeStyle = t.outline;
      ctx.stroke();
      ctx.fillStyle = t.muted;
      ctx.fillText("Reset", rx + resetW / 2, y + 1 + ROW_H / 2);
      this._hits.push({ x: rx, w: resetW, action: () => resetTab(node), reset: true });
      ctx.restore();
    },
    mouse(event, pos, node) {
      if (!isDown(event)) return false;
      const hit = this._hits.find((h) => pos[0] >= h.x && pos[0] <= h.x + h.w);
      if (!hit) return false;
      hit.action();
      if (!hit.reset) refresh(node);
      return true;
    },
  };
}

function hslPickerWidget() {
  return {
    name: "jh_hsl_colors",
    type: "jh_colorlab_hsl",
    value: null,
    serialize: false,
    options: { serialize: false }, // keeps it out of the API prompt too
    __jhTab: "hsl",
    computeSize: (width) => [width, ROW_H + 18],
    _hits: [],
    draw(ctx, node, width, y) {
      const t = theme();
      const s = state(node);
      const margin = 14;
      const step = (width - margin * 2) / HSL_COLORS.length;
      const r = Math.min(10, step / 2 - 3);
      this._hits = [];
      ctx.save();
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      HSL_COLORS.forEach((c, i) => {
        const cx = margin + step * i + step / 2;
        const cy = y + 4 + r;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = c.css;
        ctx.fill();
        if (c.id === s.hsl) {
          ctx.lineWidth = 2;
          ctx.strokeStyle = "#fff";
          ctx.beginPath();
          ctx.arc(cx, cy, r + 3, 0, Math.PI * 2);
          ctx.stroke();
        }
        const changed = ["hue", "sat", "lum"].some((p) => {
          const w = node.widgets?.find((x) => x.name === `hsl_${c.id}_${p}`);
          return w && isModified(node, w);
        });
        if (changed) {
          ctx.beginPath();
          ctx.arc(cx, cy + r + 6, 2, 0, Math.PI * 2);
          ctx.fillStyle = ACCENT;
          ctx.fill();
        }
        this._hits.push({ x: cx - step / 2, w: step, id: c.id });
      });
      const sel = HSL_COLORS.find((c) => c.id === s.hsl);
      ctx.font = "11px sans-serif";
      ctx.fillStyle = t.muted;
      ctx.fillText(`${sel.label}: hue / saturation / luminance`, width / 2, y + ROW_H + 10);
      ctx.restore();
    },
    mouse(event, pos, node) {
      if (!isDown(event)) return false;
      const hit = this._hits.find((h) => pos[0] >= h.x && pos[0] <= h.x + h.w);
      if (!hit) return false;
      state(node).hsl = hit.id;
      refresh(node);
      return true;
    },
  };
}

function hsvCss(hue, sat) {
  const h = ((hue % 360) + 360) % 360;
  const sv = Math.max(0, Math.min(1, sat / 100));
  const v = 0.9;
  const f = (n) => {
    const k = (n + h / 60) % 6;
    return v - v * sv * Math.max(0, Math.min(k, 4 - k, 1));
  };
  return `rgb(${[f(5), f(3), f(1)].map((x) => Math.round(x * 255)).join(",")})`;
}

function gradeWidget(node, region, part) {
  return node.widgets?.find((w) => w.name === `grade_${region}_${part}`);
}

function setWidgetValue(node, widget, value) {
  if (!widget || widget.value === value) return;
  const old = widget.value;
  widget.value = value;
  widget.callback?.(value, app.canvas, node);
  node.onWidgetChanged?.(widget.name, value, old, widget);
  node.graph?.incrementVersion?.();
}

// Three Lightroom-style wheels (Shadows / Midtones / Highlights). A wheel only
// edits the existing grade_<region>_hue / _sat widgets (angle = hue, distance from
// the centre = saturation), so saved workflows and the API prompt are unchanged.
function gradeWheelsWidget() {
  const LABEL_H = 16;
  const geometry = (width) => {
    // Layout may call computeSize() without a width; fall back to the minimum node width.
    if (!Number.isFinite(width)) width = MIN_WIDTH;
    const margin = 10;
    const cell = (width - margin * 2) / GRADE_REGIONS.length;
    const radius = Math.max(24, Math.min(46, cell / 2 - 8));
    return { margin, cell, radius };
  };
  return {
    name: "jh_grade_wheels",
    type: "jh_colorlab_wheels",
    value: null,
    serialize: false,
    options: { serialize: false }, // keeps it out of the API prompt too
    __jhTab: "grading",
    computeSize(width) {
      return [Number.isFinite(width) ? width : MIN_WIDTH, LABEL_H + geometry(width).radius * 2 + 14];
    },
    _centres: [],
    draw(ctx, node, width, y) {
      const t = theme();
      const s = state(node);
      const { margin, cell, radius } = geometry(width);
      this._centres = [];
      ctx.save();
      GRADE_REGIONS.forEach((g, i) => {
        const cx = margin + cell * i + cell / 2;
        const cy = y + LABEL_H + 6 + radius;
        this._centres.push({ id: g.id, cx, cy, radius });

        // Hue ring as a conic gradient (0 deg = red at 3 o'clock, counter-clockwise),
        // whitened toward the centre where saturation is 0.
        if (ctx.createConicGradient) {
          const conic = ctx.createConicGradient(0, cx, cy);
          for (let k = 0; k <= 12; k++) conic.addColorStop(k / 12, hsvCss(360 - k * 30, 100));
          ctx.fillStyle = conic;
        } else {
          ctx.fillStyle = "#888";
        }
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fill();
        const fade = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
        fade.addColorStop(0, "rgba(230,230,230,1)");
        fade.addColorStop(1, "rgba(230,230,230,0)");
        ctx.fillStyle = fade;
        ctx.fill();
        ctx.lineWidth = g.id === s.grade ? 2.5 : 1;
        ctx.strokeStyle = g.id === s.grade ? ACCENT : t.outline;
        ctx.stroke();

        const hue = Number(gradeWidget(node, g.id, "hue")?.value ?? 0);
        const sat = Number(gradeWidget(node, g.id, "sat")?.value ?? 0);
        const angle = (hue * Math.PI) / 180;
        const mx = cx + Math.cos(angle) * radius * (sat / 100);
        const my = cy - Math.sin(angle) * radius * (sat / 100);
        ctx.beginPath();
        ctx.arc(mx, my, 5, 0, Math.PI * 2);
        ctx.fillStyle = hsvCss(hue, sat);
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#fff";
        ctx.stroke();
        ctx.lineWidth = 1;
        ctx.strokeStyle = "#000a";
        ctx.beginPath();
        ctx.arc(mx, my, 6.5, 0, Math.PI * 2);
        ctx.stroke();

        const edited = ["hue", "sat", "lum", "bal"].some((p) => {
          const w = gradeWidget(node, g.id, p);
          return w && isModified(node, w);
        });
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = g.id === s.grade ? "#fff" : t.text;
        ctx.fillText(g.label, cx, y + LABEL_H / 2 + 2);
        if (edited) {
          const tw = ctx.measureText(g.label).width;
          ctx.beginPath();
          ctx.arc(cx + tw / 2 + 6, y + LABEL_H / 2 + 2, 2.5, 0, Math.PI * 2);
          ctx.fillStyle = ACCENT;
          ctx.fill();
        }
      });
      ctx.restore();
    },
    _pick(node, local) {
      return this._centres.find((c) => Math.hypot(local[0] - c.cx, local[1] - c.cy) <= c.radius + 8);
    },
    _apply(node, wheel, local) {
      const dx = local[0] - wheel.cx;
      const dy = wheel.cy - local[1];
      const dist = Math.hypot(dx, dy);
      const sat = Math.round(Math.min(100, (dist / wheel.radius) * 100));
      if (sat <= 3) {
        // The centre means "no tint": keep the hue, drop the saturation.
        setWidgetValue(node, gradeWidget(node, wheel.id, "sat"), 0);
      } else {
        const hue = Math.round(((Math.atan2(dy, dx) * 180) / Math.PI + 360) % 360);
        setWidgetValue(node, gradeWidget(node, wheel.id, "hue"), hue);
        setWidgetValue(node, gradeWidget(node, wheel.id, "sat"), sat);
      }
      node.setDirtyCanvas?.(true, true);
    },
    onPointerDown(pointer, node, canvas) {
      const local = [canvas.graph_mouse[0] - node.pos[0], canvas.graph_mouse[1] - node.pos[1]];
      const wheel = this._pick(node, local);
      if (!wheel) return false;
      const s = state(node);
      if (s.grade !== wheel.id) {
        s.grade = wheel.id;
        refresh(node);
      }
      this._apply(node, wheel, local);
      pointer.onDrag = (e) => this._apply(node, wheel, [e.canvasX - node.pos[0], e.canvasY - node.pos[1]]);
      return true;
    },
  };
}

function sectionWidget(title, tab) {
  return {
    name: `jh_section_${title}`,
    type: "jh_colorlab_section",
    value: null,
    serialize: false,
    options: { serialize: false }, // keeps it out of the API prompt too
    __jhTab: tab,
    computeSize: (width) => [width, 18],
    draw(ctx, node, width, y) {
      const t = theme();
      ctx.save();
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillStyle = t.muted;
      ctx.fillText(title.toUpperCase(), 14, y + 10);
      const textW = ctx.measureText(title.toUpperCase()).width;
      ctx.strokeStyle = t.outline;
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      ctx.moveTo(22 + textW, y + 10);
      ctx.lineTo(width - 14, y + 10);
      ctx.stroke();
      ctx.restore();
    },
  };
}

// --- wiring --------------------------------------------------------------------------

function insertBefore(node, widget, beforeName) {
  // addCustomWidget appends; move the drawn-only widget in front of its group.
  const added = node.addCustomWidget(widget);
  const list = node.widgets;
  list.splice(list.indexOf(added), 1);
  const at = beforeName ? list.findIndex((w) => w.name === beforeName) : 0;
  list.splice(at < 0 ? list.length : at, 0, added);
  return added;
}

function relabel(node) {
  for (const w of valueWidgets(node)) {
    let label = BASIC_LABELS[w.name];
    if (!label && w.name.startsWith("hsl_")) {
      label = { hue: "Hue", sat: "Saturation", lum: "Luminance" }[w.name.split("_")[2]];
    } else if (!label && w.name.startsWith("grade_")) {
      label = { hue: "Hue", sat: "Saturation", lum: "Luminance", bal: "Balance" }[w.name.split("_")[2]];
    } else if (!label && w.name.startsWith("gray_")) {
      const c = w.name.slice(5);
      label = c.charAt(0).toUpperCase() + c.slice(1);
    }
    if (label) w.label = label;
  }
}

function setup(node) {
  if (node.__jhColorLabTabs) return;
  node.__jhColorLabTabs = true;
  insertBefore(node, tabBarWidget(), null);
  for (const [before, title] of Object.entries(SECTIONS)) {
    if (node.widgets.some((w) => w.name === before)) {
      insertBefore(node, sectionWidget(title, tabOf(before)), before);
    }
  }
  insertBefore(node, hslPickerWidget(), "hsl_red_hue");
  insertBefore(node, gradeWheelsWidget(), "grade_shadow_hue");
  relabel(node);
  refresh(node);
}

function defaultsFrom(nodeData) {
  const out = {};
  for (const group of ["required", "optional"]) {
    for (const [name, spec] of Object.entries(nodeData.input?.[group] ?? {})) {
      const opts = Array.isArray(spec) ? spec[1] : undefined;
      if (opts && "default" in opts) out[name] = opts.default;
      else if (Array.isArray(spec) && Array.isArray(spec[0])) out[name] = spec[0][0];
    }
  }
  return out;
}

app.registerExtension({
  name: "JH.PixelPro.ColorLabTabs",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const defaults = defaultsFrom(nodeData);
    const vueNodes = () => app.ui?.settings?.getSettingValue?.("Comfy.VueNodes.Enabled") === true;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      this.__jhDefaults = defaults;
      // The drawn-only widgets target the LiteGraph canvas; Vue nodes keep the plain list.
      if (!vueNodes()) setup(this);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const hadState = !!info?.properties?.[STATE_KEY];
      const result = onConfigure?.apply(this, arguments);
      if (this.__jhColorLabTabs) {
        // Workflows saved before the tabs existed open on the first edited tab.
        if (!hadState) openFirstEdited(this);
        requestAnimationFrame(() => refresh(this));
      }
      return result;
    };
  },
});
