import os
import io
import sys
import json
import base64
import uuid
import dataclasses
from dataclasses import dataclass, field, asdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    from fontTools.ttLib import TTFont
except Exception:
    TTFont = None

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                               QFormLayout, QLabel, QLineEdit, QPushButton,
                               QSpinBox, QDoubleSpinBox, QComboBox, QColorDialog,
                               QFileDialog, QMessageBox, QInputDialog)
from PySide6.QtGui import QColor

ROOT = os.path.dirname(os.path.abspath(__file__))
PRESETS_DIR = os.path.join(ROOT, "presets")
os.makedirs(PRESETS_DIR, exist_ok=True)

TEXTURE_W = 1000
TEXTURE_H = 320
UV_UNIT = 16.0
BASELINE_Y = 2
RENDER_PX = 200

GAP = 2
SWATCH_H = 20
FACE_REGION_H = TEXTURE_H - SWATCH_H
SWATCH_SIZE = 8

CAP_A = "\U0001F633"
OPEN_QUOTE = "\U0001F629"
TERM_OPEN = "┫"
TERM_CLOSE = "┣"
SMART = {"‘": "'", "’": "'", "“": '"', "”": '"'}

DIGITS = [str(i) for i in range(10)]
LETTERS = [chr(ord("a") + i) for i in range(26)]
SYMBOLS = list("!#$%&'()*+,-./:;<=>?@[\\]^_{}~£€")

ALL_KEYS = DIGITS + LETTERS + [CAP_A, OPEN_QUOTE, TERM_OPEN, TERM_CLOSE] + SYMBOLS
PROCEDURAL = {TERM_OPEN, TERM_CLOSE}

KEY_SOURCE = {d: d for d in DIGITS}
KEY_SOURCE.update({l: l.upper() for l in LETTERS})
KEY_SOURCE[CAP_A] = "A"
KEY_SOURCE[OPEN_QUOTE] = "‘"
for sym in SYMBOLS:
    KEY_SOURCE[sym] = sym


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


@dataclass
class Settings:
    font_name: str = "Custom"
    font_id: str = "custom-title"
    preview_text: str = "testing text"
    depth: float = 22.0
    outline_thickness: float = 4.0
    blockiness: float = 30.0
    boldness: float = 60.0
    scale: float = 1.0
    letter_spacing: float = 0.0
    highlight_strength: float = 45.0
    gradient_top: list = field(default_factory=lambda: [255, 64, 64])
    gradient_bottom: list = field(default_factory=lambda: [128, 0, 0])
    outline_color: list = field(default_factory=lambda: [20, 0, 0])
    side_color: list = field(default_factory=lambda: [90, 0, 0])

    def clamped(self):
        c = Settings.from_dict(self.to_dict())
        c.depth = clamp(c.depth, 0.0, 50.0)
        c.outline_thickness = clamp(c.outline_thickness, 0.0, 20.0)
        c.blockiness = clamp(c.blockiness, 0.0, 100.0)
        c.boldness = clamp(c.boldness, 0.0, 100.0)
        c.scale = clamp(c.scale, 0.25, 4.0)
        c.letter_spacing = clamp(c.letter_spacing, -20.0, 40.0)
        c.highlight_strength = clamp(c.highlight_strength, 0.0, 100.0)
        for n in ("gradient_top", "gradient_bottom", "outline_color", "side_color"):
            setattr(c, n, rgb(getattr(c, n)))
        return c

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        valid = {f.name for f in dataclasses.fields(cls)}
        kw = {}
        for k, v in (data or {}).items():
            if k in valid:
                kw[k] = list(v) if isinstance(v, (list, tuple)) else v
        return cls(**kw)


def rgb(c):
    c = list(c) if c else [255, 255, 255]
    while len(c) < 3:
        c.append(0)
    return [int(clamp(int(round(x)), 0, 255)) for x in c[:3]]


def safe_name(name):
    keep = "-_. abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ("".join(ch for ch in name if ch in keep).strip()) or "preset"


def list_presets():
    return [os.path.splitext(f)[0] for f in sorted(os.listdir(PRESETS_DIR)) if f.lower().endswith(".json")]


def save_preset(name, s):
    path = os.path.join(PRESETS_DIR, safe_name(name) + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(s.to_dict(), fh, indent=2)
    return os.path.splitext(os.path.basename(path))[0]


def load_preset(name):
    with open(os.path.join(PRESETS_DIR, safe_name(name) + ".json"), encoding="utf-8") as fh:
        return Settings.from_dict(json.load(fh)).clamped()


def delete_preset(name):
    path = os.path.join(PRESETS_DIR, safe_name(name) + ".json")
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False


class Font:
    def __init__(self, path):
        self.path = path
        self.font = ImageFont.truetype(path, RENDER_PX)
        self.ascent, self.descent = self.font.getmetrics()
        self.family = self._family()
        self.cmap = self._cmap()
        self.cap = self._cap_height()

    def _family(self):
        if TTFont:
            try:
                with TTFont(self.path, fontNumber=0, lazy=True) as tt:
                    for nid in (4, 1):
                        rec = tt["name"].getDebugName(nid)
                        if rec:
                            return rec.strip()
            except Exception:
                pass
        return os.path.splitext(os.path.basename(self.path))[0]

    def _cmap(self):
        if not TTFont:
            return None
        try:
            with TTFont(self.path, fontNumber=0, lazy=True) as tt:
                return tt.getBestCmap()
        except Exception:
            return None

    def has(self, ch):
        if self.cmap is None:
            return True
        return bool(ch) and ord(ch[0]) in self.cmap

    def render(self, ch):
        if not ch:
            return None
        m = RENDER_PX
        img = Image.new("L", (RENDER_PX * 3 + m * 2, self.ascent + self.descent + m * 2), 0)
        d = ImageDraw.Draw(img)
        by = m + self.ascent
        try:
            d.text((m, by), ch, fill=255, font=self.font, anchor="ls")
        except Exception:
            d.text((m, m), ch, fill=255, font=self.font)
            by = m + self.ascent
        arr = np.asarray(img, np.float32) / 255.0
        ys, xs = np.nonzero(arr > 0.2)
        if len(xs) == 0:
            return None
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        return arr[y0:y1, x0:x1], float(by - y0)

    def _cap_height(self):
        best = 0.0
        for ch in "HEXAMBND":
            r = self.render(ch)
            if r:
                best = max(best, r[1])
        return best if best > 1 else self.ascent * 0.72


def downscale(arr, w, h):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    im = im.resize((max(1, w), max(1, h)), Image.BOX)
    return np.asarray(im, np.float32) / 255.0


def dilate(mask, r):
    if r <= 0:
        return mask.copy()
    im = Image.fromarray((mask.astype(np.uint8) * 255)).filter(ImageFilter.MaxFilter(r * 2 + 1))
    return np.asarray(im) > 127


def greedy_rects(mask):
    m = np.ascontiguousarray(mask, dtype=bool)
    h, w = m.shape
    covered = np.zeros((h, w), dtype=bool)
    rects = []
    for y in range(h):
        x = 0
        while x < w:
            if not m[y, x] or covered[y, x]:
                x += 1
                continue
            rw = 1
            while x + rw < w and m[y, x + rw] and not covered[y, x + rw]:
                rw += 1
            rh = 1
            while y + rh < h and m[y + rh, x:x + rw].all() and not covered[y + rh, x:x + rw].any():
                rh += 1
            covered[y:y + rh, x:x + rw] = True
            rects.append((x, y, rw, rh))
            x += rw
    return rects


def terminator_mask(key, grid_h):
    gh = max(6, grid_h)
    gw = max(2, round(grid_h * 0.18))
    mask = np.ones((gh, gw), dtype=bool)
    notch = max(1, gh // 4)
    if key == TERM_OPEN:
        mask[notch:gh - notch, 0:max(1, gw // 2)] = False
    else:
        mask[notch:gh - notch, gw - max(1, gw // 2):] = False
    return mask, float(gh)


def box_glyph(grid_h):
    gh = max(6, grid_h)
    gw = max(3, round(grid_h * 0.55))
    mask = np.ones((gh, gw), dtype=bool)
    if gh > 4 and gw > 4:
        mask[2:gh - 2, 2:gw - 2] = False
    return mask, float(gh)


def glyph_mask(font, key, grid_h, ppc, thr):
    if key in PROCEDURAL:
        m, b = terminator_mask(key, grid_h)
        return m, b, True

    src = KEY_SOURCE.get(key, key)
    r = font.render(src)
    if r is None and len(src) == 1 and src.isalpha():
        r = font.render(src.lower())
    if r is None and src in SMART:
        r = font.render(SMART[src])
    if r is None:
        m, b = box_glyph(grid_h)
        return m, b, False

    cov, baseline = r
    cw = max(1, round(cov.shape[1] / ppc))
    ch = max(1, round(cov.shape[0] / ppc))
    cc = downscale(cov, cw, ch)
    mask = cc >= thr
    if not mask.any():
        mask = cc >= max(0.18, float(cc.max()) * 0.5)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        m, b = box_glyph(grid_h)
        return m, b, False
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    return mask[y0:y1, x0:x1], baseline / ppc - y0, font.has(src)


def render_face(mask, s, tpc):
    gh, gw = mask.shape
    W, H = max(1, gw * tpc), max(1, gh * tpc)
    big = np.asarray(Image.fromarray((mask.astype(np.uint8) * 255)).resize((W, H), Image.NEAREST)) > 127
    top = np.array(rgb(s.gradient_top), np.float32)
    bot = np.array(rgb(s.gradient_bottom), np.float32)
    ramp = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    rows = top[None, :] * (1.0 - ramp) + bot[None, :] * ramp
    grad = np.repeat(rows[:, None, :], W, axis=1)
    out = np.zeros((H, W, 4), np.uint8)
    out[:, :, :3] = np.clip(grad, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.where(big, 255, 0).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def shelf_pack(sizes):
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][1])
    pos = [(0, 0)] * len(sizes)
    x = y = shelf = 0
    for i in order:
        w, h = sizes[i]
        if w > TEXTURE_W:
            return None
        if x + w > TEXTURE_W:
            y += shelf + GAP
            x = 0
            shelf = 0
        if y + h > FACE_REGION_H:
            return None
        pos[i] = (x, y)
        x += w + GAP
        shelf = max(shelf, h)
    return pos


def plan_tpc(cells):
    for tpc in (3, 2, 1):
        if shelf_pack([(w * tpc, h * tpc) for (w, h) in cells]) is not None:
            return tpc
    return 1


def u_of(px):
    return round(px * UV_UNIT / TEXTURE_W, 5)


def v_of(px):
    return round(px * UV_UNIT / TEXTURE_H, 5)


def lighten(c, f):
    return [min(255, int(round(x + (255 - x) * f))) for x in c]


def darken(c, f):
    return [max(0, int(round(x * (1 - f)))) for x in c]


def build_atlas(faces, tpc, s):
    sizes = [im.size for (_k, im) in faces]
    pos = shelf_pack(sizes)
    if pos is None:
        return None
    img = Image.new("RGBA", (TEXTURE_W, TEXTURE_H), (0, 0, 0, 0))
    placements = {}
    for (key, face), (x, y) in zip(faces, pos):
        img.paste(face, (x, y), face)
        placements[key] = (x, y)

    border = FACE_REGION_H
    outline = rgb(s.outline_color)
    side = rgb(s.side_color)
    top = lighten(rgb(s.gradient_top), 0.35 + 0.45 * s.highlight_strength / 100.0)
    bottom = darken(rgb(s.gradient_bottom), 0.45)
    img.paste(Image.new("RGBA", (TEXTURE_W, SWATCH_H), tuple(outline) + (255,)), (0, border))
    sw = {}
    for i, (name, color) in enumerate([("outline", outline), ("side", side), ("top", top), ("bottom", bottom)]):
        sx = 2 + i * (SWATCH_SIZE + GAP)
        sy = border + 2
        img.paste(Image.new("RGBA", (SWATCH_SIZE, SWATCH_SIZE), tuple(color) + (255,)), (sx, sy))
        cx, cy = sx + 3, sy + 3
        sw[name] = [u_of(cx), v_of(cy), u_of(cx + 2), v_of(cy + 2)]
    return img, placements, tpc, sw


def outline_ring(mask, oc):
    if oc <= 0:
        return []
    p = oc
    padded = np.zeros((mask.shape[0] + 2 * p, mask.shape[1] + 2 * p), dtype=bool)
    padded[p:p + mask.shape[0], p:p + mask.shape[1]] = mask
    ring = dilate(padded, p) & ~padded
    return [(cx - p, cy - p, rw, rh) for (cx, cy, rw, rh) in greedy_rects(ring)]


def r4(x):
    return round(float(x), 4)


def build_cubes(mask, baseline_cell, placement, tpc, sw, cell_units, depth, oc):
    ax, ay = placement
    z0, z1 = 0.0, float(depth)
    cubes = []
    min_x = float("inf")
    max_x = float("-inf")

    def yedge(cy, rh):
        return (BASELINE_Y + (baseline_cell - (cy + rh)) * cell_units,
                BASELINE_Y + (baseline_cell - cy) * cell_units)

    for (cx, cy, rw, rh) in outline_ring(mask, oc):
        x0, x1 = cx * cell_units, (cx + rw) * cell_units
        yb, yt = yedge(cy, rh)
        min_x, max_x = min(min_x, x0), max(max_x, x1)
        o = list(sw["outline"])
        cubes.append({"from": [r4(x0), r4(yb), r4(z0)], "to": [r4(x1), r4(yt), r4(z1)],
                      "faces": {d: list(o) for d in ("north", "south", "east", "west", "up", "down")}})

    for (cx, cy, rw, rh) in greedy_rects(mask):
        x0, x1 = cx * cell_units, (cx + rw) * cell_units
        yb, yt = yedge(cy, rh)
        min_x, max_x = min(min_x, x0), max(max_x, x1)
        north = [u_of(ax + cx * tpc), v_of(ay + cy * tpc), u_of(ax + (cx + rw) * tpc), v_of(ay + (cy + rh) * tpc)]
        cubes.append({"from": [r4(x0), r4(yb), r4(z0)], "to": [r4(x1), r4(yt), r4(z1)],
                      "faces": {"north": north, "south": list(sw["side"]), "east": list(sw["side"]),
                                "west": list(sw["side"]), "up": list(sw["top"]), "down": list(sw["bottom"])}})

    if cubes and min_x != float("inf"):
        c2 = min_x + max_x
        for c in cubes:
            fx, tx = c["from"][0], c["to"][0]
            c["from"][0] = r4(c2 - tx)
            c["to"][0] = r4(c2 - fx)
    if min_x == float("inf"):
        min_x = max_x = 0.0
    return cubes, r4(min_x), r4(max_x)


def slug(raw):
    out = []
    for ch in (raw or "custom-title").lower().strip():
        if ch.isalnum():
            out.append(ch)
        elif ch in " _-":
            out.append("-")
    cleaned = "".join(out).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned or "custom-title"


def build_pack(path, s, prog=None):
    s = s.clamped()
    font = Font(path)

    grid_h = max(8, int(round(36 + (10 - 36) * s.blockiness / 100.0)))
    cap = max(4.0, font.cap)
    ppc = cap / grid_h
    cell_units = (44.0 * s.scale) / grid_h
    depth = float(s.depth)
    oc = max(0, min(8, int(round(s.outline_thickness / cell_units)))) if s.outline_thickness > 0 else 0
    thr = 0.7 - 0.4 * (s.boldness / 100.0)

    if prog:
        prog("Rasterising glyphs...")
    raw = {k: glyph_mask(font, k, grid_h, ppc, thr) for k in ALL_KEYS}
    sizes = [(m.shape[1], m.shape[0]) for (m, _b, _h) in raw.values()]
    tpc = plan_tpc(sizes)

    faces = [(k, render_face(raw[k][0], s, tpc)) for k in ALL_KEYS]
    if prog:
        prog("Packing texture...")
    atlas = build_atlas(faces, tpc, s)
    while atlas is None and tpc > 1:
        tpc -= 1
        faces = [(k, render_face(raw[k][0], s, tpc)) for k in ALL_KEYS]
        atlas = build_atlas(faces, tpc, s)
    if atlas is None:
        raise RuntimeError("Glyphs do not fit the 1000x320 texture.")
    img, placements, tpc, sw = atlas

    if prog:
        prog("Building 3D models...")
    glyphs = {}
    for k in ALL_KEYS:
        mask, baseline, _has = raw[k]
        cubes, minx, maxx = build_cubes(mask, baseline, placements[k], tpc, sw, cell_units, depth, oc)
        glyphs[k] = (cubes, round(maxx - minx, 4))

    meta = {
        "id": slug(s.font_id), "name": s.font_name or "Custom Title",
        "spaceWidth": max(4, int(round(0.22 * 44.0 * s.scale))),
        "characterSpacing": int(max(0, min(20, round(s.letter_spacing)))),
        "preview_text": s.preview_text,
    }
    return {"meta": meta, "atlas": img, "glyphs": glyphs}


def key_for(ch, glyphs):
    if ch == " ":
        return None
    if ch.lower() in glyphs:
        return ch.lower()
    if ch in glyphs:
        return ch
    return None


def fname(s):
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    return ("".join(c if c in keep else ("_" if c == " " else "") for c in s)) or "title"


def group(name, children, is_open=False):
    return {"name": name, "origin": [0, 0, 0], "rotation": [0, 0, 0], "color": 0,
            "uuid": str(uuid.uuid4()), "export": True, "mirror_uv": False,
            "isOpen": is_open, "locked": False, "visibility": True, "autouv": 0, "children": children}


def build_bbmodel(pack, text):
    meta = pack["meta"]
    glyphs = pack["glyphs"]
    text = text if text.strip() else "CURSE SMP"
    cs = meta["characterSpacing"]

    cursor = 0.0
    placed = []
    for ch in text:
        key = key_for(ch, glyphs)
        if key is None:
            cursor += meta["spaceWidth"] + cs
            continue
        cubes, width = glyphs[key]
        if not cubes:
            cursor += meta["spaceWidth"] + cs
            continue
        maxx = max(v for c in cubes for v in (c["from"][0], c["to"][0]))
        shift = -(cursor + maxx)
        moved = []
        for c in cubes:
            cc = json.loads(json.dumps(c))
            cc["from"][0] += shift
            cc["to"][0] += shift
            moved.append(cc)
        placed.append((key, moved))
        cursor += width + cs + 2

    allc = [c for (_k, cubes) in placed for c in cubes]
    if allc:
        xs = [v for c in allc for v in (c["from"][0], c["to"][0])]
        mid = (min(xs) + max(xs)) / 2.0
        for c in allc:
            c["from"][0] -= mid
            c["to"][0] -= mid

    tex_uuid = str(uuid.uuid4())
    elements = []
    char_groups = []
    for gi, (key, cubes) in enumerate(placed):
        ids = []
        for ci, c in enumerate(cubes):
            cu = str(uuid.uuid4())
            fc = {}
            for d, uv in c["faces"].items():
                fc[d] = {"uv": [round(uv[0] / UV_UNIT * TEXTURE_W, 4), round(uv[1] / UV_UNIT * TEXTURE_H, 4),
                                round(uv[2] / UV_UNIT * TEXTURE_W, 4), round(uv[3] / UV_UNIT * TEXTURE_H, 4)],
                         "texture": tex_uuid}
            elements.append({"name": f"{key}_{gi}_{ci}", "box_uv": False, "type": "cube", "uuid": cu,
                             "from": c["from"], "to": c["to"], "color": 0, "origin": [0, 0, 0], "faces": fc})
            ids.append(cu)
        char_groups.append(group(key, ids))
    outliner = [group(text.strip() or "Title", char_groups, True)] if elements else []

    buf = io.BytesIO()
    pack["atlas"].save(buf, format="PNG")
    src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "meta": {"format_version": "4.5", "model_format": "minecraft_title",
                 "box_uv": False, "creation_tool": "blockbenchfontgenerator"},
        "name": f"{meta['name']} - {text}", "model_identifier": "",
        "resolution": {"width": TEXTURE_W, "height": TEXTURE_H},
        "elements": elements, "outliner": outliner,
        "textures": [{"path": "", "name": "flat.png", "folder": "", "namespace": "", "id": "0",
                      "width": TEXTURE_W, "height": TEXTURE_H, "uv_width": TEXTURE_W, "uv_height": TEXTURE_H,
                      "particle": False, "render_mode": "default", "visible": True, "internal": True,
                      "saved": False, "mode": "bitmap", "uuid": tex_uuid,
                      "relative_path": f"./fonts/{meta['id']}/textures/flat.png", "source": src}],
    }


def write_pack(pack, out_root, prog=None):
    meta = pack["meta"]
    if prog:
        prog("Writing .bbmodel...")
    bb = build_bbmodel(pack, meta["preview_text"])
    name = f"{meta['id']}__{fname(meta['preview_text'].strip())}.bbmodel"
    path = os.path.join(out_root, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(bb, fh, ensure_ascii=False)
    return {"bbmodel": path, "glyph_count": len(pack["glyphs"])}


NUMS = [
    ("depth", "Depth", 0, 50, 1, 0),
    ("outline_thickness", "Outline thickness", 0, 20, 1, 0),
    ("blockiness", "Pixelation", 0, 100, 1, 0),
    ("boldness", "Corner sharpness", 0, 100, 1, 0),
    ("scale", "Scale", 0.25, 4.0, 0.05, 2),
    ("letter_spacing", "Letter spacing", -20, 40, 1, 0),
    ("highlight_strength", "Highlight", 0, 100, 1, 0),
]
COLS = [("gradient_top", "Gradient top"), ("gradient_bottom", "Gradient bottom"),
        ("outline_color", "Outline"), ("side_color", "Side / depth")]
NO_PRESETS = "(no saved presets)"


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("blockbenchfontgenerator")
        self.setAcceptDrops(True)
        self.font_path = None
        d = Settings()
        self.colors = {k: list(getattr(d, k)) for (k, _l) in COLS}

        root = QVBoxLayout(self)

        fr = QHBoxLayout()
        self.path_lbl = QLabel("(no font loaded)")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self.browse)
        fr.addWidget(self.path_lbl, 1)
        fr.addWidget(browse)
        root.addLayout(fr)

        form = QFormLayout()
        self.name = QLineEdit(d.font_name)
        self.fid = QLineEdit(d.font_id)
        self.text = QLineEdit(d.preview_text)
        form.addRow("Name", self.name)
        form.addRow("Font id", self.fid)
        form.addRow("Title text", self.text)

        self.spins = {}
        for key, label, lo, hi, step, dec in NUMS:
            sp = QDoubleSpinBox() if dec else QSpinBox()
            sp.setRange(lo, hi)
            if dec:
                sp.setDecimals(dec)
                sp.setSingleStep(step)
            v = getattr(d, key)
            sp.setValue(float(v) if dec else int(round(v)))
            self.spins[key] = sp
            form.addRow(label, sp)

        self.colbtns = {}
        for key, label in COLS:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, k=key: self.pick(k))
            self._paint_btn(btn, self.colors[key])
            self.colbtns[key] = btn
            form.addRow("Color", btn)
        root.addLayout(form)

        pr = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.currentTextChanged.connect(self.choose_preset)
        save = QPushButton("Save")
        save.clicked.connect(self.save)
        dele = QPushButton("Delete")
        dele.clicked.connect(self.delete)
        pr.addWidget(QLabel("Preset"))
        pr.addWidget(self.combo, 1)
        pr.addWidget(save)
        pr.addWidget(dele)
        root.addLayout(pr)
        self.reload_presets()

        self.export_btn = QPushButton("Export Blockbench font pack")
        self.export_btn.clicked.connect(self.export)
        self.export_btn.setEnabled(False)
        root.addWidget(self.export_btn)

        self.status = QLabel("Load a .ttf or .otf font to begin.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

    def _paint_btn(self, btn, c):
        r, g, b = c
        fg = "#000" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "#fff"
        btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {fg};")

    def collect(self):
        data = {"font_name": self.name.text(), "font_id": self.fid.text(),
                "author": "breadbb", "preview_text": self.text.text()}
        for key in self.spins:
            data[key] = self.spins[key].value()
        for key in self.colors:
            data[key] = list(self.colors[key])
        return Settings.from_dict(data)

    def apply(self, s):
        cur = self.collect()
        self.name.setText(cur.font_name)
        self.fid.setText(cur.font_id)
        self.text.setText(cur.preview_text)
        for key, sp in self.spins.items():
            v = getattr(s, key)
            sp.setValue(float(v) if isinstance(sp, QDoubleSpinBox) else int(round(v)))
        for key in self.colors:
            self.colors[key] = list(getattr(s, key))
            self._paint_btn(self.colbtns[key], self.colors[key])

    def pick(self, key):
        c = QColorDialog.getColor(QColor(*self.colors[key]), self, "Pick color")
        if c.isValid():
            self.colors[key] = [c.red(), c.green(), c.blue()]
            self._paint_btn(self.colbtns[key], self.colors[key])

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select a font", "", "Fonts (*.ttf *.otf *.TTF *.OTF)")
        if path:
            self.load_font(path)

    def load_font(self, path):
        try:
            font = Font(path)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid font", str(exc))
            return
        self.font_path = path
        self.path_lbl.setText(font.family)
        if self.name.text() in ("", "Custom"):
            self.name.setText(font.family)
        if self.fid.text() in ("", "custom-title"):
            self.fid.setText(slug(font.family))
        self.export_btn.setEnabled(True)
        self.status.setText(f"Loaded: {font.family}")

    def reload_presets(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        names = list_presets()
        self.combo.addItems(names if names else [NO_PRESETS])
        self.combo.blockSignals(False)

    def choose_preset(self, name):
        if not name or name == NO_PRESETS:
            return
        try:
            self.apply(load_preset(name))
        except Exception:
            pass

    def save(self):
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if ok and name.strip():
            stem = save_preset(name.strip(), self.collect())
            self.reload_presets()
            i = self.combo.findText(stem)
            if i >= 0:
                self.combo.blockSignals(True)
                self.combo.setCurrentIndex(i)
                self.combo.blockSignals(False)
            self.status.setText(f"Saved preset '{stem}'.")

    def delete(self):
        name = self.combo.currentText()
        if name and name != NO_PRESETS and delete_preset(name):
            self.reload_presets()
            self.status.setText("Preset deleted.")

    def export(self):
        if not self.font_path:
            return
        out = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not out:
            return
        self.export_btn.setEnabled(False)

        def prog(msg):
            self.status.setText(msg)
            QApplication.processEvents()

        try:
            pack = build_pack(self.font_path, self.collect(), prog)
            info = write_pack(pack, out, prog)
            self.status.setText(f"Exported .bbmodel: {info['bbmodel']}")
            QMessageBox.information(self, "Done", f"Exported .bbmodel:\n{info['bbmodel']}")
        except Exception as exc:
            self.status.setText(f"Failed: {exc}")
            QMessageBox.critical(self, "Export failed", str(exc))
        self.export_btn.setEnabled(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.toLocalFile().lower().endswith((".ttf", ".otf"))
                                          for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith((".ttf", ".otf")):
                self.load_font(p)
                return


def main():
    app = QApplication(sys.argv)
    w = App()
    w.resize(380, 660)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())