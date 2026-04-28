import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

DATA_DIR = Path(__file__).parent / "data"
LAND_USE_DIR = DATA_DIR / "land use"
FIRE_DIR = DATA_DIR / "Fire"
VEGETATION_DIR = DATA_DIR / "Vegetation"
WATER_QUALITY_DIR = DATA_DIR / "water quality"
OVERLAY_ALPHA = 0.65

SLIDER_CSS = """
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: transparent; overflow: hidden; }
  #comp {
    position: relative; width: 100%; display: block;
    cursor: col-resize; user-select: none; -webkit-user-select: none;
  }
  #base-img { display: block; width: 100%; height: auto; }
  .ov { position: absolute; top: 0; left: 0; width: 100%; pointer-events: none; }
  .ov img { display: block; width: 100%; height: auto; }
  .handle {
    position: absolute; top: 0; height: 100%; width: 48px;
    transform: translateX(-50%);
    display: flex; align-items: center; justify-content: center;
    cursor: col-resize; z-index: 20;
  }
  .bar {
    position: absolute; left: 50%; top: 0;
    transform: translateX(-50%);
    width: 3px; height: 100%;
    background: #fff; box-shadow: 0 0 5px rgba(0,0,0,0.7);
  }
  .circle {
    position: relative; z-index: 1;
    width: 38px; height: 38px; background: #fff;
    border-radius: 50%; box-shadow: 0 0 6px rgba(0,0,0,0.55);
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; color: #444; font-family: sans-serif;
  }
  .lbl {
    position: absolute; top: 10px;
    background: rgba(0,0,0,0.55); color: #fff;
    padding: 4px 10px; border-radius: 4px;
    font-size: 13px; font-family: sans-serif;
    pointer-events: none; z-index: 10; white-space: nowrap;
  }
"""

RESIZE_JS = """
  function sendHeight() {
    const h = Math.max(
      document.documentElement.scrollHeight,
      document.body.scrollHeight,
      comp.getBoundingClientRect().height
    );
    if (h > 0) window.parent.postMessage(
      {isStreamlitMessage:true, type:'streamlit:setFrameHeight', height: Math.ceil(h)+4}, '*');
  }
  window.addEventListener('load', sendHeight);
  window.addEventListener('resize', sendHeight);
  const ro = new ResizeObserver(sendHeight);
  ro.observe(document.body);
  ro.observe(comp);
  document.querySelectorAll('img').forEach(img => {
    if (img.complete) return;
    img.addEventListener('load', sendHeight, {once:true});
  });
  setTimeout(sendHeight, 0);
  setTimeout(sendHeight, 100);
"""


def white_to_green(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    is_white = (arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 200)
    arr[is_white] = [0, 200, 0]
    return Image.fromarray(arr, "RGB")


def make_overlay(base: Image.Image, mask: Image.Image) -> Image.Image:
    base_rgb = base.convert("RGB")
    mask_rgba = mask.convert("RGBA")
    if mask_rgba.size != base_rgb.size:
        mask_rgba = mask_rgba.resize(base_rgb.size, Image.LANCZOS)
    arr = np.array(mask_rgba, dtype=np.uint8)
    arr[:, :, 3] = 255
    return Image.fromarray(arr, "RGBA")


def resize_to(img: Image.Image, max_width: int = 1100) -> Image.Image:
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    return img


def split_into_tiles(img: Image.Image, rows: int = 2, cols: int = 2) -> list[Image.Image]:
    tiles: list[Image.Image] = []
    for row in range(rows):
        top = (row * img.height) // rows
        bottom = img.height if row == rows - 1 else ((row + 1) * img.height) // rows
        for col in range(cols):
            left = (col * img.width) // cols
            right = img.width if col == cols - 1 else ((col + 1) * img.width) // cols
            tiles.append(img.crop((left, top, right, bottom)))
    return tiles


def fit_to_box(img: Image.Image, max_width: int, max_height: int) -> Image.Image:
    ratio = min(max_width / img.width, max_height / img.height)
    if ratio < 1:
        new_size = (max(1, int(img.width * ratio)), max(1, int(img.height * ratio)))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def to_b64(img: Image.Image, format: str = "JPEG", quality: int = 82) -> str:
    buf = BytesIO()
    if format.upper() == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode()


def add_legend_overlay(
    base_img: Image.Image,
    legend_img: Image.Image,
    padding: int = 12,
    legend_resize_scale: float = 1.0,
    max_width_ratio: float = 0.6,
    max_height_ratio: float = 0.72,
) -> Image.Image:
    result = base_img.convert("RGBA").copy()
    max_width = max(480, int(result.width * max_width_ratio))
    max_height = max(320, int(result.height * max_height_ratio))
    legend = legend_img.convert("RGB")
    if legend_resize_scale != 1.0:
        new_size = (
            max(1, int(legend.width * legend_resize_scale)),
            max(1, int(legend.height * legend_resize_scale)),
        )
        legend = legend.resize(new_size, Image.LANCZOS)
    legend = fit_to_box(legend, max_width=max_width, max_height=max_height)
    x = max(padding, result.width - legend.width - padding)
    y = max(padding, result.height - legend.height - padding)
    result.paste(legend.convert("RGBA"), (x, y))
    return result


def apply_opacity(img: Image.Image, opacity: float) -> Image.Image:
    opacity = max(0.0, min(1.0, float(opacity)))
    rgba = img.convert("RGBA")
    arr = np.array(rgba, dtype=np.uint8)
    arr[:, :, 3] = int(round(255 * opacity))
    return Image.fromarray(arr, "RGBA")


def single_slider_html(
    img_base: Image.Image,
    img_overlay: Image.Image,
    label_left: str = "RGB",
    label_right: str = "Mask",
    legend_img: Image.Image | None = None,
    mask_opacity: float = OVERLAY_ALPHA,
    legend_scale: float = 0.6,
    legend_resize_scale: float = 1.0,
) -> str:
    if legend_img is not None:
        img_base = add_legend_overlay(
            img_base,
            legend_img,
            max_width_ratio=legend_scale,
            max_height_ratio=min(0.9, legend_scale + 0.12),
            legend_resize_scale=legend_resize_scale,
        )
        img_overlay = apply_opacity(img_overlay, mask_opacity)
        img_overlay = add_legend_overlay(
            img_overlay,
            legend_img,
            max_width_ratio=legend_scale,
            max_height_ratio=min(0.9, legend_scale + 0.12),
            legend_resize_scale=legend_resize_scale,
        )
    else:
        img_overlay = apply_opacity(img_overlay, mask_opacity)
    b0 = to_b64(img_base, format="JPEG")
    b1 = to_b64(img_overlay, format="PNG")
    legend_block = ""
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
{SLIDER_CSS}
#ov1 {{ clip-path: inset(0 calc(100% - var(--p1)) 0 0); }}
#lbl-l {{ left: 12px; }}
#lbl-r {{ right: 12px; }}
.legend {{
  position: absolute; right: 12px; bottom: 48px;
  max-width: 28%; max-height: 42%;
  z-index: 15; pointer-events: none;
}}
.legend img {{
  display: block; width: 100%; height: auto;
}}
</style></head><body>
<div id="comp" style="--p1:50%; --mask-opacity:{mask_opacity}">
  <img id="base-img" src="data:image/jpeg;base64,{b0}">
  <div class="ov" id="ov1"><img src="data:image/png;base64,{b1}"></div>
  <div class="handle" id="h1" style="left:50%">
    <div class="bar"></div><div class="circle">⇔</div>
  </div>
  <div class="lbl" id="lbl-l">{label_left}</div>
  <div class="lbl" id="lbl-r">{label_right}</div>
{legend_block}</div>
<script>
(function() {{
  const comp = document.getElementById('comp');
  const h1 = document.getElementById('h1');
  let p1 = 50, dragging = false;
  function apply() {{
    comp.style.setProperty('--p1', p1 + '%');
    h1.style.left = p1 + '%';
  }}
  function pct(e) {{
    const r = comp.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
    return Math.max(0, Math.min(100, x / r.width * 100));
  }}
  function onMove(e) {{ if (!dragging) return; p1 = pct(e); apply(); }}
  h1.addEventListener('mousedown', e => {{ dragging = true; e.preventDefault(); }});
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', () => dragging = false);
  h1.addEventListener('touchstart', e => {{ dragging = true; e.preventDefault(); }}, {{passive:false}});
  document.addEventListener('touchmove', onMove, {{passive:false}});
  document.addEventListener('touchend', () => dragging = false);
  {RESIZE_JS}
  apply();
}})();
</script></body></html>"""


def dual_slider_html(
    img_base: Image.Image,
    img_ov1: Image.Image,
    img_ov2: Image.Image,
    label1: str = "Mask 1",
    label2: str = "Mask 2",
    mask_opacity: float = OVERLAY_ALPHA,
) -> str:
    b0 = to_b64(img_base, format="JPEG")
    b1 = to_b64(apply_opacity(img_ov1, mask_opacity), format="PNG")
    b2 = to_b64(apply_opacity(img_ov2, mask_opacity), format="PNG")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
{SLIDER_CSS}
#ov1 {{ clip-path: inset(0 calc(100% - var(--p1)) 0 0); }}
#ov2 {{ clip-path: inset(0 0 0 var(--p2)); }}
#lbl1 {{ left: 12px; }}
#lblc {{ left: 50%; transform: translateX(-50%); }}
#lbl2 {{ right: 12px; }}
</style></head><body>
<div id="comp" style="--p1:33%;--p2:67%; --mask-opacity:{mask_opacity}">
  <img id="base-img" src="data:image/jpeg;base64,{b0}">
  <div class="ov" id="ov1"><img src="data:image/png;base64,{b1}"></div>
  <div class="ov" id="ov2"><img src="data:image/png;base64,{b2}"></div>
  <div class="handle" id="h1" style="left:33%">
    <div class="bar"></div><div class="circle">⇔</div>
  </div>
  <div class="handle" id="h2" style="left:67%">
    <div class="bar"></div><div class="circle">⇔</div>
  </div>
  <div class="lbl" id="lbl1">{label1}</div>
  <div class="lbl" id="lblc">RGB</div>
  <div class="lbl" id="lbl2">{label2}</div>
</div>
<script>
(function() {{
  const comp = document.getElementById('comp');
  const h1 = document.getElementById('h1');
  const h2 = document.getElementById('h2');
  const lblc = document.getElementById('lblc');
  let p1 = 33, p2 = 67, dragging = 0;
  function apply() {{
    comp.style.setProperty('--p1', p1 + '%');
    comp.style.setProperty('--p2', p2 + '%');
    h1.style.left = p1 + '%';
    h2.style.left = p2 + '%';
    lblc.style.left = ((p1 + p2) / 2) + '%';
  }}
  function pct(e) {{
    const r = comp.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
    return Math.max(0, Math.min(100, x / r.width * 100));
  }}
  function onMove(e) {{
    if (!dragging) return;
    const v = pct(e), GAP = 5;
    if (dragging === 1) p1 = Math.min(v, p2 - GAP);
    else                p2 = Math.max(v, p1 + GAP);
    apply();
  }}
  h1.addEventListener('mousedown', e => {{ dragging = 1; e.preventDefault(); }});
  h2.addEventListener('mousedown', e => {{ dragging = 2; e.preventDefault(); }});
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', () => dragging = 0);
  h1.addEventListener('touchstart', e => {{ dragging = 1; e.preventDefault(); }}, {{passive:false}});
  h2.addEventListener('touchstart', e => {{ dragging = 2; e.preventDefault(); }}, {{passive:false}});
  document.addEventListener('touchmove', onMove, {{passive:false}});
  document.addEventListener('touchend', () => dragging = 0);
  {RESIZE_JS}
  apply();
}})();
</script></body></html>"""


@st.cache_data(show_spinner=False)
def load_project_images(folder: str):
    d = DATA_DIR / folder
    img0 = resize_to(Image.open(d / "image_0.png").convert("RGB"))
    img1 = resize_to(Image.open(d / "image_1.png").convert("RGB"))
    img2 = resize_to(Image.open(d / "image_2.png").convert("RGB"))
    ov1 = make_overlay(img0, white_to_green(img1))
    ov2 = make_overlay(img0, white_to_green(img2))
    return img0, ov1, ov2


@st.cache_data(show_spinner=False)
def load_fire_images():
    base = resize_to(Image.open(FIRE_DIR / "rgb.png").convert("RGB"))
    overlay = resize_to(Image.open(FIRE_DIR / "swir.png").convert("RGB"))
    if overlay.size != base.size:
        overlay = overlay.resize(base.size, Image.LANCZOS)
    return base, overlay


@st.cache_data(show_spinner=False)
def load_vegetation_images():
    base = resize_to(Image.open(VEGETATION_DIR / "input_rgb.png").convert("RGB"))
    ndvi = resize_to(Image.open(VEGETATION_DIR / "ndvi.png").convert("RGB"))
    moisture = resize_to(Image.open(VEGETATION_DIR / "moisture_index.png").convert("RGB"))
    if ndvi.size != base.size:
        ndvi = ndvi.resize(base.size, Image.LANCZOS)
    if moisture.size != base.size:
        moisture = moisture.resize(base.size, Image.LANCZOS)
    ndvi_overlay = make_overlay(base, white_to_green(ndvi))
    moisture_overlay = make_overlay(base, white_to_green(moisture))
    return base, ndvi_overlay, moisture_overlay


@st.cache_data(show_spinner=False)
def load_land_use_tile_example(tile_index: int):
    tiles_dir = LAND_USE_DIR / "tiles"
    legend_path = LAND_USE_DIR / "land_cover_legend.png"
    base_path = tiles_dir / f"casablanca_01_tile_{tile_index}.png"
    mask_path = tiles_dir / f"casablanca_01_lulc_colored_sliding_tile_{tile_index}.png"

    if not base_path.exists() or not mask_path.exists() or not legend_path.exists():
        return None

    with Image.open(base_path) as base_img:
        base = base_img.convert("RGB")
    with Image.open(mask_path) as mask_img:
        mask = mask_img.convert("RGB")
    with Image.open(legend_path) as legend_img:
        legend = legend_img.convert("RGB")

    if mask.size != base.size:
        mask = mask.resize(base.size, Image.LANCZOS)

    overlay = mask.convert("RGBA")
    arr = np.array(overlay, dtype=np.uint8)
    arr[:, :, 3] = 255
    overlay = Image.fromarray(arr, "RGBA")
    return base, overlay, legend


WATER_QUALITY_SPECS = {
    "chl": {
        "label": "CHL",
        "title": "Chlorophyll-a",
        "path": "MDN_chl_heatmap_jet.png",
        "unit": "mg/m3",
    },
    "tss": {
        "label": "TSS",
        "title": "Total Suspended Solids",
        "path": "MDN_tss_heatmap_jet.png",
        "unit": "g/m3",
    },
    "cdom": {
        "label": "CDOM",
        "title": "Colored Dissolved Organic Matter",
        "path": "MDN_cdom_heatmap_jet.png",
        "unit": "m-1",
    },
}


@st.cache_data(show_spinner=False)
def load_water_quality_statistics() -> dict[str, dict[str, float | str]]:
    stats_path = WATER_QUALITY_DIR / "stream statistics.txt"
    rows: dict[str, dict[str, float | str]] = {}
    if not stats_path.exists():
        return rows

    import re

    pattern = re.compile(
        r"^(CHL|TSS|CDOM)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+(.*)$"
    )
    for line in stats_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        key = match.group(1).lower()
        rows[key] = {
            "mean": float(match.group(2)),
            "median": float(match.group(3)),
            "min": float(match.group(4)),
            "max": float(match.group(5)),
            "std": float(match.group(6)),
            "unit": WATER_QUALITY_SPECS[key]["unit"],
            "title": WATER_QUALITY_SPECS[key]["title"],
            "label": WATER_QUALITY_SPECS[key]["label"],
        }
    return rows


@st.cache_data(show_spinner=False)
def load_water_quality_example(index_key: str):
    if index_key not in WATER_QUALITY_SPECS:
        return None

    tiles_dir = WATER_QUALITY_DIR / "tiles"
    stats = load_water_quality_statistics().get(index_key)
    if stats is None:
        return None

    base_tiles = []
    heatmap_tiles = []
    for idx in range(1, 9):
        base_path = tiles_dir / f"water_original_tile_{idx}.png"
        heatmap_path = tiles_dir / f"water_{index_key}_tile_{idx}.png"
        if not base_path.exists() or not heatmap_path.exists():
            return None
        with Image.open(base_path) as base_img:
            base_tiles.append(base_img.convert("RGB"))
        with Image.open(heatmap_path) as heatmap_img:
            heatmap_tiles.append(heatmap_img.convert("RGB"))
    return base_tiles, heatmap_tiles, stats


def heatmap_bar_html(stats: dict[str, float | str], accent_label: str) -> str:
    min_v = float(stats["min"])
    max_v = float(stats["max"])
    mean_v = float(stats["mean"])
    median_v = float(stats["median"])
    std_v = float(stats["std"])
    unit = str(stats["unit"])
    title = str(stats["title"])

    span = max(max_v - min_v, 1e-9)
    mean_pos = max(0.0, min(100.0, ((mean_v - min_v) / span) * 100.0))
    median_pos = max(0.0, min(100.0, ((median_v - min_v) / span) * 100.0))

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
      * {{ box-sizing: border-box; }}
      body {{ margin: 0; font-family: sans-serif; background: transparent; }}
      .wrap {{
        width: 100%; padding: 0 2px 0 2px;
        color: #f5f5f5;
      }}
      .card {{
        background: rgba(10, 16, 24, 0.92);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 12px 16px 12px 16px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.22);
      }}
      .top {{
        display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
        margin-bottom: 6px;
      }}
      .name {{ font-size: 22px; font-weight: 700; letter-spacing: 0.2px; }}
      .sub {{ font-size: 13px; opacity: 0.82; }}
      .bar {{
        position: relative; height: 28px; border-radius: 999px; overflow: hidden;
        background: linear-gradient(90deg,
          #0b1d51 0%,
          #123a8a 12%,
          #1d73c9 28%,
          #29b6f6 45%,
          #6fe7d1 58%,
          #c8f27d 70%,
          #ffe45c 82%,
          #ff9e43 92%,
          #e53935 100%);
        border: 1px solid rgba(255,255,255,0.16);
      }}
      .marker {{
        position: absolute; top: -5px; width: 2px; height: 38px;
        background: rgba(255,255,255,0.96);
        box-shadow: 0 0 8px rgba(0,0,0,0.55);
      }}
      .marker.mean {{ left: calc({mean_pos}% - 1px); }}
      .marker.median {{ left: calc({median_pos}% - 1px); background: rgba(255,255,255,0.45); }}
      .ticks {{
        display: flex; justify-content: space-between; margin-top: 8px;
        font-size: 12px; color: rgba(255,255,255,0.84);
      }}
      .stats {{
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
        margin-top: 14px;
      }}
      .stat {{
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 10px 12px;
      }}
      .k {{ font-size: 11px; opacity: 0.72; text-transform: uppercase; letter-spacing: 0.06em; }}
      .v {{ font-size: 18px; font-weight: 700; margin-top: 4px; }}
    </style></head><body>
      <div class="wrap">
        <div class="card">
          <div class="top">
            <div>
            <div class="name">{accent_label} heatmap scale</div>
              <div class="sub">{title} unit: {unit}</div>
            </div>
            <div class="sub">Min to max range from statistics file</div>
          </div>
          <div class="bar">
            <div class="marker mean" title="Mean"></div>
            <div class="marker median" title="Median"></div>
          </div>
          <div class="ticks">
            <span>min {min_v:.3f}</span>
            <span>mean {mean_v:.3f}</span>
            <span>max {max_v:.3f}</span>
          </div>
          <div class="stats">
            <div class="stat"><div class="k">Mean</div><div class="v">{mean_v:.3f}</div></div>
            <div class="stat"><div class="k">Median</div><div class="v">{median_v:.3f}</div></div>
            <div class="stat"><div class="k">Std Dev</div><div class="v">{std_v:.3f}</div></div>
            <div class="stat"><div class="k">Range</div><div class="v">{min_v:.3f} - {max_v:.3f}</div></div>
          </div>
        </div>
      </div>
    </body></html>"""


def water_quality_section_html(
    img_base: Image.Image,
    img_overlay: Image.Image,
    stats: dict[str, float | str],
    title: str,
    mask_opacity: float,
) -> str:
    min_v = float(stats["min"])
    max_v = float(stats["max"])
    mean_v = float(stats["mean"])
    median_v = float(stats["median"])
    std_v = float(stats["std"])
    unit = str(stats["unit"])
    span = max(max_v - min_v, 1e-9)
    mean_pos = max(0.0, min(100.0, ((mean_v - min_v) / span) * 100.0))
    median_pos = max(0.0, min(100.0, ((median_v - min_v) / span) * 100.0))

    b0 = to_b64(img_base, format="JPEG")
    b1 = to_b64(apply_opacity(img_overlay, mask_opacity), format="PNG")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
{SLIDER_CSS}
body {{ margin: 0; background: transparent; font-family: sans-serif; }}
.wrap {{ width: 100%; color: #f5f5f5; }}
.imgbox {{ position: relative; }}
#ov1 {{ clip-path: inset(0 calc(100% - var(--p1)) 0 0); }}
#lbl-l {{ left: 12px; }}
#lbl-r {{ right: 12px; }}
.bar-card {{
  margin-top: 4px;
  background: rgba(10, 16, 24, 0.92);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
  padding: 12px 16px 12px 16px;
  box-shadow: 0 8px 28px rgba(0,0,0,0.22);
}}
.top {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 6px; }}
.name {{ font-size: 22px; font-weight: 700; }}
.sub {{ font-size: 13px; opacity: 0.82; }}
.wq-bar {{
  position: relative; height: 34px; border-radius: 999px; overflow: hidden;
  background: linear-gradient(90deg,
    #0b1d51 0%, #123a8a 12%, #1d73c9 28%, #29b6f6 45%, #6fe7d1 58%,
    #c8f27d 70%, #ffe45c 82%, #ff9e43 92%, #e53935 100%);
  border: 1px solid rgba(255,255,255,0.28);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08);
}}
.wq-marker {{ position: absolute; top: -5px; width: 2px; height: 38px; background: rgba(255,255,255,0.96); }}
.wq-marker.mean {{ left: calc({mean_pos}% - 1px); }}
.wq-marker.median {{ left: calc({median_pos}% - 1px); background: rgba(255,255,255,0.45); }}
.ticks {{ display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px; color: rgba(255,255,255,0.84); }}
.stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 12px; }}
.stat {{ background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 10px 12px; }}
.k {{ font-size: 11px; opacity: 0.72; text-transform: uppercase; letter-spacing: 0.06em; }}
.v {{ font-size: 18px; font-weight: 700; margin-top: 4px; }}
</style></head><body>
<div class="wrap" id="wq-wrap">
  <div class="imgbox" id="comp" style="--p1:50%; --mask-opacity:{mask_opacity}">
    <img id="base-img" src="data:image/jpeg;base64,{b0}" style="display:block;width:100%;height:auto;">
    <div class="ov" id="ov1"><img src="data:image/png;base64,{b1}"></div>
    <div class="handle" id="h1" style="left:50%">
      <div class="bar"></div><div class="circle">&harr;</div>
    </div>
    <div class="lbl" id="lbl-l">Original</div>
    <div class="lbl" id="lbl-r">Heatmap</div>
  </div>
  <div class="bar-card">
    <div class="top">
      <div>
        <div class="name">{title}</div>
        <div class="sub">Unit: {unit}</div>
      </div>
      <div class="sub">Min to max range from statistics file</div>
    </div>
    <div class="wq-bar">
      <div class="wq-marker mean" title="Mean"></div>
      <div class="wq-marker median" title="Median"></div>
    </div>
    <div class="ticks">
      <span>min {min_v:.3f}</span>
      <span>mean {mean_v:.3f}</span>
      <span>max {max_v:.3f}</span>
    </div>
    <div class="stats">
      <div class="stat"><div class="k">Mean</div><div class="v">{mean_v:.3f}</div></div>
      <div class="stat"><div class="k">Median</div><div class="v">{median_v:.3f}</div></div>
      <div class="stat"><div class="k">Std Dev</div><div class="v">{std_v:.3f}</div></div>
      <div class="stat"><div class="k">Range</div><div class="v">{min_v:.3f} - {max_v:.3f}</div></div>
    </div>
  </div>
</div>
<script>
(function() {{
  const comp = document.getElementById('comp');
  const h1 = document.getElementById('h1');
  let p1 = 50, dragging = false;
  function apply() {{
    comp.style.setProperty('--p1', p1 + '%');
    h1.style.left = p1 + '%';
  }}
  function pct(e) {{
    const r = comp.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
    return Math.max(0, Math.min(100, x / r.width * 100));
  }}
  function onMove(e) {{ if (!dragging) return; p1 = pct(e); apply(); }}
  h1.addEventListener('mousedown', e => {{ dragging = true; e.preventDefault(); }});
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', () => dragging = false);
  h1.addEventListener('touchstart', e => {{ dragging = true; e.preventDefault(); }}, {{passive:false}});
  document.addEventListener('touchmove', onMove, {{passive:false}});
  document.addEventListener('touchend', () => dragging = false);

  function sendHeight() {{
    const wrap = document.getElementById('wq-wrap');
    const h = Math.max(
      document.documentElement.scrollHeight,
      document.body.scrollHeight,
      wrap.getBoundingClientRect().height
    );
    if (h > 0) window.parent.postMessage(
      {{isStreamlitMessage:true, type:'streamlit:setFrameHeight', height: Math.ceil(h) + 4}}, '*');
  }}

  window.addEventListener('load', sendHeight);
  window.addEventListener('resize', sendHeight);
  const ro = new ResizeObserver(sendHeight);
  ro.observe(document.body);
  ro.observe(document.getElementById('wq-wrap'));
  document.querySelectorAll('img').forEach(img => {{
    if (img.complete) return;
    img.addEventListener('load', sendHeight, {{once:true}});
  }});
  setTimeout(sendHeight, 0);
  setTimeout(sendHeight, 100);
  apply();
}})();
</script></body></html>"""


def render_osd_page(mask_opacity: float = OVERLAY_ALPHA):
    mask_opacity = float(st.session_state.get("mask_opacity", mask_opacity))
    st.title("🛢️  Oil Spill Detection")
    with st.spinner("Loading images…"):
        img0, overlay1, overlay2 = load_project_images("osd")

    st.caption("Black mask pixels are fully transparent; coloured class pixels are semi-transparent.")
    st.divider()

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.subheader("Overlay — Mask 1")
        components.html(
            single_slider_html(img0, overlay1, label_left="Mask 1", label_right="RGB", mask_opacity=mask_opacity),
            height=500,
            scrolling=False,
        )

    with col_r:
        st.subheader("Overlay — Mask 2")
        components.html(
            single_slider_html(img0, overlay2, label_left="Mask 2", label_right="RGB", mask_opacity=mask_opacity),
            height=500,
            scrolling=False,
        )

    st.divider()
    st.subheader("Both Masks — Dual Slider")
    st.caption("Drag the left handle to reveal Mask 1 · Drag the right handle to reveal Mask 2")

    components.html(
        dual_slider_html(img0, overlay1, overlay2, label1="Mask 1", label2="Mask 2", mask_opacity=mask_opacity),
        height=800,
        scrolling=False,
    )


def render_fire_page(mask_opacity: float = OVERLAY_ALPHA):
    mask_opacity = float(st.session_state.get("mask_opacity", mask_opacity))
    st.title("🔥  Fire")
    st.caption("Use the slider to compare the RGB image with the SWIR image.")

    with st.spinner("Loading fire images…"):
        rgb, swir = load_fire_images()
    components.html(
        single_slider_html(rgb, swir, label_left="RGB", label_right="Mask 1", mask_opacity=mask_opacity),
        height=max(540, int(rgb.height * 0.62) + 120),
        scrolling=False,
    )


def render_vegetation_page(mask_opacity: float = OVERLAY_ALPHA):
    mask_opacity = float(st.session_state.get("mask_opacity", mask_opacity))
    st.title("🌿  Vegetation")
    st.caption("NDVI and moisture are shown separately first, then combined into one dual-slider canvas.")

    with st.spinner("Loading vegetation images…"):
        base, ndvi_overlay, moisture_overlay = load_vegetation_images()
    st.subheader("NDVI")
    components.html(
        single_slider_html(base, ndvi_overlay, label_left="RGB", label_right="Mask 1", mask_opacity=mask_opacity),
        height=max(540, int(base.height * 0.62) + 120),
        scrolling=False,
    )

    st.subheader("Moisture")
    components.html(
        single_slider_html(base, moisture_overlay, label_left="RGB", label_right="Mask 2", mask_opacity=mask_opacity),
        height=max(540, int(base.height * 0.62) + 120),
        scrolling=False,
    )

    st.divider()
    st.subheader("NDVI + Moisture")
    st.caption("Drag the left handle for NDVI and the right handle for moisture.")
    components.html(
        dual_slider_html(base, ndvi_overlay, moisture_overlay, label1="Mask 1", label2="Mask 2", mask_opacity=mask_opacity),
        height=max(780, int(base.height * 0.95) + 180),
        scrolling=False,
    )


def render_land_use_page():
    st.title("🗺️  Land Use Classification")
    st.caption("The ESA_LULC layer is overlaid on the input image. The legend is placed at the bottom right of the overlay.")

    example_names = ["example_1", "example_2", "example_3"]
    current = st.session_state.get("land_use_example", example_names[0])
    if current not in example_names:
        current = example_names[0]
        st.session_state["land_use_example"] = current

    cols = st.columns(3, gap="medium")
    for col, example_name in zip(cols, example_names):
        label = example_name.replace("_", " ").title()
        with col:
            if st.button(
                label,
                key=f"land_use_{example_name}",
                use_container_width=True,
                type="primary" if example_name == current else "secondary",
            ):
                st.session_state["land_use_example"] = example_name
                st.rerun()

    st.divider()

    with st.spinner("Loading example…"):
        loaded = load_land_use_example(st.session_state.get("land_use_example", current))

    if loaded is None:
        st.error("One or more land use example files are missing. Expected `input_rgb.png`, `ESA_LULC.png`, and `legend.png`.")
        st.stop()

    img0, overlay, legend = loaded
    height = max(960, int(img0.height * 1.25) + 260)
    components.html(
        single_slider_html(
            img0,
            overlay,
            label_left="Mask",
            label_right="RGB",
            legend_img=legend,
        ),
        height=height,
        scrolling=False,
    )


def render_land_use_page(mask_opacity: float = OVERLAY_ALPHA):
    render_land_use_tiles_page(mask_opacity=mask_opacity)


def render_water_quality_page(mask_opacity: float = OVERLAY_ALPHA):
    st.title("🌊  Water Quality")
    st.caption("Choose an example first, then choose one index to display on this page.")

    mask_opacity = float(st.session_state.get("mask_opacity", mask_opacity))
    current_example = int(st.session_state.get("water_quality_example", 0))
    example_numbers = list(range(8))
    for row_start in range(0, 8, 4):
        cols = st.columns(4, gap="small")
        for idx, col in zip(example_numbers[row_start:row_start + 4], cols):
            with col:
                if st.button(
                    f"Example {idx + 1}",
                    key=f"water_quality_example_{idx}",
                    use_container_width=True,
                    type="primary" if idx == current_example else "secondary",
                ):
                    st.session_state["water_quality_example"] = idx
                    st.rerun()

    st.divider()
    index_names = ["chl", "tss", "cdom"]
    current_index = st.session_state.get("water_quality_index", index_names[0])
    if current_index not in index_names:
        current_index = index_names[0]
        st.session_state["water_quality_index"] = current_index

    index_cols = st.columns(3, gap="medium")
    for col, key in zip(index_cols, index_names):
        spec = WATER_QUALITY_SPECS[key]
        with col:
            if st.button(
                spec["title"],
                key=f"water_quality_index_{key}",
                use_container_width=True,
                type="primary" if key == current_index else "secondary",
            ):
                st.session_state["water_quality_index"] = key
                st.rerun()

    tile_index = max(0, min(7, int(st.session_state.get("water_quality_example", current_example))))
    selected = st.session_state.get("water_quality_index", current_index)
    spec = WATER_QUALITY_SPECS[selected]

    with st.spinner(f"Loading {spec['title']}..."):
        loaded = load_water_quality_example(selected)

    if loaded is None:
        st.error("Missing water quality files. Expected the original TCI image, three heatmaps, and the statistics text file.")
        st.stop()

    base_tiles, heatmap_tiles, stats = loaded
    base_img = base_tiles[tile_index]
    heatmap_img = heatmap_tiles[tile_index]

    st.subheader(f"{spec['title']} - example {tile_index + 1}")
    components.html(
        water_quality_section_html(
            base_img,
            heatmap_img,
            stats,
            spec["title"],
            mask_opacity=mask_opacity,
        ),
        height=1240,
        scrolling=False,
    )


def render_land_use_page(mask_opacity: float = OVERLAY_ALPHA):
    render_land_use_tiles_page(mask_opacity=mask_opacity)


def render_land_use_tiles_page(mask_opacity: float = OVERLAY_ALPHA):
    mask_opacity = float(st.session_state.get("mask_opacity", mask_opacity))
    st.title("🗺️  Land Use Classification")
    st.caption("The Casablanca scene is split into three tiles, and the legend is embedded in the bottom-right corner.")

    example_names = ["example_1", "example_2", "example_3"]
    current = st.session_state.get("land_use_example", example_names[0])
    if current not in example_names:
        current = example_names[0]
        st.session_state["land_use_example"] = current

    cols = st.columns(3, gap="medium")
    for idx, (col, example_name) in enumerate(zip(cols, example_names), start=1):
        with col:
            if st.button(
                f"Example {idx}",
                key=f"land_use_{example_name}",
                use_container_width=True,
                type="primary" if example_name == current else "secondary",
            ):
                st.session_state["land_use_example"] = example_name
                st.rerun()

    st.divider()

    selected = st.session_state.get("land_use_example", current)
    tile_index = int(selected.rsplit("_", 1)[-1])

    with st.spinner("Loading tile..."):
        loaded = load_land_use_tile_example(tile_index)

    if loaded is None:
        st.error("One or more tile files are missing under `data/land use/tiles`.")
        st.stop()

    img0, overlay, legend = loaded
    height = max(720, int(img0.height * 0.9) + 140)
    components.html(
        single_slider_html(
            img0,
            overlay,
            label_left="Mask",
            label_right="RGB",
            legend_img=legend,
            mask_opacity=mask_opacity,
            legend_scale=0.96,
            legend_resize_scale=3.5,
        ),
        height=height,
        scrolling=False,
    )


def render_land_use_page(mask_opacity: float = OVERLAY_ALPHA):
    render_land_use_tiles_page(mask_opacity=mask_opacity)


def render_land_use_page():
    st.title("🗺️  Land Use Classification")
    st.caption("The Casablanca scene is split into three tiles, and the legend is embedded in the bottom-right corner.")

    example_names = ["example_1", "example_2", "example_3"]
    current = st.session_state.get("land_use_example", example_names[0])
    if current not in example_names:
        current = example_names[0]
        st.session_state["land_use_example"] = current

    cols = st.columns(3, gap="medium")
    for idx, (col, example_name) in enumerate(zip(cols, example_names), start=1):
        with col:
            if st.button(
                f"Example {idx}",
                key=f"land_use_{example_name}",
                use_container_width=True,
                type="primary" if example_name == current else "secondary",
            ):
                st.session_state["land_use_example"] = example_name
                st.rerun()

    st.divider()

    selected = st.session_state.get("land_use_example", current)
    tile_index = int(selected.rsplit("_", 1)[-1])

    with st.spinner("Loading tile..."):
        loaded = load_land_use_tile_example(tile_index)

    if loaded is None:
        st.error("One or more tile files are missing under `data/land use/tiles`.")
        st.stop()

    img0, overlay = loaded
    height = max(720, int(img0.height * 0.9) + 140)
    components.html(
        single_slider_html(
            img0,
            overlay,
            label_left="Mask",
            label_right="RGB",
        ),
        height=height,
        scrolling=False,
    )


def render_land_use_page():
    st.title("🗺️  Land Use Classification")
    st.caption("The Casablanca scene is split into three tiles, and the legend is embedded in the bottom-right corner.")

    example_names = ["example_1", "example_2", "example_3"]
    current = st.session_state.get("land_use_example", example_names[0])
    if current not in example_names:
        current = example_names[0]
        st.session_state["land_use_example"] = current

    cols = st.columns(3, gap="medium")
    for idx, (col, example_name) in enumerate(zip(cols, example_names), start=1):
        with col:
            if st.button(
                f"Example {idx}",
                key=f"land_use_{example_name}",
                use_container_width=True,
                type="primary" if example_name == current else "secondary",
            ):
                st.session_state["land_use_example"] = example_name
                st.rerun()

    st.divider()

    selected = st.session_state.get("land_use_example", current)
    tile_index = int(selected.rsplit("_", 1)[-1])

    with st.spinner("Loading tile..."):
        loaded = load_land_use_tile_example(tile_index)

    if loaded is None:
        st.error("One or more tile files are missing under `data/land use/tiles`.")
        st.stop()

    img0, overlay = loaded
    height = max(720, int(img0.height * 0.9) + 140)
    components.html(
        single_slider_html(
            img0,
            overlay,
            label_left="Mask",
            label_right="RGB",
        ),
        height=height,
        scrolling=False,
    )


def render_land_use_page():
    st.title("🗺️  Land Use Classification")
    st.caption("The ESA_LULC layer is overlaid on the input image. Example 4 is split into tiles so it loads more easily.")

    example_names = ["example_1", "example_2", "example_3", "example_4"]
    current = st.session_state.get("land_use_example", example_names[0])
    if current not in example_names:
        current = example_names[0]
        st.session_state["land_use_example"] = current

    cols = st.columns(4, gap="medium")
    for col, example_name in zip(cols, example_names):
        label = example_name.replace("_", " ").title()
        with col:
            if st.button(
                label,
                key=f"land_use_{example_name}",
                use_container_width=True,
                type="primary" if example_name == current else "secondary",
            ):
                st.session_state["land_use_example"] = example_name
                st.rerun()

    st.divider()

    selected = st.session_state.get("land_use_example", current)
    with st.spinner("Loading example..."):
        loaded = load_land_use_tiles(selected) if selected == "example_4" else load_land_use_example(selected)

    if loaded is None:
        st.error("One or more land use example files are missing.")
        st.stop()

    if selected == "example_4":
        base_tiles, overlay_tiles, _ = loaded
        st.caption("This example is split into 4 tiles so each part loads more easily.")

        tile_key = int(st.session_state.get("land_use_example_4_tile", 0))
        tile_buttons = st.columns(4, gap="small")
        for idx, col in enumerate(tile_buttons):
            with col:
                if st.button(
                    f"Tile {idx + 1}",
                    key=f"land_use_example_4_tile_{idx}",
                    use_container_width=True,
                    type="primary" if idx == tile_key else "secondary",
                ):
                    st.session_state["land_use_example_4_tile"] = idx
                    st.rerun()

        tile_key = max(0, min(3, int(st.session_state.get("land_use_example_4_tile", 0))))
        img0 = base_tiles[tile_key]
        overlay = overlay_tiles[tile_key]
        height = max(720, int(img0.height * 0.9) + 140)
        components.html(
            single_slider_html(
                img0,
                overlay,
                label_left="Mask",
                label_right="RGB",
            ),
            height=height,
            scrolling=False,
        )
        return

    img0, overlay, legend = loaded
    height = max(960, int(img0.height * 1.25) + 260)
    components.html(
        single_slider_html(
            img0,
            overlay,
            label_left="Mask",
            label_right="RGB",
            legend_img=legend,
        ),
        height=height,
        scrolling=False,
    )
