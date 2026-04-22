import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

DATA_DIR = Path(__file__).parent / "data"
LAND_USE_DIR = DATA_DIR / "land use"
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


def make_overlay(base: Image.Image, mask: Image.Image, alpha: float = OVERLAY_ALPHA) -> Image.Image:
    base_rgb = base.convert("RGB")
    mask_rgba = mask.convert("RGBA")
    arr = np.array(mask_rgba, dtype=np.uint8)
    is_black = (arr[:, :, 0] < 10) & (arr[:, :, 1] < 10) & (arr[:, :, 2] < 10)
    arr[:, :, 3] = np.where(is_black, 0, int(alpha * 255))
    overlay = Image.fromarray(arr, "RGBA")
    result = base_rgb.convert("RGBA")
    result.paste(overlay, (0, 0), overlay)
    return result.convert("RGB")


def resize_to(img: Image.Image, max_width: int = 1100) -> Image.Image:
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    return img


def fit_to_box(img: Image.Image, max_width: int, max_height: int) -> Image.Image:
    ratio = min(max_width / img.width, max_height / img.height)
    if ratio < 1:
        new_size = (max(1, int(img.width * ratio)), max(1, int(img.height * ratio)))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def to_b64(img: Image.Image, quality: int = 82) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def add_legend_overlay(base_img: Image.Image, legend_img: Image.Image, padding: int = 12) -> Image.Image:
    result = base_img.convert("RGB").copy()
    max_width = max(160, int(result.width * 0.26))
    max_height = max(120, int(result.height * 0.45))
    legend = fit_to_box(legend_img.convert("RGB"), max_width=max_width, max_height=max_height)
    x = max(padding, result.width - legend.width - padding)
    y = max(padding, result.height - legend.height - padding)
    result.paste(legend, (x, y))
    return result


def single_slider_html(
    img_base: Image.Image,
    img_overlay: Image.Image,
    label_left: str = "Base",
    label_right: str = "Overlay",
    legend_img: Image.Image | None = None,
) -> str:
    b0 = to_b64(img_base)
    b1 = to_b64(img_overlay)
    legend_block = ""
    if legend_img is not None:
        b_legend = to_b64(legend_img)
        legend_block = f"""
  <div class="legend">
    <img src="data:image/jpeg;base64,{b_legend}">
  </div>
"""
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
{SLIDER_CSS}
#ov1 {{ clip-path: inset(0 calc(100% - var(--p1)) 0 0); }}
#lbl-l {{ left: 12px; }}
#lbl-r {{ right: 12px; }}
.legend {{
  position: absolute; right: 12px; bottom: 12px;
  max-width: 28%; max-height: 42%;
  z-index: 15; pointer-events: none;
}}
.legend img {{
  display: block; width: 100%; height: auto;
}}
</style></head><body>
<div id="comp" style="--p1:50%">
  <img id="base-img" src="data:image/jpeg;base64,{b0}">
  <div class="ov" id="ov1"><img src="data:image/jpeg;base64,{b1}"></div>
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
) -> str:
    b0 = to_b64(img_base)
    b1 = to_b64(img_ov1)
    b2 = to_b64(img_ov2)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
{SLIDER_CSS}
#ov1 {{ clip-path: inset(0 calc(100% - var(--p1)) 0 0); }}
#ov2 {{ clip-path: inset(0 0 0 var(--p2)); }}
#lbl1 {{ left: 12px; }}
#lblc {{ left: 50%; transform: translateX(-50%); }}
#lbl2 {{ right: 12px; }}
</style></head><body>
<div id="comp" style="--p1:33%;--p2:67%">
  <img id="base-img" src="data:image/jpeg;base64,{b0}">
  <div class="ov" id="ov1"><img src="data:image/jpeg;base64,{b1}"></div>
  <div class="ov" id="ov2"><img src="data:image/jpeg;base64,{b2}"></div>
  <div class="handle" id="h1" style="left:33%">
    <div class="bar"></div><div class="circle">⇔</div>
  </div>
  <div class="handle" id="h2" style="left:67%">
    <div class="bar"></div><div class="circle">⇔</div>
  </div>
  <div class="lbl" id="lbl1">{label1}</div>
  <div class="lbl" id="lblc">Base</div>
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
def load_land_use_example(example_name: str):
    d = LAND_USE_DIR / example_name
    base_path = d / "input_rgb.png"
    mask_path = d / "ESA_LULC.png"
    legend_path = d / "legend.png"
    if not base_path.exists() or not mask_path.exists() or not legend_path.exists():
        return None

    base = resize_to(Image.open(base_path).convert("RGB"))
    mask = resize_to(Image.open(mask_path).convert("RGB"))
    if mask.size != base.size:
        mask = mask.resize(base.size, Image.LANCZOS)
    legend = Image.open(legend_path).convert("RGB")
    overlay = make_overlay(base, white_to_green(mask))
    return base, overlay, legend


def render_osd_page():
    st.title("🛢️  Oil Spill Detection")
    with st.spinner("Loading images…"):
        img0, overlay1, overlay2 = load_project_images("osd")

    st.caption("Black mask pixels are fully transparent; coloured class pixels are semi-transparent.")
    st.divider()

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.subheader("Overlay — Mask 1")
        components.html(
            single_slider_html(img0, overlay1, label_left="Mask 1", label_right="Base"),
            height=500,
            scrolling=False,
        )

    with col_r:
        st.subheader("Overlay — Mask 2")
        components.html(
            single_slider_html(img0, overlay2, label_left="Mask 2", label_right="Base"),
            height=500,
            scrolling=False,
        )

    st.divider()
    st.subheader("Both Masks — Dual Slider")
    st.caption("Drag the left handle to reveal Mask 1 · Drag the right handle to reveal Mask 2")

    components.html(
        dual_slider_html(img0, overlay1, overlay2, label1="Mask 1", label2="Mask 2"),
        height=800,
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
            label_left="ESA_LULC Overlay",
            label_right="Input RGB",
            legend_img=legend,
        ),
        height=height,
        scrolling=False,
    )
