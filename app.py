import streamlit as st

from viewer import (
    render_fire_page,
    render_land_use_tiles_page,
    render_osd_page,
    render_water_quality_page,
    render_vegetation_page,
)

st.set_page_config(layout="wide", page_title="Remote Sensing Showcase")

mask_opacity = st.sidebar.slider(
    "Mask opacity",
    min_value=0.0,
    max_value=1.0,
    value=float(st.session_state.get("mask_opacity", 0.65)),
    step=0.05,
    key="mask_opacity",
)

navigation = st.navigation(
    {
        "Remote Sensing": [
            st.Page(render_osd_page, title="Oil Spill Detection", url_path="oil-spill-detection", default=True),
            st.Page(render_fire_page, title="Fire", url_path="fire"),
            st.Page(render_vegetation_page, title="Vegetation", url_path="vegetation"),
        ],
        "Land Use Classification": [
            st.Page(render_land_use_tiles_page, title="Land Use Examples", url_path="land-use-examples"),
        ],
        "Water Quality": [
            st.Page(render_water_quality_page, title="Water Quality", url_path="water-quality"),
        ],
    },
    position="sidebar",
    expanded=True,
)

navigation.run()
