import streamlit as st

from viewer import (
    render_fire_page,
    render_land_use_page,
    render_osd_page,
    render_vegetation_page,
)

st.set_page_config(layout="wide", page_title="Remote Sensing Showcase")

navigation = st.navigation(
    {
        "Remote Sensing": [
            st.Page(render_osd_page, title="Oil Spill Detection", url_path="oil-spill-detection", default=True),
            st.Page(render_fire_page, title="Fire", url_path="fire"),
            st.Page(render_vegetation_page, title="Vegetation", url_path="vegetation"),
        ],
        "Land Use Classification": [
            st.Page(render_land_use_page, title="Land Use Examples", url_path="land-use-examples"),
        ],
    },
    position="sidebar",
    expanded=True,
)

navigation.run()
