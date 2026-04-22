import streamlit as st

from viewer import (
    render_land_use_page,
    render_osd_page,
)

st.set_page_config(layout="wide", page_title="Remote Sensing Showcase")

navigation = st.navigation(
    {
        "Remote Sensing": [
            st.Page(render_osd_page, title="Oil Spill Detection", icon="🛢️", url_path="oil-spill-detection", default=True),
        ],
        "Land Use Classification": [
            st.Page(render_land_use_page, title="Land Use Examples", icon="🗺️", url_path="land-use-examples"),
        ],
    },
    position="sidebar",
    expanded=True,
)

navigation.run()
