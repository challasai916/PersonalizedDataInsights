import streamlit as st
import pandas as pd
import numpy as np
from pages import home, data_upload, exploratory_analysis, user_behavior, recommendation, segmentation, ab_testing, metrics

# Set page config
st.set_page_config(
    page_title="Personalization Data Science Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'recommendation_model' not in st.session_state:
    st.session_state.recommendation_model = None
if 'segmentation_model' not in st.session_state:
    st.session_state.segmentation_model = None
if 'ab_test_results' not in st.session_state:
    st.session_state.ab_test_results = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "Data Upload", "Exploratory Analysis", "User Behavior", 
     "Recommendation Engine", "User Segmentation", "A/B Testing", "Performance Metrics"]
)

# Render the selected page
if page == "Home":
    home.show()
elif page == "Data Upload":
    data_upload.show()
elif page == "Exploratory Analysis":
    exploratory_analysis.show()
elif page == "User Behavior":
    user_behavior.show()
elif page == "Recommendation Engine":
    recommendation.show()
elif page == "User Segmentation":
    segmentation.show()
elif page == "A/B Testing":
    ab_testing.show()
elif page == "Performance Metrics":
    metrics.show()

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This platform helps analyze user behavior and implement personalization features "
    "for online platforms."
)
