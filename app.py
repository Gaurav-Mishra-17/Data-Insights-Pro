import streamlit as st
from PIL import Image
import io
import base64
from urllib.request import urlopen

# App configuration
st.set_page_config(
    page_title="DataInsights Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load an image from URL
def load_image_from_url(url):
    try:
        with urlopen(url) as response:
            image_data = response.read()
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Landing page content
def main():
    # Header section
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image("https://d3an9kf42ylj3p.cloudfront.net/uploads/2022/08/pg_analyticstools_aug22.jpg", use_container_width=True)

    with col2:
        st.title("DataInsights Pro")
        st.subheader("AI-Powered Data Analytics Platform, No Technical Expertise Required")
    
    st.markdown("---")
    
    # Introduction
    st.markdown("""
    ## Welcome to your AI Analytics Assistant
    
    Unlock the power of your data without needing technical expertise. Our platform guides you through every step of the data analytics process with intelligent suggestions and automated insights.
    
    ### What You Can Do:
    """)
    
    # Features overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📤 Upload & Clean")
        st.markdown("""
        - Upload various data formats
        - Automated data quality assessment
        - One-click data cleaning
        - Intelligent missing value handling
        """)
        st.image("https://www.poimapper.com/wp-content/uploads/2019/03/0_FR2egZQUOVJ_4NcS.png", use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Explore & Visualize")
        st.markdown("""
        - Interactive data filtering
        - Auto-generated visualizations
        - Correlation analysis
        - Customizable dashboards
        """)
        st.image("https://www.elegantthemes.com/blog/wp-content/uploads/2019/05/featured-data-visualization.png", use_container_width=True)

    with col3:
        st.markdown("#### 🧠 Analyze & Predict")
        st.markdown("""
        - Ask questions in natural language
        - Automated statistical analysis
        - Machine learning without coding
        - Trend forecasting & insights
        """)
        st.image("data:image/png;base64,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", use_container_width=True)
    
    st.markdown("---")
    
    # Getting started section
    st.markdown("## Getting Started")
    st.markdown("""
    1. Click on the **Data Upload** page in the sidebar
    2. Upload your dataset (CSV, Excel, etc.)
    3. Follow the AI-powered workflow to analyze your data
    4. Generate insights, visualizations, and predictions with just a few clicks
    """)
    
    # Call-to-action
    st.markdown("---")
    st.info("👈 Start by selecting **Data Upload** from the sidebar to begin your data analytics journey!")

# Initialize session state for data
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'data_info' not in st.session_state:
    st.session_state.data_info = {}
if 'cleaning_history' not in st.session_state:
    st.session_state.cleaning_history = []
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = []
if 'models' not in st.session_state:
    st.session_state.models = {}

if __name__ == "__main__":
    main()
