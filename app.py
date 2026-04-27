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
        st.image("https://cdn.careerfoundry.com/en/wp-content/uploads/2020/10/data-analytics-tools.webp", use_container_width=True)
    
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
