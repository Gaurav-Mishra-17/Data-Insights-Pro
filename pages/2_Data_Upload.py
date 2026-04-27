import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime
import sys

# Add utils to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_processor import DataProcessor
from utils.helpers import get_data_summary

st.set_page_config(
    page_title="DataInsights Pro",
    layout="wide"
)

def main():
    st.title("📤 Data Upload & Summary")
    st.markdown("### Upload your Dataset to begin the Analytics Miracle")
    
    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose a CSV, Excel, or TSV file",
        type=["csv", "xlsx", "xls", "tsv"],
        help="Upload your dataset file (max size: 200MB)"
    )
    
    # Process new file upload first to overwrite existing data
    if uploaded_file is not None:
        try:
            # Detect file type and read accordingly
            filename = uploaded_file.name
            file_extension = os.path.splitext(filename)[1].lower()
            
            # Progress indicator
            with st.spinner("Reading your data..."):
                if file_extension == '.csv':
                    data = pd.read_csv(uploaded_file, na_values=[""], keep_default_na=False)
                elif file_extension == '.tsv':
                    data = pd.read_csv(uploaded_file, sep='\t', na_values=[""], keep_default_na=False)
                elif file_extension in ['.xlsx', '.xls']:
                    # Multi-sheet handling for Excel files
                    excel_file = pd.ExcelFile(uploaded_file)
                    if len(excel_file.sheet_names) > 1:
                        sheet_name = st.selectbox("Select a sheet:", excel_file.sheet_names)
                        data = pd.read_excel(uploaded_file, sheet_name=sheet_name, na_values=[""], keep_default_na=False)
                    else:
                        data = pd.read_excel(uploaded_file, na_values=[""], keep_default_na=False)
                else:
                    st.error("Unsupported file format")
                    return
            
            # Overwrite session state with new data
            st.session_state.data = data.copy()
            st.session_state.original_data = data.copy()
            st.session_state.filename = filename
            st.session_state.cleaning_history = []
            st.session_state.visualizations = []
            st.session_state.models = {}
            # Clear data_info to ensure it’s regenerated for new data
            if 'data_info' in st.session_state:
                del st.session_state.data_info
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return
    
    # Display data from session state if it exists
    if 'data' in st.session_state and st.session_state.data is not None:
        data = st.session_state.data
        filename = st.session_state.filename
        
        # Display success message
        st.success(f"✅ Successfully loaded '{filename}' with {data.shape[0]} rows and {data.shape[1]} columns.")
        
        # Data summary
        # st.markdown("## Data Preview & Summary")
        
        # Preview the first few rows
        st.markdown("### Data Preview")
        st.dataframe(data.head(5))

        # Generate and store data info if not already present
        if 'data_info' not in st.session_state:
            with st.spinner("Analyzing your data..."):
                st.session_state.data_info = get_data_summary(data)
        
        # AI-powered dataset insights
        with st.expander("Data Insights", expanded=True):
            st.markdown("### Automated Data Assessment")
             
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{data.shape[0]:,}")
            with col2:
                st.metric("Columns", f"{data.shape[1]:,}")
            with col3:
                st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
            with col4:
                st.metric("Data Types", f"{data.dtypes.nunique()}")

            # Memory usage
            memory_usage = data.memory_usage(deep=True).sum()
            if memory_usage < 1024:
                memory_str = f"{memory_usage} bytes"
            elif memory_usage < 1024**2:
                memory_str = f"{memory_usage/1024:.2f} KB"
            elif memory_usage < 1024**3:
                memory_str = f"{memory_usage/(1024**2):.2f} MB"
            else:
                memory_str = f"{memory_usage/(1024**3):.2f} GB"
            
            st.info(f"Memory Usage: {memory_str}")

            # Data quality overview
            quality_score = DataProcessor.calculate_quality_score(data)
            
            # Display quality score with color-coded indicator
            col1, col2 = st.columns([1, 3])
            with col1:
                if quality_score >= 80:
                    st.success(f"Quality Score: {quality_score}/100")
                elif quality_score >= 50:
                    st.warning(f"Quality Score: {quality_score}/100")
                else:
                    st.error(f"Quality Score: {quality_score}/100")
            
            with col2:
                if quality_score >= 80:
                    st.markdown("Your data appears to be high quality. Minor cleaning recommended.")
                elif quality_score >= 50:
                    st.markdown("Your data has some quality issues. Cleaning recommended.")
                else:
                    st.markdown("Your data requires significant cleaning before analysis.")
            
            # Potential issues detected
            issues = DataProcessor.detect_issues(data)
            if issues:
                st.markdown("#### Potential Issues Detected:")
                for issue in issues:
                    st.markdown(f"- {issue}")
                
                st.info("Proceed to the **Data Cleaning** page to resolve these issues.")
            else:
                st.success("No major issues detected in your dataset.")
        
        # Display column types
        with st.expander("Summary Statistics", expanded=True):
            # Get numeric and non-numeric columns
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            
            if numeric_cols:
                # Display summary statistics for numeric columns
                st.markdown("#### Numeric Columns")
                numeric_stats = data[numeric_cols].describe().T
                # Add additional metrics
                if not numeric_stats.empty:
                    numeric_stats['missing'] = data[numeric_cols].isna().sum()
                    numeric_stats['missing_pct'] = (data[numeric_cols].isna().sum() / len(data) * 100).round(2)
                    st.dataframe(numeric_stats)
            
            # Non-numeric columns summary
            non_numeric_cols = data.select_dtypes(exclude=np.number).columns.tolist()
            
            if non_numeric_cols:
                st.markdown("#### Non-Numeric Columns")
                # Create a custom summary for non-numeric columns
                non_numeric_summary = []
                
                for col in non_numeric_cols:
                    unique_count = data[col].nunique()
                    missing_count = data[col].isna().sum()
                    missing_pct = (missing_count / len(data) * 100).round(2)
                    
                    # Get the most common value and its frequency
                    if unique_count > 0:
                        most_common = data[col].value_counts().index[0] if not data[col].value_counts().empty else None
                        most_common_count = data[col].value_counts().iloc[0] if not data[col].value_counts().empty else 0
                        most_common_pct = (most_common_count / data[col].count() * 100).round(2)
                    else:
                        most_common = None
                        most_common_count = 0
                        most_common_pct = 0
                    
                    non_numeric_summary.append({
                        'Column': col,
                        'Type': str(data[col].dtype),  # Convert dtype to string for JSON serialization
                        'Unique Values': unique_count,
                        'Missing Values': missing_count,
                        'Missing %': missing_pct,
                        'Top Values': str(most_common)[:20] + ('...' if str(most_common) and len(str(most_common)) > 20 else ''),
                        'Frequency': most_common_count,
                        'Frequency %': most_common_pct
                    })
                
                non_numeric_df = pd.DataFrame(non_numeric_summary)
                st.dataframe(non_numeric_df)
                   
        # Add "What's Next" section
        st.markdown("---")
        st.markdown("## What's Next?")
        st.info("👉 Go to the **Data Cleaning** page to prepare your data for analysis")
    
    else:
        # Show placeholder when no file is uploaded and no data in session state
        if 'data' not in st.session_state or st.session_state.data is None:
            st.markdown("### Sample Datasets")
            st.markdown("Don't have a dataset? Try one of these sample datasets to explore the platform capabilities:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Retail Sales Dataset"):
                    # Mock load dataset button - this would normally load sample data
                    st.info("Please upload a dataset to continue. Sample datasets are only placeholders in this version.")
            
            with col2:
                if st.button("Customer Survey Data"):
                    st.info("Please upload a dataset to continue. Sample datasets are only placeholders in this version.")
            
            with col3:
                if st.button("Time Series Data"):
                    st.info("Please upload a dataset to continue. Sample datasets are only placeholders in this version.")
            
            # Display platform capabilities overview
            st.markdown("---")
                    
            st.markdown("""
            ### Platform Features
            
            Our AI-powered analytics platform will guide you through:
            
            1. **Automated Data Cleaning** - Fix missing values, outliers, and data type issues
            2. **Interactive Exploration** - Filter, sort, and analyze your data with ease
            3. **Smart Visualizations** - Generate charts and graphs with a single click
            4. **AI Insights** - Get automated analysis and recommendations
            5. **Statistical Analysis** - Run tests and models without coding
            6. **Predictive Modeling** - Build and train machine learning models
            
            To get started, upload your data file above.
            """)

if __name__ == "__main__":
    main()