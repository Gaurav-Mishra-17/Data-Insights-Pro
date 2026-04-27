import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Add utils to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_processor import DataProcessor
from utils.helpers import get_data_summary

st.set_page_config(
    page_title="DataInsights Pro",
    layout="wide"
)

def main():
    st.title("🧹 Data Cleaning & Preparation")
    
    # Check if data is uploaded
    if "data" not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ Please upload a Dataset in the **Data Upload** page.")
        st.stop()
    
    data = st.session_state.data
    
    # Sidebar with cleaning options
    # st.sidebar.header("Cleaning Options")
    
    # Tabs for different cleaning operations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Summary", 
        "Missing Values", 
        "Outliers", 
        "Data Types", 
        "Duplicates",
        "Feature Engineering"
    ])

    # Summary tab
    with tab1:
        st.header("Data Quality Summary")
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{data.shape[0]:,}")
        with col2:
            missing_pct = (data.isna().sum().sum() / (data.shape[0] * data.shape[1]) * 100)
            st.metric("Missing Values", f"{missing_pct:.2f}%")
        with col3:
            duplicate_pct = (data.duplicated().sum() / data.shape[0] * 100)
            st.metric("Duplicates", f"{duplicate_pct:.2f}%")
        with col4:
            quality_score = DataProcessor.calculate_quality_score(data)
            st.metric("Quality Score", f"{quality_score}/100")
        
        st.info("Note: Blank cells are treated as missing values. Values like 'None' or 'N/A' or 'Unknown' are considered valid.")

        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(data.head(5))
        
        # Missing values heatmap
        st.subheader("Missing Values Representation")
        # Limit to displaying at most 50 columns for performance
        display_cols = data.columns[:50] if len(data.columns) > 50 else data.columns
        
        # Create missing values heatmap
        missing_data = data[display_cols].isna().astype(float)
        
        # if missing_data.any().any():
            # Create a sample of rows for better visualization
            # sample_size = min(100, data.shape[0])
            # fig = px.imshow(
                # missing_data.head(sample_size),
                # color_continuous_scale=['#FFFFFF', '#0078D7'],
                # labels=dict(x="Columns", y="Rows", color="Missing"),
                # title=f"Missing Values Heatmap (First {sample_size} rows)",
                # width=1000,
                # height=600
            #)
            # st.plotly_chart(fig, use_container_width=True)
            
            # Display missing values by column
        missing_by_col = data.isna().sum().sort_values(ascending=False)
        missing_by_col = missing_by_col[missing_by_col > 0]
        
        if not missing_by_col.empty:
            # st.subheader("Missing Values by Column")
            fig = px.bar(
                x=missing_by_col.index, 
                y=missing_by_col.values,
                labels={'x': 'Column', 'y': 'Missing Count'},
                title="Missing Value Counts by Column",
                color_discrete_sequence=['#0078D7']
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found in the dataset!")
        
        # AI-powered recommendations
        # st.subheader("AI Recommendations")
        # issues = DataProcessor.detect_issues(data)
        
        # if issues:
            # for issue in issues:
                # st.write(f"🤖 {issue}")
        # else:
            # st.success("No major issues detected in your dataset.")
    
    # Missing values tab
    with tab2:
        st.header("Handle Missing Values")
        
        # Get columns with missing values
        cols_with_missing = data.columns[data.isna().any()].tolist()
        
        if not cols_with_missing:
            st.success("✅ No missing values detected in your dataset!")
        else:
            st.info(f"Found {len(cols_with_missing)} columns with missing values")
            
            # Select column to fix
            selected_col = st.selectbox(
                "Select a column to fix missing values:",
                cols_with_missing,
                key="missing_col_select"
            )
            
            # Display missing percentage for selected column
            missing_pct = (data[selected_col].isna().sum() / len(data) * 100)
            st.markdown(f"**Missing values in '{selected_col}':** {missing_pct:.2f}% ({data[selected_col].isna().sum():,} out of {len(data):,} rows)")
            
            # Display distribution of non-missing values
            st.subheader(f"Distribution of non-missing values in '{selected_col}'")
            
            if pd.api.types.is_numeric_dtype(data[selected_col]):
                # For numeric columns
                fig = px.histogram(
                    data[data[selected_col].notna()], 
                    x=selected_col,
                    marginal="box", 
                    title=f"Distribution of {selected_col}",
                    color_discrete_sequence=['#0078D7']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{data[selected_col].mean():.2f}")
                with col2:
                    st.metric("Median", f"{data[selected_col].median():.2f}")
                with col3:
                    st.metric("Min", f"{data[selected_col].min():.2f}")
                with col4:
                    st.metric("Max", f"{data[selected_col].max():.2f}")
                
                # AI recommended handling method
                st.markdown("### AI Recommended Action")
                
                # Determine if the column has outliers
                q1 = data[selected_col].quantile(0.25)
                q3 = data[selected_col].quantile(0.75)
                iqr = q3 - q1
                outlier_count = ((data[selected_col] < (q1 - 1.5 * iqr)) | (data[selected_col] > (q3 + 1.5 * iqr))).sum()
                outlier_pct = outlier_count / data[selected_col].count() * 100
                
                if outlier_pct > 5:
                    recommended_method = "median"
                    reason = "The data contains outliers, so median is more robust than mean."
                elif abs(data[selected_col].skew()) > 1:
                    recommended_method = "median"
                    reason = "The data is skewed, so median is more representative than mean."
                else:
                    recommended_method = "mean"
                    reason = "The data is roughly symmetric without significant outliers."
                
                st.markdown(f"**Recommended method:** Fill with {recommended_method}")
                st.markdown(f"**Reason:** {reason}")

            elif isinstance(data[selected_col].dtype, pd.CategoricalDtype) or data[selected_col].nunique() < 10:
                # For categorical/low cardinality columns
                value_counts = data[selected_col].value_counts().reset_index()
                value_counts.columns = [selected_col, 'Count']
                fig = px.bar(
                    value_counts, 
                    x=selected_col, 
                    y='Count',
                    title=f"Value Counts for {selected_col}",
                    color_discrete_sequence=['#0078D7']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                most_common = data[selected_col].mode()[0]
                
                # AI recommended handling method
                st.markdown("### AI Recommended Action")
                recommended_method = "mode"
                reason = "For categorical data, the most frequent value often represents the typical case."
                st.markdown(f"**Recommended method:** Fill with most frequent value ('{most_common}')")
                st.markdown(f"**Reason:** {reason}")
                
            else:
                # For other data types (object, datetime, etc.)
                st.write("Preview of non-missing values:")
                st.write(data[data[selected_col].notna()][selected_col].head())
                
                # AI recommended handling method
                st.markdown("### AI Recommended Action")
                recommended_method = "constant"
                reason = "This preserves information about missingness for text or complex data."
                st.markdown("**Recommended method:** Fill with 'Unknown' or create a new category")
                st.markdown(f"**Reason:** {reason}")
            
            # Handling options
            st.subheader("Choose handling method")
            
            handling_method = st.radio(
                "How would you like to handle missing values?",
                ["Fill with Mean", "Fill with Median", "Fill with Mode", "Fill with Constant", "Drop rows with Missing Values", "Do Nothing"],
                index=0 if recommended_method == "mean" else (1 if recommended_method == "median" else 2),
                horizontal=True
            )

            # Display explanation for each cleaning method
            if handling_method == "Fill with Mean":
                st.info("This will replace missing values with the Mean of the column. Suitable for Numeric Columns without Outliers.")
            elif handling_method == "Fill with Median":
                st.info("This will replace missing values with the Median of the column. Suitable for Numeric Columns with Outliers.")
            elif handling_method == "Fill with Mode":
                st.info("This will replace missing values with the Mode of the column. Suitable for Categorical Columns.")
            elif handling_method == "Fill with Constant":
                st.info("This will replace missing values with a Constant value (0 or Unknown). Suitable for all Columns.")
            elif handling_method == "Drop rows with Missing Values":
                st.info("This will drop rows with Missing Values from the Dataset.")

            # If constant fill is selected, ask for the value
            constant_value = None
            if handling_method == "Fill with Constant":
                if pd.api.types.is_numeric_dtype(data[selected_col]):
                    constant_value = st.number_input("Enter value to fill with:", value=0)
                else:
                    constant_value = st.text_input("Enter value to fill with:", value="Unknown")
            
            # Apply button
            if st.button("Apply Missing Value Handling"):
                # Store original data for undo functionality
                if 'original_data' not in st.session_state:
                    st.session_state.original_data = data.copy()
                
                # Create a copy of the data to apply changes
                new_data = data.copy()
                
                if handling_method == "Fill with Mean":
                    if pd.api.types.is_numeric_dtype(new_data[selected_col]):
                        mean_val = new_data[selected_col].mean()
                        new_data[selected_col] = new_data[selected_col].fillna(mean_val)
                        st.success(f"Filled missing values in '{selected_col}' with Mean ({mean_val:.2f})")
                    else:
                        st.error("Cannot use Mean for non-numeric column")
                        return

                elif handling_method == "Fill with Median":
                    if pd.api.types.is_numeric_dtype(new_data[selected_col]):
                        median_val = new_data[selected_col].median()
                        new_data[selected_col] = new_data[selected_col].fillna(median_val)
                        st.success(f"Filled missing values in '{selected_col}' with Median ({median_val:.2f})")
                    else:
                        st.error("Cannot use Median for non-numeric column")
                        return

                elif handling_method == "Fill with Mode":
                    mode_val = new_data[selected_col].mode()[0]
                    new_data[selected_col] = new_data[selected_col].fillna(mode_val)
                    st.success(f"Filled missing values in '{selected_col}' with Mode ({mode_val})")

                elif handling_method == "Fill with Constant":
                    new_data[selected_col] = new_data[selected_col].fillna(constant_value)
                    st.success(f"Filled missing values in '{selected_col}' with Constant ({constant_value})")

                elif handling_method == "Drop rows with Missing Values":
                    rows_before = len(new_data)
                    new_data = new_data.dropna(subset=[selected_col])
                    rows_after = len(new_data)
                    st.success(f"Dropped {rows_before - rows_after} rows with Missing Values in '{selected_col}'")
                
                # Record the change in history
                if 'cleaning_history' not in st.session_state:
                    st.session_state.cleaning_history = []
                
                st.session_state.cleaning_history.append({
                    'operation': 'missing_values',
                    'column': selected_col,
                    'method': handling_method,
                    'constant_value': constant_value
                })
                
                # Update the data in session state
                st.session_state.data = new_data
                
                # Rerun to update the UI
                import time
                time.sleep(2)  
                st.rerun()
    
    # Outliers tab
    with tab3:
        st.header("Handle Outliers")

        # Only numeric columns can have outliers
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        
        outlier_columns = []

        for col in numeric_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Check if column has any outliers
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            if not outliers.empty:
                outlier_columns.append(col)

        # Display result
        if outlier_columns:
            st.warning(f"Outliers were detected in the following columns: {', '.join(outlier_columns)}")
        else:
            st.success("✅ No outliers detected in any numeric column!")

        if not numeric_cols:
            st.warning("No numeric columns found in the dataset.")
        else:
            # Select column for outlier detection
            selected_col = st.selectbox(
                "Select a numeric column to check for outliers:",
                numeric_cols,
                key="outlier_col_select"
            )
            
            # Calculate IQR for outlier detection
            q1 = data[selected_col].quantile(0.25)
            q3 = data[selected_col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Identify outliers
            outliers = data[(data[selected_col] < lower_bound) | (data[selected_col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_pct = outlier_count / len(data) * 100
            
            # Display outlier information
            st.markdown(f"**Column:** {selected_col}")
            st.markdown(f"**Outliers detected:** {outlier_count:,} rows ({outlier_pct:.2f}% of data)")
            st.markdown(f"**IQR range:** {lower_bound:.2f} to {upper_bound:.2f}")
            
            # Visualize distribution with outliers
            st.subheader("Distribution with Outliers Highlighted")
            
            fig = px.box(
                data, 
                y=selected_col,
                title=f"Box Plot for {selected_col}",
                color_discrete_sequence=['#0078D7']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Histogram with outlier markers
            fig2 = px.histogram(
                data, 
                x=selected_col,
                title=f"Histogram for {selected_col}",
                color_discrete_sequence=['#0078D7'],
                marginal="box",
                opacity=0.7,
            )
            
            # Add vertical lines for bounds
            fig2.add_vline(x=lower_bound, line_dash="dash", line_color="red", annotation_text="Lower Bound")
            fig2.add_vline(x=upper_bound, line_dash="dash", line_color="red", annotation_text="Upper Bound")
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Display outlier samples
            if outlier_count > 0:
                with st.expander("View Outlier Samples"):
                    st.dataframe(outliers.head(10))
            
            # AI recommendation for handling outliers
            st.subheader("AI Recommendation")
            
            if outlier_pct < 1:
                recommend_method = "Remove outliers"
                reason = "The dataset has very few outliers, removing them won't significantly affect analysis."
            elif outlier_pct < 5:
                recommend_method = "Cap outliers"
                reason = "Moderate number of outliers detected. Capping preserves data points while reducing extreme values."
            else:
                recommend_method = "Transform data"
                reason = "High percentage of outliers suggests a skewed distribution. Consider log or other transformations."
            
            st.markdown(f"**Recommended method:** {recommend_method}")
            st.markdown(f"**Reason:** {reason}")
            
            # Outlier handling options
            st.subheader("Choose handling method")
            
            handling_method = st.radio(
                "How would you like to handle outliers?",
                ["Remove Outliers", "Cap Outliers at Boundaries", "Transform Data (Log)", "Do Nothing"],
                horizontal=True
            )

            # Add explanation for "Remove Outliers"
            if handling_method == "Remove Outliers":
                st.info("This will remove rows with extreme values (Outliers) from the Dataset, making it cleaner and more consistent.")

            elif handling_method == "Cap Outliers at Boundaries":
                st.info("This will limit extreme values to a specified range, keeping all rows but reducing the impact of Outliers.")

            elif handling_method == "Transform Data (Log)":
                st.info("This will apply a Logarithmic Transformation to compress large values and reduce the effect of Outliers.")
            
            # Apply button
            if st.button("Apply Outlier Handling"):
                # Store original data for undo functionality
                if 'original_data' not in st.session_state:
                    st.session_state.original_data = data.copy()
                
                # Create a copy of the data to apply changes
                new_data = data.copy()

                if handling_method == "Remove Outliers":
                    rows_before = len(new_data)
                    new_data = new_data[(new_data[selected_col] >= lower_bound) & 
                                      (new_data[selected_col] <= upper_bound)]
                    rows_after = len(new_data)
                    st.success(f"Removed {rows_before - rows_after} outliers from '{selected_col}'")

                elif handling_method == "Cap Outliers at Boundaries":
                    new_data[selected_col] = new_data[selected_col].clip(lower=lower_bound, upper=upper_bound)
                    st.success(f"Capped outliers in '{selected_col}' to range [{lower_bound:.2f}, {upper_bound:.2f}]")

                elif handling_method == "Transform Data (Log)":
                    # Handle zero or negative values before log transform
                    min_val = new_data[selected_col].min()
                    
                    if min_val <= 0:
                        shift_value = abs(min_val) + 1  # Add a small offset
                        new_data[selected_col] = np.log(new_data[selected_col] + shift_value)
                        st.success(f"Applied Log transformation to '{selected_col}' (with shift of {shift_value:.2f})")
                    else:
                        new_data[selected_col] = np.log(new_data[selected_col])
                        st.success(f"Applied Log transformation to '{selected_col}'")

                # Preview
                st.write("Preview of new columns:")
                st.dataframe(new_data.head(5))
                
                # Record the change in history
                if 'cleaning_history' not in st.session_state:
                    st.session_state.cleaning_history = []
                
                st.session_state.cleaning_history.append({
                    'operation': 'outliers',
                    'column': selected_col,
                    'method': handling_method,
                    'bounds': [lower_bound, upper_bound]
                })
                
                # Update the data in session state
                st.session_state.data = new_data
                
                # Rerun to update the UI
                # st.rerun()
    
    # Data types tab
    with tab4:
        st.header("Fix Data Types")
        
        # Display current data types
        st.subheader("Current Data Types")
        dtypes_df = pd.DataFrame({
            'Column': data.columns,
            'Current Type': [str(dtype) for dtype in data.dtypes],
            'Sample Values': [str(data[col].dropna().head(3).tolist()) for col in data.columns]
        })
        
        st.dataframe(dtypes_df)
        
        # Add auto-inference option
        if st.checkbox("Enable automatic data type inference", value=False):
            # Custom date formats input
            custom_dates = st.text_input(
                "Custom date formats (comma-separated, optional):",
                placeholder="Example: %Y-%m-%d,%d/%m/%Y"
            )
            date_formats = custom_dates.split(',') if custom_dates else None
            
            # Confidence threshold
            threshold = st.slider(
                "Confidence threshold for type inference:",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                step=0.05
            )
            
            if st.button("Run Automatic Type Inference"):
                with st.spinner("Inferring data types..."):
                    new_df, changes = DataProcessor.enhanced_auto_infer_data_types(
                        st.session_state.data,
                        date_formats=date_formats,
                        threshold=threshold
                    )
                    
                    if changes:
                        st.session_state.data = DataProcessor.apply_inferred_types(new_df, changes)
                        st.success(f"Updated {len(changes)} column(s)")
                    else:
                        st.info("No type changes suggested")
        
        else:            
            # Select column to change type
            selected_col = st.selectbox(
                "Select a column to modify its data type:",
                data.columns,
                key="dtype_col_select"
            )
            
            # Current type
            current_type = str(data[selected_col].dtype)
            st.info(f"Current type of '{selected_col}': {current_type}")
            
            # Sample values
            st.markdown("**Sample values:**")
            st.dataframe(data[selected_col].dropna().head(5))
            
            # AI suggestion for data type
            suggested_type = DataProcessor.suggest_data_type(data, selected_col)
            
            if suggested_type != current_type:
                st.markdown("### AI Recommendation")
                st.success(f"Suggested type: **{suggested_type}**")
                
                if suggested_type == "datetime64":
                    st.markdown("This column appears to contain date/time information.")
                elif suggested_type == "category":
                    unique_count = data[selected_col].nunique()
                    st.markdown(f"This column has only {unique_count} unique values and would be more efficient as a category.")
                elif suggested_type == "float64" or suggested_type == "int64":
                    st.markdown("This column contains numeric values but may be stored as strings or objects.")
                elif suggested_type == "binary":
                    st.markdown("This column contains binary data (0s and 1s) and should be stored as a binary type.")

            # New type selection
            new_type = st.selectbox(
                "Select new data type:",
                ["int64", "float64", "string", "datetime64", "category", "boolean", "binary"],
                index=["int64", "float64", "string", "datetime64", "category", "boolean", "binary"].index(suggested_type) 
                if suggested_type in ["int64", "float64", "string", "datetime64", "category", "boolean", "binary"] else 0
            )
            
            # Display explanation for each data type
            if new_type == "int64":
                st.info("Converting to int64 will round down any decimal values.")
            elif new_type == "float64":
                st.info("Converting to float64 will preserve decimal values.")  
            elif new_type == "string":
                st.info("Converting to string will treat all values as text.")
            elif new_type == "datetime64":
                st.info("Converting to datetime64 will interpret the values as date/time.")
            elif new_type == "category":
                st.info("Converting to category will optimize memory usage for columns with few unique values.")
            elif new_type == "boolean":
                st.info("Converting to boolean will interpret values as True/False. Ensure the column contains valid boolean representations.")
            elif new_type == "binary":
                st.info("Converting to binary will interpret values as 0/1. Ensure the column contains valid binary representations.")
                
            # Additional options for datetime
            date_format = None
            if new_type == "datetime64":
                date_format = st.text_input(
                    "Specify date format (leave blank for auto-detection):",
                    placeholder="e.g., %Y-%m-%d or %d/%m/%Y"
                )
            
            # Apply button
            if st.button("Apply Type Conversion"):
                # Store original data for undo functionality
                if 'original_data' not in st.session_state:
                    st.session_state.original_data = data.copy()
                
                # Create a copy of the data to apply changes
                new_data = data.copy()
                
                try:
                    if new_type == "int64":
                        new_data[selected_col] = pd.to_numeric(new_data[selected_col], errors='coerce').fillna(0).astype('int64')
                    
                    elif new_type == "float64":
                        new_data[selected_col] = pd.to_numeric(new_data[selected_col], errors='coerce')
                    
                    elif new_type == "string":
                        new_data[selected_col] = new_data[selected_col].astype('string')
                    
                    elif new_type == "datetime64":
                        if date_format and date_format.strip():
                            new_data[selected_col] = pd.to_datetime(new_data[selected_col], format=date_format, errors='coerce')
                        else:
                            new_data[selected_col] = pd.to_datetime(new_data[selected_col], errors='coerce')
                    
                    elif new_type == "category":
                        new_data[selected_col] = new_data[selected_col].astype('category')
                    
                    elif new_type == "boolean":
                        new_data[selected_col] = new_data[selected_col].map({'True': True, 'False': False, 
                                                                        '1': True, '0': False, 
                                                                        1: True, 0: False,
                                                                        'yes': True, 'no': False,
                                                                        'Y': True, 'N': False}, na_action='ignore')
                        new_data[selected_col] = new_data[selected_col].astype('boolean')
                    
                    elif new_type == "binary":
                        new_data[selected_col] = new_data[selected_col].map({'True': 1, 'False': 0, 
                                                                        '1': 1, '0': 0, 
                                                                        1: 1, 0: 0,
                                                                        'yes': 1, 'no': 0,
                                                                        'Y': 1, 'N': 0}, na_action='ignore')
                        new_data[selected_col] = new_data[selected_col].astype('int8')

                    # Record the change in history
                    if 'cleaning_history' not in st.session_state:
                        st.session_state.cleaning_history = []
                    
                    st.session_state.cleaning_history.append({
                        'operation': 'data_type',
                        'column': selected_col,
                        'old_type': current_type,
                        'new_type': new_type,
                        'date_format': date_format
                    })
                    
                    # Update the data in session state
                    st.session_state.data = new_data
                    
                    st.success(f"Successfully converted '{selected_col}' to {new_type}")
                    
                    # Rerun to update the UI
                    import time
                    time.sleep(2)  # Optional delay for better UX
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error converting data type: {str(e)}")
        
    # Duplicates tab
    with tab5:
        st.header("Handle Duplicate Rows")
        
        # Check for duplicates
        dup_count = data.duplicated().sum()
        dup_pct = (dup_count / len(data)) * 100 if len(data) > 0 else 0
        
        # Display duplicate information
        st.metric("Duplicate Rows", f"{dup_count:,} ({dup_pct:.2f}%)")
        
        if dup_count > 0:
            # Show sample of duplicates
            st.subheader("Sample of Duplicated Rows")
            duplicated_indices = data.duplicated(keep='first')
            duplicated_data = data[duplicated_indices]
            st.dataframe(duplicated_data.head(10))
            
            # Select columns to consider for duplicate detection
            st.subheader("Customize Duplicate Detection")
            
            all_cols = st.checkbox("Consider all columns", value=True)
            selected_cols = []
            
            if not all_cols:
                selected_cols = st.multiselect(
                    "Select columns to consider for duplicate detection:",
                    data.columns.tolist()
                )
                
                if selected_cols:
                    # Recalculate duplicates based on selected columns
                    sub_dup_count = data.duplicated(subset=selected_cols).sum()
                    sub_dup_pct = (sub_dup_count / len(data)) * 100
                    st.metric("Duplicates (based on selected columns)", f"{sub_dup_count:,} ({sub_dup_pct:.2f}%)")
            
            # Handling options
            st.subheader("Choose handling method")
            
            handling_method = st.radio(
                "How would you like to handle duplicates?",
                ["Keep first occurrence", "Keep last occurrence", "Keep none (remove all duplicates)", "Do nothing"],
                horizontal=True, help = ("Keep first occurrence: , Keep last occurrence: , Keep none (remove all duplicates): , Do nothing: ")
            )
            
            # Apply button
            if st.button("Apply Duplicate Handling"):
                # Store original data for undo functionality
                if 'original_data' not in st.session_state:
                    st.session_state.original_data = data.copy()
                
                # Create a copy of the data to apply changes
                new_data = data.copy()
                
                if handling_method != "Do nothing":
                    rows_before = len(new_data)
                    
                    if handling_method == "Keep first occurrence":
                        keep_param = 'first'
                    elif handling_method == "Keep last occurrence":
                        keep_param = 'last'
                    else:  # "Keep none"
                        keep_param = False
                    
                    if all_cols:
                        new_data = new_data.drop_duplicates(keep=keep_param)
                    else:
                        new_data = new_data.drop_duplicates(subset=selected_cols, keep=keep_param)
                    
                    rows_after = len(new_data)
                    st.success(f"Removed {rows_before - rows_after} duplicate rows")
                    
                    # Record the change in history
                    if 'cleaning_history' not in st.session_state:
                        st.session_state.cleaning_history = []
                    
                    st.session_state.cleaning_history.append({
                        'operation': 'duplicates',
                        'method': handling_method,
                        'all_columns': all_cols,
                        'selected_columns': selected_cols if not all_cols else []
                    })
                    
                    # Update the data in session state
                    st.session_state.data = new_data
                    
                    # Rerun to update the UI
                    st.rerun()
    
    # Cleaning history and controls
    st.sidebar.header("Cleaning History")
    
    if 'cleaning_history' in st.session_state and st.session_state.cleaning_history:
        history = st.session_state.cleaning_history
        
        for i, operation in enumerate(history):
            # Display the operation with a cross button
            col1, col2 = st.sidebar.columns([1, 0.3])
            with col1:
                if operation['operation'] == 'missing_values':
                    st.sidebar.markdown(f"{i+1}. Fixed missing values in **{operation['column']}** using **{operation['method']}**")
                elif operation['operation'] == 'outliers':
                    st.sidebar.markdown(f"{i+1}. Handled outliers in **{operation['column']}** using **{operation['method']}**")
                elif operation['operation'] == 'data_type':
                    st.sidebar.markdown(f"{i+1}. Changed type of **{operation['column']}** from **{operation['old_type']}** to **{operation['new_type']}**")
                elif operation['operation'] == 'duplicates':
                    if operation.get('all_columns', False):
                        st.markdown(f"{i+1}. Handled duplicates using **{operation['method']}** (all columns)")
                    else:
                        st.markdown(f"{i+1}. Handled duplicates using **{operation['method']}** (columns: {', '.join(operation.get('selected_columns', []))})")
                elif operation['operation'] == 'feature_engineering':
                    # Handle feature engineering operations
                    if 'column' in operation:
                        st.markdown(f"{i+1}. Applied **{operation['type']}** to **{operation['column']}**")
                    elif 'columns' in operation:
                        st.markdown(f"{i+1}. Applied **{operation['type']}** to Numeric Columns.")
                    else:
                        st.markdown(f"{i+1}. Applied **{operation['type']}**")

            with col2:
                 # Add a cross button to delete the action
                if st.button("❌", key=f"delete_{i}"):
                    # Remove the selected action
                    st.session_state.cleaning_history.pop(i)

                    # Reapply all remaining actions
                    temp_data = st.session_state.original_data.copy()
                    for op in st.session_state.cleaning_history:   
                        # Reapply each operation (logic provided earlier)
                        if op['operation'] == 'missing_values':
                            col = op['column']
                            method = op['method']

                            if method == "Fill with mean":
                                if pd.api.types.is_numeric_dtype(temp_data[col]):
                                    mean_val = temp_data[col].mean()
                                    temp_data[col] = temp_data[col].fillna(mean_val)

                            elif method == "Fill with median":
                                if pd.api.types.is_numeric_dtype(temp_data[col]):
                                    median_val = temp_data[col].median()
                                    temp_data[col] = temp_data[col].fillna(median_val)

                            elif method == "Fill with mode":
                                mode_val = temp_data[col].mode()[0]
                                temp_data[col] = temp_data[col].fillna(mode_val)

                            elif method == "Fill with constant":
                                temp_data[col] = temp_data[col].fillna(op['constant_value'])

                            elif method == "Drop rows with missing values":
                                temp_data = temp_data.dropna(subset=[col])

                        elif op['operation'] == 'outliers':
                            col = op['column']
                            method = op['method']
                            bounds = op['bounds']

                            if method == "Remove outliers":
                                temp_data = temp_data[(temp_data[col] >= bounds[0]) & (temp_data[col] <= bounds[1])]

                            elif method == "Cap outliers at boundaries":
                                temp_data[col] = temp_data[col].clip(lower=bounds[0], upper=bounds[1])

                            elif method == "Transform data (log)":
                                min_val = temp_data[col].min()

                                if min_val <= 0:
                                    shift_value = abs(min_val) + 1
                                    temp_data[col] = np.log(temp_data[col] + shift_value)
                                else:
                                    temp_data[col] = np.log(temp_data[col])

                        elif op['operation'] == 'data_type':
                            col = op['column']
                            new_type = op['new_type']

                            if new_type == "int64":
                                temp_data[col] = pd.to_numeric(temp_data[col], errors='coerce').fillna(0).astype('int64')

                            elif new_type == "float64":
                                temp_data[col] = pd.to_numeric(temp_data[col], errors='coerce')

                            elif new_type == "string":
                                temp_data[col] = temp_data[col].astype('string')

                            elif new_type == "datetime64":
                                if op.get('date_format') and op['date_format'].strip():
                                    temp_data[col] = pd.to_datetime(temp_data[col], format=op['date_format'], errors='coerce')
                                else:
                                    temp_data[col] = pd.to_datetime(temp_data[col], errors='coerce')

                            elif new_type == "category":
                                temp_data[col] = temp_data[col].astype('category')

                            elif new_type == "boolean":
                                temp_data[col] = temp_data[col].map({'True': True, 'False': False,
                                                                    '1': True, '0': False,
                                                                    1: True, 0: False,
                                                                    'yes': True, 'no': False,
                                                                    'Y': True, 'N': False}, na_action='ignore')
                                temp_data[col] = temp_data[col].astype('boolean')

                        elif op['operation'] == 'duplicates':
                            method = op['method']

                            if method != "Do nothing":
                                if method == "Keep first occurrence":
                                    keep_param = 'first'
                                elif method == "Keep last occurrence":
                                    keep_param = 'last'
                                else:  # "Keep none"
                                    keep_param = False

                                if op['all_columns']:
                                    temp_data = temp_data.drop_duplicates(keep=keep_param)
                                else:
                                    temp_data = temp_data.drop_duplicates(subset=op['selected_columns'], keep=keep_param)

                    # Update the data with the recomputed version
                    st.session_state.data = temp_data

                    # Rerun to update the UI
                    st.rerun()

        # Undo button
        if st.sidebar.button("Undo Last Operation"):
            # Remove the last operation from history
            st.session_state.cleaning_history.pop()
            
            # If there are no more operations, revert to original data
            if not st.session_state.cleaning_history:
                st.session_state.data = st.session_state.original_data.copy()
            else:
                # Otherwise, reapply all remaining operations from scratch
                temp_data = st.session_state.original_data.copy()
                
                for op in st.session_state.cleaning_history:
                    # Reapply each operation
                    if op['operation'] == 'missing_values':
                        # Handle missing values
                        col = op['column']
                        method = op['method']
                        
                        if method == "Fill with mean":
                            if pd.api.types.is_numeric_dtype(temp_data[col]):
                                mean_val = temp_data[col].mean()
                                temp_data[col] = temp_data[col].fillna(mean_val)
                        
                        elif method == "Fill with median":
                            if pd.api.types.is_numeric_dtype(temp_data[col]):
                                median_val = temp_data[col].median()
                                temp_data[col] = temp_data[col].fillna(median_val)
                        
                        elif method == "Fill with mode":
                            mode_val = temp_data[col].mode()[0]
                            temp_data[col] = temp_data[col].fillna(mode_val)
                        
                        elif method == "Fill with constant":
                            temp_data[col] = temp_data[col].fillna(op['constant_value'])
                        
                        elif method == "Drop rows with missing values":
                            temp_data = temp_data.dropna(subset=[col])
                    
                    elif op['operation'] == 'outliers':
                        # Handle outliers
                        col = op['column']
                        method = op['method']
                        bounds = op['bounds']
                        
                        if method == "Remove outliers":
                            temp_data = temp_data[(temp_data[col] >= bounds[0]) & (temp_data[col] <= bounds[1])]
                        
                        elif method == "Cap outliers at boundaries":
                            temp_data[col] = temp_data[col].clip(lower=bounds[0], upper=bounds[1])
                        
                        elif method == "Transform data (log)":
                            min_val = temp_data[col].min()
                            
                            if min_val <= 0:
                                shift_value = abs(min_val) + 1
                                temp_data[col] = np.log(temp_data[col] + shift_value)
                            else:
                                temp_data[col] = np.log(temp_data[col])
                    
                    elif op['operation'] == 'data_type':
                        # Handle data type conversion
                        col = op['column']
                        new_type = op['new_type']
                        
                        if new_type == "int64":
                            temp_data[col] = pd.to_numeric(temp_data[col], errors='coerce').fillna(0).astype('int64')
                        
                        elif new_type == "float64":
                            temp_data[col] = pd.to_numeric(temp_data[col], errors='coerce')
                        
                        elif new_type == "string":
                            temp_data[col] = temp_data[col].astype('string')
                        
                        elif new_type == "datetime64":
                            if op.get('date_format') and op['date_format'].strip():
                                temp_data[col] = pd.to_datetime(temp_data[col], format=op['date_format'], errors='coerce')
                            else:
                                temp_data[col] = pd.to_datetime(temp_data[col], errors='coerce')
                        
                        elif new_type == "category":
                            temp_data[col] = temp_data[col].astype('category')
                        
                        elif new_type == "boolean":
                            temp_data[col] = temp_data[col].map({'True': True, 'False': False, 
                                                               '1': True, '0': False, 
                                                               1: True, 0: False,
                                                               'yes': True, 'no': False,
                                                               'Y': True, 'N': False}, na_action='ignore')
                            temp_data[col] = temp_data[col].astype('boolean')
                    
                    elif op['operation'] == 'duplicates':
                        # Handle duplicates
                        method = op['method']
                        
                        if method != "Do nothing":
                            if method == "Keep first occurrence":
                                keep_param = 'first'
                            elif method == "Keep last occurrence":
                                keep_param = 'last'
                            else:  # "Keep none"
                                keep_param = False
                            
                            if op['all_columns']:
                                temp_data = temp_data.drop_duplicates(keep=keep_param)
                            else:
                                temp_data = temp_data.drop_duplicates(subset=op['selected_columns'], keep=keep_param)
                
                # Update the data with our recomputed version
                st.session_state.data = temp_data
            
            # Rerun to update the UI
            st.rerun()
        
        # Reset button
        if st.sidebar.button("Reset All Changes"):
            st.session_state.data = st.session_state.original_data.copy()
            st.session_state.cleaning_history = []
            st.rerun()
    
    else:
        st.sidebar.info("No cleaning operations performed yet.")
    
    # Download button for cleaned data
    st.sidebar.header("Download Data")
    
    if st.session_state.data is not None:
        csv = st.session_state.data.to_csv(index=False).encode('utf-8')
        
        st.sidebar.download_button(
            label="Download Cleaned Data (CSV)",
            data=csv,
            file_name=f"cleaned_{st.session_state.filename}" if st.session_state.filename else "cleaned_data.csv",
            mime="text/csv"
        )
    
    # Feature Engineering tab
    with tab6:
        st.header("Feature Engineering")
        st.write("Transform and prepare your data for modeling.")
        
        # Add sub-tabs
        fe_tab1, fe_tab2, fe_tab3 = st.tabs(["Categorical Encoding", "Feature Scaling", "Feature Transformation"])
        
        # Categorical Encoding
        with fe_tab1:
            st.subheader("Categorical Encoding")
            st.write("Convert categorical variables into numeric format for machine learning models.")
            
            # Get categorical columns plus any object columns
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not categorical_cols:
                st.warning("No categorical columns found in the dataset.")
            else:
                # Select column
                selected_cat_col = st.selectbox(
                    "Select categorical column to encode:",
                    categorical_cols
                )
                
                # Show current values
                st.write("Current unique values:")
                value_counts = data[selected_cat_col].value_counts().reset_index()
                value_counts.columns = ['Value', 'Count']
                st.dataframe(value_counts)
                
                # Select encoding method
                encoding_method = st.radio(
                    "Select encoding method:",
                    ["One-Hot Encoding", "Label Encoding", "Ordinal Encoding"]
                )
                
                # Common parameters
                new_col_prefix = st.text_input("Output column prefix:", 
                                              value=f"{selected_cat_col}_encoded")
                
                # Default values for method-specific options
                drop_first = True  # Default for one-hot encoding
                unique_vals = []
                ordered_categories = []
                
                # Method-specific options
                if encoding_method == "One-Hot Encoding":
                    drop_first_option = st.checkbox("Drop first category (avoids multicollinearity)", value=False)
                    encoder = OneHotEncoder(sparse_output=False, drop='first' if drop_first_option else None)
                    if drop_first_option:
                        st.info("The first category is dropped to avoid multicollinearity. This is common for linear models.")
                    else:
                        st.info("All categories are included in the encoding.")
                    
                elif encoding_method == "Ordinal Encoding":
                    # Get unique values and handle missing values
                    unique_vals = data[selected_cat_col].dropna().unique().tolist()
                    
                    st.write("Drag to specify order (first = lowest value):")
                    ordered_categories = st.multiselect(
                        "Category order:",
                        options=unique_vals,
                        default=unique_vals
                    )
                    
                    if len(ordered_categories) != len(unique_vals):
                        st.warning(f"You must include all {len(unique_vals)} categories")
                
                # Apply encoding button
                if st.button("Apply Encoding"):
                    try:
                        # Create a copy of the data
                        new_data = data.copy()
                        
                        # Fill missing values with a special marker for consistent handling
                        temp_col = new_data[selected_cat_col].fillna("MISSING_VALUE")
                        
                        new_columns = []
                        if encoding_method == "One-Hot Encoding":
                            
                            # Reshape for sklearn
                            X = temp_col.values.reshape(-1, 1)
                            
                            # Check if columns with these names already exist
                            existing_cols = new_data.columns.tolist()
                            
                            # Get drop_first option from above
                            drop_first_option = drop_first
                            
                            # Initialize encoder
                            encoder = OneHotEncoder(sparse_output=False, drop=None)

                            # Transform the data
                            encoded_array = encoder.fit_transform(X)
                            
                            # Get feature names and ensure uniqueness
                            if hasattr(encoder, 'get_feature_names_out'):
                                base_features = encoder.get_feature_names_out([selected_cat_col])
                            else:
                                # Fallback for older scikit-learn versions
                                categories = encoder.categories_[0]
                                if drop_first_option:
                                    categories = categories[1:]
                                base_features = [f"{selected_cat_col}_{cat}" for cat in categories]
                            
                            # Add unique identifiers to feature names
                            feature_names = []
                            for idx, feat in enumerate(base_features):
                                feature_names.append(f"{feat}_{idx}")
                            
                            # Create a DataFrame with the encoded values
                            encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=new_data.index)
                            
                            # Concatenate with original data
                            new_data = pd.concat([new_data, encoded_df], axis=1)
                            
                            # Success message
                            st.success(f"Successfully applied One-Hot Encoding! Added {len(feature_names)} new columns.")
                            
                            # Preview
                            st.write("Preview of new columns:")
                            st.dataframe(encoded_df.head(5))
                        
                        elif encoding_method == "Label Encoding":
                            
                            # Initialize and fit encoder
                            encoder = LabelEncoder()
                            
                            # Handle missing values by replacing them with a placeholder
                            encoded_values = encoder.fit_transform(temp_col)
                            
                            # Create new column name
                            new_col_name = new_col_prefix
                            
                            # Add encoded column to dataframe
                            new_data[new_col_name] = encoded_values
                            
                            # Display mapping for reference
                            st.success(f"Successfully applied Label Encoding! Added column '{new_col_name}'")
                            
                            # Create mapping table
                            mapping = pd.DataFrame({
                                'Original Value': encoder.classes_,
                                'Encoded Value': range(len(encoder.classes_))
                            })
                            
                            st.write("Value Mapping:")
                            st.dataframe(mapping)
                            
                            # Preview
                            st.write("Preview:")
                            preview_cols = [selected_cat_col, new_col_name]
                            st.dataframe(new_data[preview_cols].head(5))
                            
                        else:  # Ordinal Encoding
                            if len(ordered_categories) != len(unique_vals):
                                st.error("You must include all categories in the ordering")
                                return
                                
                            # Create mapping from category to ordinal value
                            value_map = {cat: idx for idx, cat in enumerate(ordered_categories)}
                            
                            # Add missing value mapping if needed
                            if data[selected_cat_col].isna().any():
                                value_map["MISSING_VALUE"] = -1
                                
                            # Apply mapping
                            new_col_name = new_col_prefix
                            new_data[new_col_name] = temp_col.map(value_map)
                            
                            # Display results
                            st.success(f"Successfully applied Ordinal Encoding! Added column '{new_col_name}'")
                            
                            # Create mapping table for display
                            mapping_data = [(k if k != "MISSING_VALUE" else "<Missing>", v) 
                                           for k, v in value_map.items()]
                            mapping_df = pd.DataFrame(mapping_data, columns=['Original Value', 'Encoded Value'])
                            
                            st.write("Value Mapping:")
                            st.dataframe(mapping_df)
                            
                            # Preview
                            st.write("Preview:")
                            preview_cols = [selected_cat_col, new_col_name]
                            st.dataframe(new_data[preview_cols].head(5))
                        
                        # Show success message with clear visibility
                        st.success(f"Successfully applied {encoding_method}!")
                        
                        # Record operation in history
                        if 'cleaning_history' not in st.session_state:
                            st.session_state.cleaning_history = []
                            
                        st.session_state.cleaning_history.append({
                            'operation': 'feature_engineering',
                            'type': encoding_method.lower().replace('-', '_').replace(' ', '_'),
                            'column': selected_cat_col,
                            'new_columns': new_columns
                        })
                        
                        # Update session state with new data
                        st.session_state.data = new_data

                        # Display transformed data
                        st.subheader("Transformed Data (First 10 Rows)")
                        display_cols = [selected_cat_col] + new_columns
                        st.dataframe(st.session_state.data[display_cols].head(10))

                        st.write("New columns created:")
                        if encoding_method == "One-Hot Encoding":
                            st.dataframe(st.session_state.data.head(10))
                        else:
                            st.dataframe(st.session_state.data[[new_col_name]].head(10))

                        # Rerun to update the UI
                        # st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error applying encoding: {str(e)}")
                        
        # Feature Scaling
        with fe_tab2:
            st.subheader("Feature Scaling")
            st.write("Scale numeric features to improve model performance.")
            
            # Get numeric columns
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

            if not numeric_cols:
                st.warning("No numeric columns found in the dataset.")
            else:
                # Select columns to scale
                selected_num_cols = st.multiselect(
                    "Select numeric columns to scale:",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))]
                )
                
                if not selected_num_cols:
                    st.warning("Please select at least one column to scale.")
                else:
                    # Select scaling method
                    scaling_method = st.radio(
                        "Select scaling method:",
                        ["Min-Max Scaling", "Standard Scaling", "Robust Scaling"],
                        key="scaling_method"
                    )
                    
                    if scaling_method == "Min-Max Scaling":
                        st.write("""
                        **Min-Max Scaling** transforms features to a range between 0 and 1.
                        
                        Best for:
                        - When you need values in a bounded interval
                        - Neural networks and algorithms that expect data between 0 and 1
                        - When the distribution is not Gaussian or when the standard deviation is small
                        
                        Formula: X_scaled = (X - X_min) / (X_max - X_min)
                        """)
                        
                        # Allow custom range
                        custom_range = st.checkbox("Use custom range", value=False)
                        
                        min_val = 0
                        max_val = 1
                        
                        if custom_range:
                            col1, col2 = st.columns(2)
                            with col1:
                                min_val = st.number_input("Minimum value:", value=0.0)
                            with col2:
                                max_val = st.number_input("Maximum value:", value=1.0)
                        
                        if st.button("Apply Min-Max Scaling"):
                            try:
                                # Create a copy of the data
                                new_data = data.copy()
                                
                                for col in selected_num_cols:
                                    # Get column min and max
                                    col_min = new_data[col].min()
                                    col_max = new_data[col].max()
                                    
                                    # Check if min and max are the same (constant column)
                                    if col_min == col_max:
                                        st.warning(f"Column '{col}' has constant value ({col_min}). Scaling not applied.")
                                        continue
                                    
                                    # Apply min-max scaling
                                    new_data[f"{col}_minmax"] = (new_data[col] - col_min) / (col_max - col_min)
                                    
                                    # Scale to custom range if selected
                                    if custom_range:
                                        new_data[f"{col}_minmax"] = new_data[f"{col}_minmax"] * (max_val - min_val) + min_val
                                
                                # Record the change in history
                                if 'cleaning_history' not in st.session_state:
                                    st.session_state.cleaning_history = []
                                
                                st.session_state.cleaning_history.append({
                                    'operation': 'feature_engineering',
                                    'type': 'min_max_scaling',
                                    'columns': selected_num_cols,
                                    'min_val': min_val,
                                    'max_val': max_val
                                })
                                
                                # Update the data in session state
                                st.session_state.data = new_data
                                
                                st.success(f"Successfully applied Min-Max Scaling to {len(selected_num_cols)} columns")
                                
                                # Preview
                                st.write("Preview of scaled columns:")
                                preview_cols = []
                                for col in selected_num_cols:  # Show all columns for clarity
                                    preview_cols.extend([col, f"{col}_minmax"])
                                
                                st.dataframe(new_data[preview_cols].head())
                                
                                # Rerun to update the UI
                                # st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error applying Min-Max Scaling: {str(e)}")
                    
                    elif scaling_method == "Standard Scaling":
                        st.write("""
                        **Standard Scaling** (Z-score normalization) transforms features to have mean=0 and standard deviation=1.
                        
                        Best for:
                        - When data follows a normal distribution
                        - Algorithms that assume Gaussian distributed data (SVMs, linear/logistic regression)
                        - When outliers are minimal or have been handled
                        
                        Formula: X_scaled = (X - mean) / std_dev
                        """)
                        
                        if st.button("Apply Standard Scaling"):
                            try:
                                # Create a copy of the data
                                new_data = data.copy()
                                
                                for col in selected_num_cols:
                                    # Get column mean and std
                                    col_mean = new_data[col].mean()
                                    col_std = new_data[col].std()
                                    
                                    # Check if std is 0 (constant column)
                                    if col_std == 0:
                                        st.warning(f"Column '{col}' has zero standard deviation. Scaling not applied.")
                                        continue
                                    
                                    # Apply standard scaling
                                    new_data[f"{col}_scaled"] = (new_data[col] - col_mean) / col_std
                                
                                # Record the change in history
                                if 'cleaning_history' not in st.session_state:
                                    st.session_state.cleaning_history = []
                                
                                st.session_state.cleaning_history.append({
                                    'operation': 'feature_engineering',
                                    'type': 'standard_scaling',
                                    'columns': selected_num_cols
                                })
                                
                                # Update the data in session state
                                st.session_state.data = new_data
                                
                                st.success(f"Successfully applied Standard Scaling to {len(selected_num_cols)} columns")
                                
                                # Preview
                                st.write("Preview of scaled columns:")
                                preview_cols = []
                                for col in selected_num_cols[:3]:  # Show only first 3 for clarity
                                    preview_cols.extend([col, f"{col}_scaled"])
                                
                                st.dataframe(new_data[preview_cols].head())
                                
                                # Rerun to update the UI
                                # st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error applying Standard Scaling: {str(e)}")
                    
                    else:  # Robust Scaling
                        st.write("""
                        **Robust Scaling** uses the median and interquartile range, making it resilient to outliers.
                        
                        Best for:
                        - Datasets with many outliers
                        - When you want to preserve information about outliers
                        - When features have a high range of values
                        
                        Formula: X_scaled = (X - median) / (75th percentile - 25th percentile)
                        """)
                        
                        if st.button("Apply Robust Scaling"):
                            try:
                                # Create a copy of the data
                                new_data = data.copy()
                                
                                for col in selected_num_cols:
                                    # Get column median and IQR
                                    col_median = new_data[col].median()
                                    q1 = new_data[col].quantile(0.25)
                                    q3 = new_data[col].quantile(0.75)
                                    iqr = q3 - q1
                                    
                                    # Check if IQR is 0
                                    if iqr == 0:
                                        st.warning(f"Column '{col}' has zero IQR. Scaling not applied.")
                                        continue
                                    
                                    # Apply robust scaling
                                    new_data[f"{col}_robust"] = (new_data[col] - col_median) / iqr
                                
                                # Record the change in history
                                if 'cleaning_history' not in st.session_state:
                                    st.session_state.cleaning_history = []
                                
                                st.session_state.cleaning_history.append({
                                    'operation': 'feature_engineering',
                                    'type': 'robust_scaling',
                                    'columns': selected_num_cols
                                })
                                
                                # Update the data in session state
                                st.session_state.data = new_data
                                
                                st.success(f"Successfully applied Robust Scaling to {len(selected_num_cols)} columns")
                                
                                # Preview
                                st.write("Preview of scaled columns:")
                                preview_cols = []
                                for col in selected_num_cols[:3]:  # Show only first 3 for clarity
                                    preview_cols.extend([col, f"{col}_robust"])
                                
                                st.table(new_data[preview_cols].head())
                                
                                # Rerun to update the UI
                                # st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error applying Robust Scaling: {str(e)}")
        
        # Feature Transformation
        with fe_tab3:
            st.subheader("Feature Transformation")
            st.write("Transform features to improve model performance and address skewness.")
            
            # Get numeric columns
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            
            if not numeric_cols:
                st.warning("No numeric columns found in the dataset.")
            else:
                # Select column to transform
                selected_num_col = st.selectbox(
                    "Select numeric column to transform:",
                    numeric_cols,
                    key="transform_num_col"
                )
                
                # Show distribution of selected column
                fig = px.histogram(
                    data, 
                    x=selected_num_col,
                    title=f"Distribution of {selected_num_col}",
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Select transformation method
                transform_method = st.radio(
                    "Select transformation method:",
                    ["Log Transform", "Square Root Transform", "Box-Cox Transform", "Binning"],
                    key="transform_method"
                )
                
                if transform_method == "Log Transform":
                    st.write("""
                    **Log Transform** applies logarithm to compress high values and expand low values.
                    
                    Best for:
                    - Right-skewed distributions
                    - Data with positive values only
                    - When you need to reduce the effect of outliers
                    
                    Note: Values <= 0 will be handled by adding a constant.
                    """)
                    
                    # Check for non-positive values
                    min_val = data[selected_num_col].min()
                    
                    if min_val <= 0:
                        shift_value = abs(min_val) + 1
                        st.warning(f"Column contains non-positive values (min: {min_val}). Will add {shift_value} before log transform.")
                    
                    if st.button("Apply Log Transform"):
                        try:
                            # Create a copy of the data
                            new_data = data.copy()
                            
                            # Shift values if needed
                            if min_val <= 0:
                                shift_value = abs(min_val) + 1
                                new_data[f"{selected_num_col}_log"] = np.log(new_data[selected_num_col] + shift_value)
                            else:
                                new_data[f"{selected_num_col}_log"] = np.log(new_data[selected_num_col])
                            
                            # Record the change in history
                            if 'cleaning_history' not in st.session_state:
                                st.session_state.cleaning_history = []
                            
                            st.session_state.cleaning_history.append({
                                'operation': 'feature_engineering',
                                'type': 'log_transform',
                                'column': selected_num_col,
                                'shift': min_val <= 0,
                                'shift_value': shift_value if min_val <= 0 else 0
                            })
                            
                            # Update the data in session state
                            st.session_state.data = new_data
                            
                            st.success(f"Successfully applied Log Transform to '{selected_num_col}'")
                            
                            # Show new distribution
                            fig = px.histogram(
                                new_data, 
                                x=f"{selected_num_col}_log",
                                title=f"Distribution of {selected_num_col} (Log Transformed)",
                                marginal="box"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Rerun to update the UI
                            # st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error applying Log Transform: {str(e)}")
                
                elif transform_method == "Square Root Transform":
                    st.write("""
                    **Square Root Transform** applies square root to compress high values.
                    
                    Best for:
                    - Right-skewed distributions
                    - Count data
                    - When you need a milder transformation than log
                    
                    Note: Negative values will be handled by taking the sign and square root of absolute value.
                    """)
                    
                    # Check for negative values
                    has_negative = (data[selected_num_col] < 0).any()
                    
                    if has_negative:
                        st.warning(f"Column contains negative values. Transform will use: sign(x) * sqrt(|x|)")
                    
                    if st.button("Apply Square Root Transform"):
                        try:
                            # Create a copy of the data
                            new_data = data.copy()
                            
                            # Handle negative values if needed
                            if has_negative:
                                new_data[f"{selected_num_col}_sqrt"] = new_data[selected_num_col].apply(
                                    lambda x: np.sign(x) * np.sqrt(abs(x))
                                )
                            else:
                                new_data[f"{selected_num_col}_sqrt"] = np.sqrt(new_data[selected_num_col])
                            
                            # Record the change in history
                            if 'cleaning_history' not in st.session_state:
                                st.session_state.cleaning_history = []
                            
                            st.session_state.cleaning_history.append({
                                'operation': 'feature_engineering',
                                'type': 'sqrt_transform',
                                'column': selected_num_col,
                                'has_negative': has_negative
                            })
                            
                            # Update the data in session state
                            st.session_state.data = new_data
                            
                            st.success(f"Successfully applied Square Root Transform to '{selected_num_col}'")
                            
                            # Show new distribution
                            fig = px.histogram(
                                new_data, 
                                x=f"{selected_num_col}_sqrt",
                                title=f"Distribution of {selected_num_col} (Square Root Transformed)",
                                marginal="box"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Rerun to update the UI
                            # st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error applying Square Root Transform: {str(e)}")
                
                elif transform_method == "Box-Cox Transform":
                    st.write("""
                    **Box-Cox Transform** finds the optimal power transformation to make data more normal.
                    
                    Best for:
                    - Making data more normally distributed
                    - Linear models that require normality
                    - When you need to stabilize variance
                    
                    Note: Requires all values to be positive. Non-positive values will be shifted.
                    """)
                    
                    # Check for non-positive values
                    min_val = data[selected_num_col].min()
                    
                    if min_val <= 0:
                        shift_value = abs(min_val) + 1
                        st.warning(f"Column contains non-positive values (min: {min_val}). Will add {shift_value} before Box-Cox transform.")
                    
                    if st.button("Apply Box-Cox Transform"):
                        try:
                            # Import scipy for Box-Cox
                            from scipy import stats
                            
                            # Create a copy of the data
                            new_data = data.copy()
                            
                            # Prepare data for Box-Cox (must be positive)
                            if min_val <= 0:
                                shift_value = abs(min_val) + 1
                                transformed_data, lambda_value = stats.boxcox(new_data[selected_num_col] + shift_value)
                            else:
                                transformed_data, lambda_value = stats.boxcox(new_data[selected_num_col])
                            
                            # Add transformed column
                            new_data[f"{selected_num_col}_boxcox"] = transformed_data
                            
                            # Record the change in history
                            if 'cleaning_history' not in st.session_state:
                                st.session_state.cleaning_history = []
                            
                            st.session_state.cleaning_history.append({
                                'operation': 'feature_engineering',
                                'type': 'boxcox_transform',
                                'column': selected_num_col,
                                'lambda': lambda_value,
                                'shift': min_val <= 0,
                                'shift_value': shift_value if min_val <= 0 else 0
                            })
                            
                            # Update the data in session state
                            st.session_state.data = new_data
                            
                            st.success(f"Successfully applied Box-Cox Transform to '{selected_num_col}' (lambda = {lambda_value:.4f})")
                            
                            # Show new distribution
                            fig = px.histogram(
                                new_data, 
                                x=f"{selected_num_col}_boxcox",
                                title=f"Distribution of {selected_num_col} (Box-Cox Transformed)",
                                marginal="box"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Rerun to update the UI
                            # st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error applying Box-Cox Transform: {str(e)}")
                
                else:  # Binning
                    st.write("""
                    **Binning** groups continuous data into discrete intervals.
                    
                    Best for:
                    - Reducing the effect of minor observation errors
                    - Creating categorical features from continuous ones
                    - Handling outliers by grouping extreme values
                    
                    Advantages:
                    - Can find non-linear relationships
                    - Reduces impact of outliers
                    - Creates more robust features
                    """)
                    
                    # Binning options
                    bin_method = st.radio(
                        "Select binning method:",
                        ["Equal-width binning", "Equal-frequency binning", "Custom binning"],
                        key="bin_method"
                    )
                    
                    if bin_method == "Custom binning":
                        # Custom bin edges
                        min_val = float(data[selected_num_col].min())
                        max_val = float(data[selected_num_col].max())
                        
                        st.write(f"Column range: {min_val} to {max_val}")
                        
                        bin_edges_str = st.text_input(
                            "Enter bin edges separated by commas (e.g., 0,10,20,30):",
                            value=",".join(map(str, np.linspace(min_val, max_val, 5).round(2)))
                        )
                        
                        try:
                            bin_edges = [float(x) for x in bin_edges_str.split(",")]
                            if len(bin_edges) < 2:
                                st.error("Need at least 2 bin edges")
                                bin_edges = None
                            # Validate increasing values
                            elif any(bin_edges[i] >= bin_edges[i+1] for i in range(len(bin_edges)-1)):
                                st.error("Bin edges must be in increasing order")
                                bin_edges = None
                        except:
                            st.error("Invalid bin edges format. Use comma-separated numbers.")
                            bin_edges = None
                        
                        # Bin labels
                        default_labels = [f"Bin {i+1}" for i in range(len(bin_edges)-1)] if bin_edges else []
                        bin_labels_str = st.text_input(
                            "Enter bin labels separated by commas:",
                            value=",".join(default_labels)
                        )
                        
                        try:
                            bin_labels = bin_labels_str.split(",")
                            if len(bin_labels) != len(bin_edges) - 1:
                                st.warning(f"Number of labels ({len(bin_labels)}) doesn't match number of bins ({len(bin_edges)-1})")
                        except:
                            bin_labels = default_labels
                    else:
                        # Number of bins for equal-width or equal-frequency
                        num_bins = st.slider(
                            "Number of bins:",
                            min_value=2,
                            max_value=20,
                            value=5
                        )
                        
                        # Generate bin labels
                        bin_labels = [f"Bin {i+1}" for i in range(num_bins)]
                    
                    if st.button("Apply Binning"):
                        try:
                            # Create a copy of the data
                            new_data = data.copy()
                            
                            if bin_method == "Equal-width binning":
                                # Equal width binning
                                bins = np.linspace(
                                    data[selected_num_col].min(),
                                    data[selected_num_col].max(),
                                    num_bins + 1
                                )
                                new_data[f"{selected_num_col}_binned"] = pd.cut(
                                    new_data[selected_num_col],
                                    bins=bins,
                                    labels=bin_labels,
                                    include_lowest=True
                                )
                                
                                # Record the change in history
                                if 'cleaning_history' not in st.session_state:
                                    st.session_state.cleaning_history = []
                                
                                st.session_state.cleaning_history.append({
                                    'operation': 'feature_engineering',
                                    'type': 'equal_width_binning',
                                    'column': selected_num_col,
                                    'num_bins': num_bins,
                                    'bin_edges': bins.tolist(),
                                    'bin_labels': bin_labels
                                })
                                
                            elif bin_method == "Equal-frequency binning":
                                # Equal frequency binning (quantiles)
                                new_data[f"{selected_num_col}_binned"] = pd.qcut(
                                    new_data[selected_num_col],
                                    q=num_bins,
                                    labels=bin_labels,
                                    duplicates='drop'
                                )
                                
                                # Record the change in history
                                if 'cleaning_history' not in st.session_state:
                                    st.session_state.cleaning_history = []
                                
                                st.session_state.cleaning_history.append({
                                    'operation': 'feature_engineering',
                                    'type': 'equal_freq_binning',
                                    'column': selected_num_col,
                                    'num_bins': num_bins,
                                    'bin_labels': bin_labels
                                })
                                
                            else:  # Custom binning
                                if bin_edges:
                                    # Custom binning
                                    new_data[f"{selected_num_col}_binned"] = pd.cut(
                                        new_data[selected_num_col],
                                        bins=bin_edges,
                                        labels=bin_labels,
                                        include_lowest=True
                                    )
                                    
                                    # Record the change in history
                                    if 'cleaning_history' not in st.session_state:
                                        st.session_state.cleaning_history = []
                                    
                                    st.session_state.cleaning_history.append({
                                        'operation': 'feature_engineering',
                                        'type': 'custom_binning',
                                        'column': selected_num_col,
                                        'bin_edges': bin_edges,
                                        'bin_labels': bin_labels
                                    })
                            
                            # Update the data in session state
                            st.session_state.data = new_data
                            
                            st.success(f"Successfully applied Binning to '{selected_num_col}'")
                            
                            # Show distribution of bins
                            bin_counts = new_data[f"{selected_num_col}_binned"].value_counts().sort_index()
                            
                            fig = px.bar(
                                x=bin_counts.index,
                                y=bin_counts.values,
                                labels={'x': 'Bin', 'y': 'Count'},
                                title=f"Distribution of Binned {selected_num_col}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Preview
                            st.write("Preview:")
                            preview_cols = [selected_num_col, f"{selected_num_col}_binned"]
                            st.dataframe(new_data[preview_cols].head())
                            
                            # Rerun to update the UI
                            # st.rerun()

                        except Exception as e:
                            st.error(f"Error applying Binning: {str(e)}")
                            st.error(f"Detailed error: {type(e).__name__}")
                            import traceback
                            st.error(traceback.format_exc())



    # What's next section
    st.markdown("---")
    st.markdown("## What's Next?")
    st.info("👉 Proceed to the **Data Exploration** page to analyze your cleaned data.")
    
if __name__ == "__main__":
    main()
