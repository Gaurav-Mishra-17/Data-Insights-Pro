import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import matplotlib.pyplot as plt
import math
from scipy import stats

# Add utils to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.helpers import get_data_summary

st.set_page_config(
    page_title="DataInsights Pro",
    layout="wide"
)

def main():
    st.title("🔍 Data Exploration")
    
    # Check if data is uploaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ Please upload a Dataset in the **Data Upload** page.")
        st.stop()
    
    data = st.session_state.data
    
    # Sidebar with exploration options
    st.sidebar.header("Exploration Options")
    
    # Feature for searching and filtering columns
    search_term = st.sidebar.text_input("Search columns:", "")
    
    filtered_cols = data.columns
    if search_term:
        filtered_cols = [col for col in data.columns if search_term.lower() in col.lower()]
        if not filtered_cols:
            st.sidebar.warning(f"No columns found containing '{search_term}'")
        else:
            st.sidebar.success(f"Found {len(filtered_cols)} matching columns")
    
    # Column selection for detailed analysis
    # st.sidebar.subheader("Column Selection")
    # selected_col = st.sidebar.selectbox("Select column for detailed analysis:", filtered_cols)
    
    # Tabs for different exploration views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Column Analysis", "Correlation", "Filters & Sorting", "Advanced"
    ])
    
    # Overview tab
    with tab1:
        st.header("Dataset Overview")
        
        # Basic dataset metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{data.shape[0]:,}")
        with col2:
            st.metric("Columns", f"{data.shape[1]:,}")
        with col3:
            memory_usage = data.memory_usage(deep=True).sum()
            if memory_usage < 1024:
                memory_str = f"{memory_usage} bytes"
            elif memory_usage < 1024**2:
                memory_str = f"{memory_usage/1024:.2f} KB"
            elif memory_usage < 1024**3:
                memory_str = f"{memory_usage/(1024**2):.2f} MB"
            else:
                memory_str = f"{memory_usage/(1024**3):.2f} GB"
            st.metric("Memory Usage", memory_str)
        with col4:
            st.metric("Data Types", f"{data.dtypes.nunique()}")
        
        # Preview of the data
        st.subheader("Data Preview")
        st.dataframe(data.head(5))
        
        # Data types overview
        st.subheader("Data Types Overview")
        
        # Count columns by data type and convert to strings to ensure JSON serialization
        type_counts = pd.DataFrame({
            'Data Type': [str(dtype) for dtype in data.dtypes.value_counts().index],
            'Count': data.dtypes.value_counts().values
        })
        
        # Create a horizontal bar chart for data types
        fig = px.bar(
            type_counts, 
            y='Data Type', 
            x='Count',
            orientation='h',
            title="Column Count by Data Type",
            text='Count',
            color_discrete_sequence=['#0078D7']
        )
        fig.update_layout(xaxis_title="Number of Columns", yaxis_title="Data Type")
        st.plotly_chart(fig, width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        
        # Get numeric and non-numeric columns
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        
        if numeric_cols:
            # Display summary statistics for numeric columns
            st.markdown("#### Numeric Columns")
            numeric_stats = data[numeric_cols].describe().T
            # Add additional metrics
            if not numeric_stats.empty:
                numeric_stats['skew'] = data[numeric_cols].skew()
                numeric_stats['kurtosis'] = data[numeric_cols].kurtosis()
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
        
        # Missing values visualization
        st.subheader("Missing Values Overview")
        
        # Calculate missing values percentage for each column
        missing_data = (data.isna().sum() / len(data) * 100).sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        
        if not missing_data.empty:
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing %': missing_data.values
            })
            
            fig = px.bar(
                missing_df,
                x='Column',
                y='Missing %',
                title="Missing Values Percentage by Column",
                color='Missing %',
                color_continuous_scale=["#0078D7", "#FF0000"]
            )
            fig.update_layout(xaxis_title="Column", yaxis_title="Missing Values (%)")
            st.plotly_chart(fig, width=True)
        else:
            st.success("No missing values found in the dataset!")
    
    # Column Analysis tab
    with tab2:

        # Column selection for detailed analysis
        st.subheader("Column Selection")
        selected_col = st.selectbox("Select column for detailed analysis:", filtered_cols)

        if selected_col:
            st.header(f"Analysis of '{selected_col}'")
            
            # Column information
            col_type = str(data[selected_col].dtype)  # Convert to string for display
            unique_count = data[selected_col].nunique()
            missing_count = data[selected_col].isna().sum()
            missing_pct = (missing_count / len(data) * 100).round(2)
            
            # Display column metadata
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Data Type", col_type)
            with col2:
                st.metric("Unique Values", f"{unique_count:,}")
            with col3:
                st.metric("Missing Values", f"{missing_count:,}")
            with col4:
                st.metric("Missing %", f"{missing_pct:.2f}%")
            
            # Distribution visualization based on data type
            st.subheader("Value Distribution")
            
            if pd.api.types.is_numeric_dtype(data[selected_col]):
                # For numeric columns
                
                # Add histogram with distribution curve
                fig = px.histogram(
                    data, 
                    x=selected_col,
                    marginal="box",
                    title=f"Distribution of {selected_col}",
                    color_discrete_sequence=['#0078D7'],
                    histnorm='probability density'
                )
                
                # Add KDE curve
                try:
                    valid_data = data[selected_col].dropna()
                    if len(valid_data) > 1:
                        kde_x = np.linspace(valid_data.min(), valid_data.max(), 1000)
                        kde = stats.gaussian_kde(valid_data)
                        kde_y = kde(kde_x)
                        fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode='lines', name='Density', line=dict(color='red')))
                except Exception as e:
                    pass  # Skip KDE if it fails
                
                st.plotly_chart(fig, width=True)
                
                # Display descriptive statistics
                desc_stats = data[selected_col].describe()
                
                # Additional statistics
                skewness = data[selected_col].skew()
                kurtosis = data[selected_col].kurtosis()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if 'mean' in desc_stats:
                        st.metric("Mean", f"{desc_stats['mean']:.4f}")
                    if 'skewness' in locals():
                        st.metric("Skewness", f"{skewness:.4f}")
                with col2:
                    if '50%' in desc_stats:
                        st.metric("Median", f"{desc_stats['50%']:.4f}")
                    if 'kurtosis' in locals():
                        st.metric("Kurtosis", f"{kurtosis:.4f}")
                with col3:
                    if 'min' in desc_stats:
                        st.metric("Min", f"{desc_stats['min']:.4f}")
                    if '25%' in desc_stats:
                        st.metric("25%", f"{desc_stats['25%']:.4f}")
                with col4:
                    if 'max' in desc_stats:
                        st.metric("Max", f"{desc_stats['max']:.4f}")
                    if '75%' in desc_stats:
                        st.metric("75%", f"{desc_stats['75%']:.4f}")
                
                # Interpret skewness
                st.subheader("Distribution Interpretation")
                
                if abs(skewness) < 0.5:
                    st.success("The distribution is approximately symmetric.")
                elif 0.5 <= abs(skewness) < 1:
                    direction = "right (positive)" if skewness > 0 else "left (negative)"
                    st.info(f"The distribution is moderately skewed to the {direction} skew.")
                else:
                    direction = "right (positive)" if skewness > 0 else "left (negative)"
                    st.warning(f"The distribution is highly skewed to the {direction} skew.")

                if abs(kurtosis) < 0.5:
                    st.success("The distribution is approximately normal (mesokurtic).")
                elif kurtosis > 0:
                    st.warning("The distribution is leptokurtic — it has heavier tails and a sharper peak than a normal distribution (more prone to outliers).")
                else:
                    st.info("The distribution is platykurtic — it has lighter tails and a flatter peak than a normal distribution (less prone to outliers).")
                
                # Check for outliers
                q1 = desc_stats['25%']
                q3 = desc_stats['75%']
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = data[(data[selected_col] < lower_bound) | (data[selected_col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_pct = (outlier_count / data[selected_col].count() * 100).round(2)
                
                if outlier_count > 0:
                    st.warning(f"Detected {outlier_count:,} outliers ({outlier_pct:.2f}% of non-missing values) based on IQR method.")
                    st.markdown(f"- Lower bound: {lower_bound:.4f}")
                    st.markdown(f"- Upper bound: {upper_bound:.4f}")
                else:
                    st.success("No outliers detected using the IQR method.")
            
            elif isinstance(data[selected_col].dtype, pd.CategoricalDtype) or data[selected_col].nunique() < 20:
                # For categorical or low-cardinality columns
                
                # Value counts
                value_counts = data[selected_col].value_counts().reset_index()
                value_counts.columns = [selected_col, 'Count']
                
                # Calculate percentage
                value_counts['Percentage'] = (value_counts['Count'] / value_counts['Count'].sum() * 100).round(2)
                
                # Sort by count for better visualization
                value_counts = value_counts.sort_values('Count', ascending=False)
                
                # Show top categories if there are many
                if len(value_counts) > 10:
                    st.info(f"Showing top 10 categories out of {len(value_counts)}.")
                    display_counts = value_counts.head(10).copy()
                    # Add "Others" category
                    others_count = value_counts['Count'][10:].sum()
                    others_pct = value_counts['Percentage'][10:].sum()
                    others_row = pd.DataFrame({selected_col: ['Others'], 'Count': [others_count], 'Percentage': [others_pct]})
                    display_counts = pd.concat([display_counts, others_row], ignore_index=True)
                else:
                    display_counts = value_counts
                
                # Bar chart of value counts
                fig = px.bar(
                    display_counts,
                    y=selected_col,
                    x='Count',
                    orientation='h',
                    title=f"Value Counts for {selected_col}",
                    text='Percentage',
                    color='Count',
                    color_continuous_scale=["#C4C4C4", "#0078D7"]
                )
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig.update_layout(yaxis_title="", xaxis_title="Count")
                st.plotly_chart(fig, width=True)
                
                # Pie chart for proportion
                fig2 = px.pie(
                    display_counts,
                    names=selected_col,
                    values='Count',
                    title=f"Proportion of Categories in {selected_col}",
                    hole=0.4,
                )
                st.plotly_chart(fig2, width=True)
                
                # Display the value counts table
                st.subheader("Category Breakdown")
                st.dataframe(display_counts)
                
                # Show category insights
                st.subheader("Category Insights")
                
                if data[selected_col].nunique() == 2:
                    st.info("This appears to be a binary categorical variable, potentially suitable for classification modeling.")
                
                if (value_counts['Percentage'].iloc[0] > 80):
                    st.warning(f"The most common category '{value_counts[selected_col].iloc[0]}' accounts for {value_counts['Percentage'].iloc[0]:.2f}% of the data, indicating high class imbalance.")
                
                # Suggest encoding if appropriate
                st.markdown("**Encoding Suggestion for Machine Learning:**")
                if data[selected_col].nunique() <= 10:
                    st.success("One-hot encoding would be appropriate for this column due to the limited number of categories.")
                else:
                    st.warning("Target or frequency encoding might be more appropriate than one-hot encoding due to the high number of categories.")
            
            elif pd.api.types.is_datetime64_dtype(data[selected_col]):
                # For datetime columns
                
                # Extract time components
                valid_dates = data[selected_col].dropna()
                
                if not valid_dates.empty:
                    # Create time series plot
                    time_series_df = data.set_index(selected_col).sort_index()
                    
                    # Check if we can create a meaningful time series
                    if len(time_series_df) > 1:
                        # Check if any other numeric columns exist
                        num_cols = time_series_df.select_dtypes(include=np.number).columns
                        
                        if len(num_cols) > 0:
                            # Allow user to select a numeric column to plot against time
                            selected_y = st.selectbox(
                                "Select a numeric column to plot against time:",
                                num_cols,
                                key="timeseries_y_col"
                            )
                            
                            # Create time series plot
                            fig = px.line(
                                data_frame=time_series_df,
                                y=selected_y,
                                title=f"{selected_y} Over Time",
                                color_discrete_sequence=['#0078D7']
                            )
                            st.plotly_chart(fig, width=True)
                    
                    # Date range information
                    date_range = valid_dates.max() - valid_dates.min()
                    
                    # Display date range information
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Earliest Date", valid_dates.min().strftime('%Y-%m-%d'))
                    with col2:
                        st.metric("Latest Date", valid_dates.max().strftime('%Y-%m-%d'))
                    with col3:
                        st.metric("Date Range", f"{date_range.days} days")
                    with col4:
                        st.metric("Unique Dates", f"{valid_dates.nunique():,}")
                    
                    # Distribution by time components
                    st.subheader("Time Component Analysis")
                    
                    time_components = st.multiselect(
                        "Select time components to analyze:",
                        ["Year", "Month", "Day", "Hour", "Weekday"],
                        default=["Month", "Weekday"]
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    for i, component in enumerate(time_components):
                        if component == "Year":
                            year_counts = valid_dates.dt.year.value_counts().sort_index()
                            year_df = pd.DataFrame({
                                'Year': year_counts.index,
                                'Count': year_counts.values
                            })
                            
                            fig = px.bar(
                                year_df,
                                x='Year',
                                y='Count',
                                title="Distribution by Year",
                                color_discrete_sequence=['#0078D7']
                            )
                            
                            if i % 2 == 0:
                                with col1:
                                    st.plotly_chart(fig, width=True)
                            else:
                                with col2:
                                    st.plotly_chart(fig, width=True)
                        
                        elif component == "Month":
                            # Month names for readability
                            month_names = {
                                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                            }
                            
                            month_counts = valid_dates.dt.month.value_counts().sort_index()
                            month_df = pd.DataFrame({
                                'Month': [month_names[m] for m in month_counts.index],
                                'Count': month_counts.values,
                                'Month_num': month_counts.index
                            })
                            
                            fig = px.bar(
                                month_df.sort_values('Month_num'),
                                x='Month',
                                y='Count',
                                title="Distribution by Month",
                                color_discrete_sequence=['#0078D7']
                            )
                            
                            if i % 2 == 0:
                                with col1:
                                    st.plotly_chart(fig, width=True)
                            else:
                                with col2:
                                    st.plotly_chart(fig, width=True)
                        
                        elif component == "Day":
                            day_counts = valid_dates.dt.day.value_counts().sort_index()
                            day_df = pd.DataFrame({
                                'Day': day_counts.index,
                                'Count': day_counts.values
                            })
                            
                            fig = px.bar(
                                day_df,
                                x='Day',
                                y='Count',
                                title="Distribution by Day of Month",
                                color_discrete_sequence=['#0078D7']
                            )
                            
                            if i % 2 == 0:
                                with col1:
                                    st.plotly_chart(fig, width=True)
                            else:
                                with col2:
                                    st.plotly_chart(fig, width=True)
                        
                        elif component == "Hour":
                            if hasattr(valid_dates.dt, 'hour'):  # Check if hour component exists
                                hour_counts = valid_dates.dt.hour.value_counts().sort_index()
                                hour_df = pd.DataFrame({
                                    'Hour': hour_counts.index,
                                    'Count': hour_counts.values
                                })
                                
                                fig = px.bar(
                                    hour_df,
                                    x='Hour',
                                    y='Count',
                                    title="Distribution by Hour",
                                    color_discrete_sequence=['#0078D7']
                                )
                                
                                if i % 2 == 0:
                                    with col1:
                                        st.plotly_chart(fig, width=True)
                                else:
                                    with col2:
                                        st.plotly_chart(fig, width=True)
                        
                        elif component == "Weekday":
                            # Weekday names for readability
                            weekday_names = {
                                0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                                4: 'Friday', 5: 'Saturday', 6: 'Sunday'
                            }
                            
                            weekday_counts = valid_dates.dt.weekday.value_counts().sort_index()
                            weekday_df = pd.DataFrame({
                                'Weekday': [weekday_names[d] for d in weekday_counts.index],
                                'Count': weekday_counts.values,
                                'Weekday_num': weekday_counts.index
                            })
                            
                            fig = px.bar(
                                weekday_df.sort_values('Weekday_num'),
                                x='Weekday',
                                y='Count',
                                title="Distribution by Day of Week",
                                color_discrete_sequence=['#0078D7']
                            )
                            
                            if i % 2 == 0:
                                with col1:
                                    st.plotly_chart(fig, width=True)
                            else:
                                with col2:
                                    st.plotly_chart(fig, width=True)
                    
                    # Time-based insights
                    st.subheader("Time-Based Insights")
                    
                    # Check for seasonality, trends, etc.
                    if len(time_components) > 0:
                        if "Month" in time_components:
                            months_present = valid_dates.dt.month.nunique()
                            if months_present == 12:
                                st.info("Data covers all months, suitable for seasonal analysis.")
                            else:
                                st.warning(f"Data only covers {months_present} months out of 12, which may limit seasonal analysis.")
                        
                        if "Year" in time_components:
                            years_present = valid_dates.dt.year.nunique()
                            st.markdown(f"Data spans {years_present} years, from {valid_dates.dt.year.min()} to {valid_dates.dt.year.max()}.")
            
            elif data[selected_col].dtype == 'object' and data[selected_col].nunique() > 20:
                # For text data with high cardinality
                
                # Text length analysis
                st.subheader("Text Length Analysis")
                
                # Calculate text lengths
                text_lengths = data[selected_col].dropna().str.len()
                
                if not text_lengths.empty:
                    fig = px.histogram(
                        text_lengths,
                        title="Distribution of Text Lengths",
                        labels={'value': 'Text Length', 'count': 'Frequency'},
                        color_discrete_sequence=['#0078D7']
                    )
                    st.plotly_chart(fig, width=True)
                    
                    # Text statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Length", f"{text_lengths.mean():.1f} chars")
                    with col2:
                        st.metric("Median Length", f"{text_lengths.median():.1f} chars")
                    with col3:
                        st.metric("Min Length", f"{text_lengths.min()} chars")
                    with col4:
                        st.metric("Max Length", f"{text_lengths.max()} chars")
                    
                    # Sample values
                    st.subheader("Sample Values")
                    st.dataframe(data[selected_col].dropna().sample(5))
                    
                    # Most common values
                    st.subheader("Most Common Values")
                    common_values = data[selected_col].value_counts().head(10)
                    
                    common_df = pd.DataFrame({
                        'Value': common_values.index,
                        'Count': common_values.values,
                        'Percentage': (common_values.values / len(data) * 100).round(2)
                    })
                    
                    st.dataframe(common_df)
                else:
                    st.warning("No valid text data found in this column.")
            
            else:
                # For other data types
                st.info(f"Custom analysis for data type {data[selected_col].dtype} is not fully implemented.")
                
                # Display sample values
                st.subheader("Sample Values")
                st.write(data[selected_col].dropna().head(10).tolist())
                
                # Display most common values if applicable
                if data[selected_col].nunique() < 100:  # Only if reasonably small number of unique values
                    st.subheader("Most Common Values")
                    common_values = data[selected_col].value_counts().head(10)
                    
                    common_df = pd.DataFrame({
                        'Value': common_values.index,
                        'Count': common_values.values,
                        'Percentage': (common_values.values / len(data) * 100).round(2)
                    })
                    
                    st.dataframe(common_df)
            
            # AI-powered insights for the column
            st.subheader("AI-Generated Insights")
            
            # Generate insights based on data type and characteristics
            if pd.api.types.is_numeric_dtype(data[selected_col]):
                # Numeric insights
                insights = []
                
                # Check distribution characteristics
                mean = data[selected_col].mean()
                median = data[selected_col].median()
                skew = data[selected_col].skew()
                
                # Distribution insights
                if abs(skew) > 1:
                    direction = "right" if skew > 0 else "left"
                    insights.append(f"📊 The distribution is highly skewed to the {direction} (skewness: {skew:.2f}).")
                    
                    # Suggest transformation
                    transform = "logarithmic" if skew > 0 else "square/cube"
                    insights.append(f"💡 Consider applying a {transform} transformation before using in models.")
                
                # Check for gap between mean and median
                mean_median_diff_pct = abs(mean - median) / median * 100 if median != 0 else 0
                if mean_median_diff_pct > 10:
                    higher = "mean" if mean > median else "median"
                    insights.append(f"⚠️ The {higher} is significantly higher than the {'median' if higher == 'mean' else 'mean'}, confirming the skewed distribution.")
                
                # Outlier insights
                if outlier_pct > 0:
                    if outlier_pct > 10:
                        insights.append(f"🔍 High percentage of outliers ({outlier_pct:.2f}%) might indicate data quality issues or multiple populations.")
                    else:
                        insights.append(f"🔍 {outlier_pct:.2f}% of values are outliers. Consider handling them before modeling.")
                
                # Check for zero values if relevant
                zero_pct = (data[selected_col] == 0).mean() * 100
                if zero_pct > 10:
                    insights.append(f"0️⃣ {zero_pct:.2f}% of values are exactly zero, which may require special handling.")
                
                # Unique value ratio for numeric
                unique_ratio = data[selected_col].nunique() / len(data) * 100
                if unique_ratio < 1:
                    insights.append(f"🔄 Only {unique_ratio:.2f}% of values are unique, suggesting this might be a discrete rather than continuous variable.")
                elif unique_ratio > 90:
                    insights.append(f"🔑 {unique_ratio:.2f}% of values are unique, this could be an ID or uniquely identifying column.")
            
            elif isinstance(data[selected_col].dtype, pd.CategoricalDtype) or data[selected_col].nunique() < 20:
                # Categorical insights
                insights = []
                
                # Class imbalance
                if len(value_counts) > 0:
                    top_category_pct = value_counts['Percentage'].iloc[0]
                    if top_category_pct > 80:
                        insights.append(f"⚠️ Extreme class imbalance: the dominant category '{value_counts[selected_col].iloc[0]}' represents {top_category_pct:.2f}% of data.")
                    elif top_category_pct > 50:
                        insights.append(f"⚖️ Moderate class imbalance: the dominant category represents {top_category_pct:.2f}% of data.")
                
                # Check cardinality
                if data[selected_col].nunique() == 2:
                    insights.append("✓ Binary categorical variable, suitable for direct encoding in models.")
                elif data[selected_col].nunique() <= 10:
                    insights.append(f"📊 Low-cardinality categorical with {data[selected_col].nunique()} categories, suitable for one-hot encoding.")
                else:
                    insights.append(f"📊 Higher-cardinality categorical with {data[selected_col].nunique()} categories, consider target or frequency encoding.")
                
                # Missing data patterns
                if missing_pct > 0:
                    insights.append(f"🕳️ {missing_pct:.2f}% missing values might indicate data collection issues for this category.")
            
            elif pd.api.types.is_datetime64_dtype(data[selected_col]):
                # Datetime insights
                insights = []
                
                # Time span
                timespan = valid_dates.max() - valid_dates.min()
                insights.append(f"📅 Data spans {timespan.days} days from {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}.")
                
                # Recency
                from datetime import datetime
                today = pd.Timestamp(datetime.now().date())
                last_date = valid_dates.max()
                days_since = (today - last_date).days
                
                if days_since < 30:
                    insights.append(f"✅ Data is current, with the most recent date only {days_since} days ago.")
                elif days_since < 365:
                    insights.append(f"⚠️ Most recent data is {days_since} days old ({last_date.strftime('%Y-%m-%d')}).")
                else:
                    insights.append(f"🚨 Data may be outdated. Most recent date is {days_since} days ago ({last_date.strftime('%Y-%m-%d')}).")
                
                # Gaps
                if valid_dates.nunique() < 0.7 * (valid_dates.max() - valid_dates.min()).days:
                    insights.append("🕳️ There appear to be significant gaps in the time series data.")
                
                # Feature engineering suggestions
                insights.append("💡 Consider extracting day of week, month, quarter, and year as features for modeling.")
                
            else:
                # General insights for other types
                insights = []
                unique_ratio = data[selected_col].nunique() / len(data) * 100
                
                if unique_ratio > 90:
                    insights.append(f"🔑 This column has {unique_ratio:.2f}% unique values, suggesting it might be an identifier or key.")
                
                if missing_pct > 20:
                    insights.append(f"⚠️ High percentage of missing values ({missing_pct:.2f}%) may limit usefulness for analysis.")
            
            # Display insights
            if insights:
                for insight in insights:
                    st.markdown(insight)
            else:
                st.info("No specific insights generated for this column.")
    
    # Correlation tab
    with tab3:
        st.header("Correlation Analysis")
        
        # Get all numeric columns for correlation
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("At least two numeric columns are required for correlation analysis.")
        else:
            # Correlation method selection
            corr_method = st.radio(
                "Select correlation method:",
                ["Pearson", "Spearman", "Kendall"],
                horizontal=True,
                help="Pearson measures linear correlation, Spearman and Kendall measure monotonic relationships"
            )
            
            # Select columns for correlation
            if len(numeric_cols) > 15:
                st.warning(f"Found {len(numeric_cols)} numeric columns. Showing correlation for top 15 by default.")
                selected_corr_cols = st.multiselect(
                    "Select columns for correlation analysis:",
                    options=numeric_cols,
                    default=numeric_cols[:15]
                )
            else:
                selected_corr_cols = st.multiselect(
                    "Select columns for correlation analysis:",
                    options=numeric_cols,
                    default=numeric_cols
                )
            
            if len(selected_corr_cols) < 2:
                st.warning("Please select at least two columns for correlation analysis.")
            else:
                # Calculate correlation matrix
                corr_matrix = data[selected_corr_cols].corr(method=corr_method.lower())
                
                # Correlation heatmap
                st.subheader("Correlation Heatmap")
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    aspect="auto"
                )
                fig.update_layout(
                    width=800,
                    height=800 if len(selected_corr_cols) > 10 else 600
                )
                st.plotly_chart(fig, width=True)
                
                # Top correlations table
                st.subheader("Top Correlations")
                
                # Get the upper triangle of the correlation matrix
                corr_upper = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                ).stack().reset_index()
                corr_upper.columns = ['Variable 1', 'Variable 2', 'Correlation']
                
                # Sort by absolute correlation and get top 15
                corr_upper['Abs Correlation'] = corr_upper['Correlation'].abs()
                corr_upper = corr_upper.sort_values('Abs Correlation', ascending=False).head(15)
                corr_upper = corr_upper.drop('Abs Correlation', axis=1)
                
                # Add correlation strength description
                def correlation_strength(corr):
                    abs_corr = abs(corr)
                    if abs_corr >= 0.8:
                        return "Very Strong"
                    elif abs_corr >= 0.6:
                        return "Strong"
                    elif abs_corr >= 0.4:
                        return "Moderate"
                    elif abs_corr >= 0.2:
                        return "Weak"
                    else:
                        return "Very Weak"
                
                corr_upper['Strength'] = corr_upper['Correlation'].apply(correlation_strength)
                
                # Add direction
                corr_upper['Direction'] = corr_upper['Correlation'].apply(
                    lambda x: "Positive" if x > 0 else "Negative"
                )
                
                st.dataframe(corr_upper)
                
                # Scatter plot for selected pair
                st.subheader("Correlation Scatter Plot")
                
                if len(selected_corr_cols) >= 2:
                    # Default to the highest correlated pair
                    default_var1 = corr_upper['Variable 1'].iloc[0] if not corr_upper.empty else selected_corr_cols[0]
                    default_var2 = corr_upper['Variable 2'].iloc[0] if not corr_upper.empty else selected_corr_cols[1]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        var1 = st.selectbox("Select first variable:", selected_corr_cols, index=selected_corr_cols.index(default_var1))
                    
                    with col2:
                        var2 = st.selectbox("Select second variable:", selected_corr_cols, index=selected_corr_cols.index(default_var2))
                    
                    # Display the correlation value between selected variables
                    correlation = corr_matrix.loc[var1, var2]
                    strength = correlation_strength(correlation)
                    direction = "positive" if correlation > 0 else "negative"
                    
                    st.metric(f"Correlation ({corr_method})", f"{correlation:.4f}", 
                              f"{strength} {direction} correlation")
                    
                    # Create scatter plot
                    fig = px.scatter(
                        data, 
                        x=var1, 
                        y=var2,
                        trendline="ols" if corr_method == "Pearson" else None,
                        title=f"Scatter Plot: {var1} vs {var2}",
                        color_discrete_sequence=['#0078D7']
                    )
                    st.plotly_chart(fig, width=True)
                    
                    # Interpretation of relationship
                    st.markdown("### Interpretation")
                    
                    if abs(correlation) < 0.2:
                        st.info(f"There is a very weak {direction} relationship between these variables.")
                    elif abs(correlation) < 0.4:
                        st.info(f"There is a weak {direction} relationship between these variables.")
                    elif abs(correlation) < 0.6:
                        st.success(f"There is a moderate {direction} relationship between these variables.")
                    elif abs(correlation) < 0.8:
                        st.success(f"There is a strong {direction} relationship between these variables.")
                    else:
                        st.success(f"There is a very strong {direction} relationship between these variables.")
                    
                    relationship_strength = 'these variables move closely together' if abs(correlation) > 0.6 else 'there is some relationship, but it is not extremely strong'
                    measurement_type = 'linear relationships' if corr_method == 'Pearson' else 'monotonic relationships (whether variables increase/decrease together, not necessarily linearly)'
                    
                    st.markdown(f"""
                    **What this means:**
                    
                    - A {direction} correlation of {correlation:.4f} means that as {var1} {'increases' if correlation > 0 else 'decreases'}, {var2} tends to {'increase' if correlation > 0 else 'decrease'} as well.
                    - The strength of this relationship is **{strength.lower()}**, which means {relationship_strength}.
                    - {corr_method} correlation specifically measures {measurement_type}.
                    """)
                    
                    # Add caution about correlation
                    st.warning("**Remember:** Correlation does not imply causation. A strong correlation between two variables doesn't mean one causes the other.")
            
            # AI-powered correlation insights
            if len(selected_corr_cols) >= 2:
                st.subheader("AI-Generated Correlation Insights")
                
                # Get strong correlations
                corr_insights = []
                
                # Look for multicollinearity - use the correlation strength instead of Abs Correlation
                high_corr_pairs = corr_upper[corr_upper['Strength'] == "Very Strong"]
                if not high_corr_pairs.empty:
                    corr_insights.append("🔍 **Potential Multicollinearity Detected:**")
                    for _, row in high_corr_pairs.iterrows():
                        corr_insights.append(f"- {row['Variable 1']} and {row['Variable 2']} are very strongly correlated ({row['Correlation']:.3f}), which may cause issues in regression models.")
                    corr_insights.append("💡 Consider removing one of the variables from each highly correlated pair.")
                
                # Check for isolation
                poorly_correlated = []
                for col in selected_corr_cols:
                    # Get max absolute correlation with other columns
                    max_corr = corr_matrix[col].drop(col).abs().max()
                    if max_corr < 0.2:
                        poorly_correlated.append((col, max_corr))
                
                if poorly_correlated:
                    corr_insights.append("🧩 **Isolated Variables:**")
                    for col, max_corr in poorly_correlated:
                        corr_insights.append(f"- {col} has weak correlations with all other variables (max: {max_corr:.3f}), suggesting it contains unique information.")
                
                # Identify potential target variables for predictive modeling
                if len(selected_corr_cols) > 5:
                    corr_insights.append("🎯 **Potential Predictive Relationships:**")
                    
                    # For each variable, count how many moderate-to-strong correlations it has
                    correlation_counts = {}
                    for col in selected_corr_cols:
                        strong_corrs = sum((corr_matrix[col].drop(col).abs() > 0.4))
                        correlation_counts[col] = strong_corrs
                    
                    # Sort by number of strong correlations
                    sorted_counts = sorted(correlation_counts.items(), key=lambda x: x[1], reverse=True)
                    
                    # Report top 3 most connected variables
                    for col, count in sorted_counts[:3]:
                        if count > 0:
                            corr_insights.append(f"- {col} has moderate-to-strong relationships with {count} other variables, making it a potential target for predictive modeling.")
                
                # Display insights
                if corr_insights:
                    for insight in corr_insights:
                        st.markdown(insight)
                else:
                    st.info("No significant correlation patterns detected.")
    
    # Filters & Sorting tab
    with tab4:
        st.header("Interactive Data Filtering & Sorting")
        
        col1, col2 = st.columns(2)
        
        # Filtering section
        with col1:
            st.subheader("Filter Data")
            
            # Select columns for filtering
            filter_cols = st.multiselect(
                "Select columns to filter by:",
                data.columns.tolist(),
                default=[]
            )
            
            # Apply filters based on column types
            filtered_data = data.copy()
            
            if filter_cols:
                for col in filter_cols:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        # Numeric filter - slider for min/max
                        min_val = float(data[col].min())
                        max_val = float(data[col].max())
                        
                        # Use a slider for numeric columns
                        filter_range = st.slider(
                            f"Filter range for {col}:",
                            min_val, max_val,
                            (min_val, max_val)
                        )
                        
                        filtered_data = filtered_data[(filtered_data[col] >= filter_range[0]) & 
                                                      (filtered_data[col] <= filter_range[1])]
                    
                    elif isinstance(data[col].dtype, pd.CategoricalDtype) or data[col].nunique() < 20:
                        # Categorical filter - multiselect for values
                        cat_values = data[col].dropna().unique().tolist()
                        selected_cats = st.multiselect(
                            f"Select values for {col}:",
                            cat_values,
                            default=cat_values
                        )
                        
                        if selected_cats:
                            filtered_data = filtered_data[filtered_data[col].isin(selected_cats)]
                    
                    elif pd.api.types.is_datetime64_dtype(data[col]):
                        # Date filter - date range
                        min_date = data[col].min().date()
                        max_date = data[col].max().date()
                        
                        date_range = st.date_input(
                            f"Select date range for {col}:",
                            [min_date, max_date]
                        )
                        
                        if len(date_range) == 2:
                            filtered_data = filtered_data[(filtered_data[col].dt.date >= date_range[0]) &
                                                         (filtered_data[col].dt.date <= date_range[1])]
                    
                    else:
                        # Text search for other types
                        search_text = st.text_input(f"Search text in {col}:")
                        
                        if search_text:
                            filtered_data = filtered_data[filtered_data[col].astype(str).str.contains(search_text, case=False, na=False)]
            
            # Display filter results summary
            st.metric("Filtered Rows", f"{len(filtered_data):,} / {len(data):,}", 
                     f"{len(filtered_data)/len(data)*100:.1f}% of total")
        
        # Sorting section
        with col2:
            st.subheader("Sort Data")
            
            # Sort column selection
            sort_col = st.selectbox(
                "Select column to sort by:",
                data.columns.tolist()
            )
            
            # Sort direction
            sort_order = st.radio(
                "Sort order:",
                ["Ascending", "Descending"],
                horizontal=True
            )
            
            ascending = sort_order == "Ascending"
            
            # Apply sorting
            if sort_col:
                filtered_data = filtered_data.sort_values(by=sort_col, ascending=ascending)
        
        if len(filtered_data) > 0:
            st.info(f"Showing first 10 rows of {len(filtered_data):,} matching rows.")
        
        # Display filtered and sorted data
        st.subheader("Filtered & Sorted Data")
        st.dataframe(filtered_data.head(10))
        
        # Download filtered data button
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv,
            file_name="filtered_data.csv",
            mime="text/csv"
        )
        
        # Filtered data statistics
        if len(filtered_data) > 0:
            with st.expander("View Summary Statistics of Filtered Data"):
                # Summary stats for numeric columns in filtered data
                numeric_cols = filtered_data.select_dtypes(include=np.number).columns.tolist()
                
                if numeric_cols:
                    st.markdown("#### Numeric Columns")
                    st.dataframe(filtered_data[numeric_cols].describe())
    
    # Advanced analysis tab
    with tab5:
        st.header("Advanced Analysis")
        
        # Options for advanced analysis
        analysis_type = st.selectbox(
            "Select analysis type:",
            ["Distribution Comparison", "Group-by Analysis", "Pivot Table", "Custom Query"]
        )
        
        if analysis_type == "Distribution Comparison":
            st.subheader("Compare Distributions")
            
            # Select numeric column for comparison
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            
            if not numeric_cols:
                st.warning("No numeric columns available for comparison.")
            else:
                # Select column to compare
                compare_col = st.selectbox(
                    "Select numeric column to compare:",
                    numeric_cols,
                    key="compare_col"
                )
                
                # Select category column to group by
                cat_cols = [col for col in data.columns if data[col].nunique() < 20]
                
                if not cat_cols:
                    st.warning("No suitable category columns found for grouping.")
                else:
                    group_col = st.selectbox(
                        "Select column to group by:",
                        cat_cols,
                        key="group_col"
                    )
                    
                    # Get unique categories (limit to top 5 if many)
                    categories = data[group_col].value_counts().head(5).index.tolist()
                    
                    # Allow user to select specific categories
                    selected_cats = st.multiselect(
                        f"Select categories to compare (top 5 shown):",
                        categories,
                        default=categories[:min(3, len(categories))]
                    )
                    
                    if selected_cats:
                        # Create distribution plot
                        fig = go.Figure()
                        
                        for cat in selected_cats:
                            cat_data = data[data[group_col] == cat][compare_col].dropna()
                            
                            if len(cat_data) > 0:
                                # Add histogram trace for this category
                                fig.add_trace(go.Histogram(
                                    x=cat_data,
                                    name=str(cat),
                                    opacity=0.7,
                                    histnorm='probability density'
                                ))
                        
                        fig.update_layout(
                            title=f"Distribution of {compare_col} by {group_col}",
                            barmode='overlay',
                            xaxis_title=compare_col,
                            yaxis_title="Probability Density"
                        )
                        
                        st.plotly_chart(fig, width=True)
                        
                        # Statistical comparison
                        st.subheader("Statistical Comparison")
                        
                        # Create summary table
                        summary_data = []
                        
                        for cat in selected_cats:
                            cat_data = data[data[group_col] == cat][compare_col].dropna()
                            
                            if len(cat_data) > 0:
                                summary_data.append({
                                    'Category': cat,
                                    'Count': len(cat_data),
                                    'Mean': cat_data.mean(),
                                    'Median': cat_data.median(),
                                    'Std Dev': cat_data.std(),
                                    'Min': cat_data.min(),
                                    'Max': cat_data.max()
                                })
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df)
                            
                            # Add box plot comparison
                            fig2 = px.box(
                                data[data[group_col].isin(selected_cats)], 
                                x=group_col, 
                                y=compare_col,
                                title=f"Box Plot Comparison of {compare_col} by {group_col}",
                                color=group_col
                            )
                            st.plotly_chart(fig2, width=True)
        
        elif analysis_type == "Group-by Analysis":
            st.subheader("Group-by Analysis")
            
            # Select column to group by
            group_cols = [col for col in data.columns if data[col].nunique() < 100]
            
            if not group_cols:
                st.warning("No suitable columns found for grouping (with fewer than 100 unique values).")
            else:
                group_col = st.selectbox(
                    "Select column to group by:",
                    group_cols,
                    key="groupby_col"
                )
                
                # Select columns to aggregate
                agg_cols = st.multiselect(
                    "Select columns to aggregate:",
                    [col for col in data.select_dtypes(include=np.number).columns 
                     if col != group_col],
                    default=[]
                )
                
                if agg_cols:
                    # Select aggregation functions
                    agg_funcs = st.multiselect(
                        "Select aggregation functions:",
                        ["Count", "Sum", "Mean", "Median", "Min", "Max", "Std Dev"],
                        default=["Count", "Mean"]
                    )
                    
                    # Map friendly names to pandas aggregation functions
                    agg_map = {
                        "Count": "count",
                        "Sum": "sum",
                        "Mean": "mean",
                        "Median": "median",
                        "Min": "min",
                        "Max": "max",
                        "Std Dev": "std"
                    }
                    
                    # Create aggregation dictionary
                    agg_dict = {col: [agg_map[func] for func in agg_funcs] for col in agg_cols}
                    
                    # Perform groupby operation
                    grouped_data = data.groupby(group_col).agg(agg_dict)
                    
                    # Flatten multi-level column index for readability
                    grouped_data.columns = ['_'.join(col).strip() for col in grouped_data.columns.values]
                    
                    # Reset index for better display
                    grouped_data = grouped_data.reset_index()
                    
                    # Sort options
                    sort_by = st.selectbox(
                        "Sort results by:",
                        ["Group (alphabetical)"] + [f"{col}_{func}" for col in agg_cols for func in [agg_map[f] for f in agg_funcs]]
                    )
                    
                    sort_order = st.radio(
                        "Sort order:",
                        ["Ascending", "Descending"],
                        horizontal=True,
                        key="groupby_sort_order"
                    )
                    
                    # Apply sorting
                    if sort_by == "Group (alphabetical)":
                        grouped_data = grouped_data.sort_values(by=group_col, ascending=(sort_order == "Ascending"))
                    else:
                        grouped_data = grouped_data.sort_values(by=sort_by, ascending=(sort_order == "Ascending"))
                    
                    # Display grouped data
                    st.dataframe(grouped_data)
                    
                    # Visualization of grouped data
                    st.subheader("Visualization")
                    
                    # Limit to top N groups for readability if many groups
                    if len(grouped_data) > 15:
                        st.info(f"Showing visualization for top 15 groups out of {len(grouped_data)}.")
                        if sort_by != "Group (alphabetical)":
                            viz_data = grouped_data.head(15)
                        else:
                            # If sorted alphabetically, sort by count for visualization
                            count_col = f"{agg_cols[0]}_count" if "Count" in agg_funcs else f"{agg_cols[0]}_mean"
                            viz_data = grouped_data.sort_values(by=count_col, ascending=False).head(15)
                    else:
                        viz_data = grouped_data
                    
                    # Select visualization type
                    viz_type = st.radio(
                        "Visualization type:",
                        ["Bar Chart", "Line Chart", "Pie Chart"],
                        horizontal=True
                    )
                    
                    # Select metric to visualize
                    viz_col = st.selectbox(
                        "Select metric to visualize:",
                        [f"{col}_{func}" for col in agg_cols for func in [agg_map[f] for f in agg_funcs]]
                    )
                    
                    # Create visualization
                    if viz_type == "Bar Chart":
                        fig = px.bar(
                            viz_data,
                            x=group_col,
                            y=viz_col,
                            title=f"{viz_col} by {group_col}",
                            color_discrete_sequence=['#0078D7']
                        )
                        st.plotly_chart(fig, width=True)
                    
                    elif viz_type == "Line Chart":
                        fig = px.line(
                            viz_data,
                            x=group_col,
                            y=viz_col,
                            title=f"{viz_col} by {group_col}",
                            markers=True,
                            color_discrete_sequence=['#0078D7']
                        )
                        st.plotly_chart(fig, width=True)
                    
                    elif viz_type == "Pie Chart":
                        fig = px.pie(
                            viz_data,
                            names=group_col,
                            values=viz_col,
                            title=f"{viz_col} by {group_col}"
                        )
                        st.plotly_chart(fig, width=True)
                
                else:
                    st.info("Please select at least one column to aggregate.")
        
        elif analysis_type == "Pivot Table":
            st.subheader("Pivot Table Analysis")
            
            # Select columns for pivot
            cols_with_low_cardinality = [col for col in data.columns if data[col].nunique() < 50]
            
            if len(cols_with_low_cardinality) < 2:
                st.warning("Need at least two columns with reasonable cardinality (fewer than 50 unique values) for pivot analysis.")
            else:
                # Select index (rows)
                index_col = st.selectbox(
                    "Select row index column:",
                    cols_with_low_cardinality,
                    key="pivot_index"
                )
                
                # Select columns
                columns_col = st.selectbox(
                    "Select column headers:",
                    [col for col in cols_with_low_cardinality if col != index_col],
                    key="pivot_columns"
                )
                
                # Select values
                numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
                
                if not numeric_cols:
                    st.warning("No numeric columns available for pivot values.")
                else:
                    values_col = st.selectbox(
                        "Select values to aggregate:",
                        numeric_cols,
                        key="pivot_values"
                    )
                    
                    # Select aggregation function
                    agg_func = st.selectbox(
                        "Select aggregation function:",
                        ["Sum", "Mean", "Count", "Median", "Min", "Max"],
                        key="pivot_aggfunc"
                    )
                    
                    # Map to pandas aggregation function
                    agg_map = {
                        "Sum": np.sum,
                        "Mean": np.mean,
                        "Count": "count",
                        "Median": np.median,
                        "Min": np.min,
                        "Max": np.max
                    }
                    
                    # Create pivot table
                    pivot = pd.pivot_table(
                        data,
                        values=values_col,
                        index=index_col,
                        columns=columns_col,
                        aggfunc=agg_map[agg_func],
                        fill_value=0
                    )
                    
                    # Display pivot table
                    st.dataframe(pivot)
                    
                    # Heatmap visualization of pivot table
                    st.subheader("Pivot Table Heatmap")
                    
                    # If pivot is large, limit size for visualization
                    if pivot.shape[0] > 15 or pivot.shape[1] > 15:
                        st.info(f"Pivot table is large ({pivot.shape[0]}×{pivot.shape[1]}). Showing heatmap for top 15 rows and columns.")
                        # Get top rows and columns by sum
                        top_rows = pivot.sum(axis=1).sort_values(ascending=False).head(15).index
                        top_cols = pivot.sum(axis=0).sort_values(ascending=False).head(15).index
                        viz_pivot = pivot.loc[top_rows, top_cols]
                    else:
                        viz_pivot = pivot
                    
                    # Create heatmap
                    fig = px.imshow(
                        viz_pivot,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="Viridis"
                    )
                    fig.update_layout(
                        title=f"Pivot Table Heatmap: {agg_func} of {values_col} by {index_col} and {columns_col}"
                    )
                    st.plotly_chart(fig, width=True)
                    
                    # Marginal analysis
                    with st.expander("Row and Column Totals"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Row Totals")
                            row_totals = pivot.sum(axis=1).sort_values(ascending=False)
                            
                            # Bar chart of row totals
                            if len(row_totals) > 15:
                                st.info(f"Showing top 15 out of {len(row_totals)} rows.")
                                row_totals = row_totals.head(15)
                            
                            fig = px.bar(
                                x=row_totals.index,
                                y=row_totals.values,
                                title=f"Row Totals ({index_col})",
                                labels={"x": index_col, "y": f"Total {values_col}"},
                                color_discrete_sequence=['#0078D7']
                            )
                            st.plotly_chart(fig, width=True)
                        
                        with col2:
                            st.subheader("Column Totals")
                            col_totals = pivot.sum(axis=0).sort_values(ascending=False)
                            
                            # Bar chart of column totals
                            if len(col_totals) > 15:
                                st.info(f"Showing top 15 out of {len(col_totals)} columns.")
                                col_totals = col_totals.head(15)
                            
                            fig = px.bar(
                                x=col_totals.index,
                                y=col_totals.values,
                                title=f"Column Totals ({columns_col})",
                                labels={"x": columns_col, "y": f"Total {values_col}"},
                                color_discrete_sequence=['#0078D7']
                            )
                            st.plotly_chart(fig, width=True)
        
        elif analysis_type == "Custom Query":
            st.subheader("Custom Data Query")
            
            st.markdown("""
            This feature allows you to run custom analysis expressions on your data.
            Enter Python expressions to create calculated columns or filter data.
            
            **Examples:**
            - `data['price'] * data['quantity']` - Multiply two columns
            - `data['age'].mean()` - Calculate mean of a column
            - `data[data['price'] > 100]` - Filter rows where price is greater than 100
            
            **Note:** Use `data` to refer to your dataset in expressions.
            """)
            
            # Expression input
            query_expression = st.text_area(
                "Enter custom Python expression:",
                height=100,
                placeholder="data['column_a'] / data['column_b']"
            )
            
            if query_expression and st.button("Run Query"):
                try:
                    # Execute the expression
                    result = eval(query_expression)
                    
                    # Display results based on type
                    if isinstance(result, pd.DataFrame):
                        st.subheader("Result DataFrame")
                        st.write(f"Shape: {result.shape[0]} rows × {result.shape[1]} columns")
                        st.dataframe(result.head(100))

                        if len(result) > 100:
                            st.info(f"Showing first 100 rows of {len(result):,} total rows.")
                    
                    elif isinstance(result, pd.Series):
                        st.subheader("Result Series")
                        
                        # If it's a boolean mask, show count of True values
                        if result.dtype == bool:
                            true_count = result.sum()
                            st.metric("Matching rows", f"{true_count:,}/{len(result):,}", 
                                     f"{true_count/len(result)*100:.1f}%")
                        
                        # Otherwise show the series
                        st.dataframe(result)
                        
                        # Try to visualize if it's numeric
                        if pd.api.types.is_numeric_dtype(result):
                            st.subheader("Visualization")
                            fig = px.histogram(
                                result,
                                title="Distribution of Result",
                                color_discrete_sequence=['#0078D7']
                            )
                            st.plotly_chart(fig, width=True)
                    
                    else:
                        # For scalars and other types
                        st.subheader("Result")
                        st.json({"result": result})
                
                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")
                    st.markdown("Please check your expression and try again.")
    
    # What's next section
    st.markdown("---")
    st.markdown("## What's Next?")
    st.info("👉 Proceed to the **Visualization** page to create interactive charts and dashboards.")

if __name__ == "__main__":
    main()
