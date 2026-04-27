import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import re
import json
from datetime import datetime, timedelta
import random
from dotenv import load_dotenv
load_dotenv()

# Add utils to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.nlp_processor import NLPProcessor
from utils.data_processor import DataProcessor
from utils.visualization import Visualization
from utils.helpers import get_data_summary, generate_sample_query, detect_potential_target, suggest_feature_engineering
from utils.gemini_ai_helper import generate_insights, analyze_patterns, explain_anomalies, advanced_query_analysis

st.set_page_config(
    page_title="DataInsights Pro",
    layout="wide"
)

def main():
    st.title("🧠 AI-Powered Insights")
    
    # Check if data is uploaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ Please upload a Dataset in the **Data Upload** page.")
        st.stop()
    
    data = st.session_state.data
    
    # Sidebar options
    # st.sidebar.header("AI Options")
    
    # Tabs for different insight modes
    tab1, tab2, tab3, tab4 = st.tabs([
        "Natural Language Queries", 
        "Automated Insights", 
        "Anomaly Detection",
        "Forecasting"
    ])
    
    # Natural Language Queries Tab
    with tab1:
        st.header("Ask Questions About Your Data")
        st.markdown("""
        Ask questions in natural language, and our AI will analyze your data to provide answers.
        
        **Examples:**
        - "Show me the top 5 rows by [column]"
        - "What's the average [column]?"
        - "How does [column1] relate to [column2]?"
        - "Find trends in [column] over time"
        """)
        
        # Generate sample query based on dataset
        sample_query = generate_sample_query(data)
        
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder=sample_query
        )
        
        # Process the query when submitted
        if st.button("Get Answer", key="process_query") or query:
            if query:
                with st.spinner("Analyzing your question..."):
                    # Use NLPProcessor to interpret the query
                    # processor = NLPProcessor()
                    result = advanced_query_analysis(data, query)
                        
                    # Display result
                    # ...existing code...
                    if result:
                        # Show interpretation and explanations
                        if "interpretation" in result:
                            st.success(f"**I understood:** {result['interpretation']}")
                        if "analytical_approach" in result:
                            st.info(f"**Analytical Approach:** {result['analytical_approach']}")
                        if "additional_insights" in result:
                            st.info(f"**Additional Insights:** {result['additional_insights']}")
                        if "relevant_columns" in result:
                            st.info(f"**Relevant Columns:** {', '.join(result['relevant_columns'])}")

                        # Execute and display chart/result if python_code is present
                        if "python_code" in result and result["python_code"]:
                            st.code(result["python_code"], language="python")
                            try:
                                # Safe local namespace for exec
                                local_vars = {"df": data, "pd": pd, "np": np, "st": st, "px": px, "go": go, "plt": None}
                                exec(result["python_code"], {}, local_vars)
                            except Exception as e:
                                st.error(f"Error executing AI-generated code: {e}")

                        # If not possible, show reason
                        elif "reason_not_possible" in result:
                            st.warning(result["reason_not_possible"])
                        elif "error" in result:
                            st.error(result["error"])
                        else:
                            st.info("AI did not return a result or code.")
                    else:
                        st.error("No response from AI.")

                    #if 'data_result' in result and result['data_result'] is not None:
                    #    if isinstance(result['data_result'], pd.DataFrame):
                    #        st.subheader("Results")
                    #        st.dataframe(result['data_result'])
                    #    elif isinstance(result['data_result'], (float, int)):
                    #        st.metric("Result", f"{result['data_result']:.4f}")
                    #    else:
                    #        st.markdown(f"**Result:** {result['data_result']}")
                    
                    # Display visualization if available
                    if result and 'visualization' in result and result['visualization'] is not None:
                        st.subheader("Visualization")
                        st.plotly_chart(result['visualization'], width=True)
                    
                    # Display explanation
                    if result and 'explanation' in result and result['explanation']:
                        with st.expander("Explanation"):
                            st.markdown(result['explanation'])
                    elif result and 'error' in result and result['error']:
                        st.error(f"AI Error: {result['error']}")
                    else:
                        st.error("I couldn't understand or process your question. Please try rephrasing it or use one of the example queries.")
            else:
                st.info("Please enter a question about your data.")
    
        # Suggested queries based on the dataset
        st.subheader("Suggested Questions")
        
        # Generate suggested queries
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        suggested_queries = []
        
        # Add distribution questions for numeric columns
        for col in numeric_cols[:3]:  # Limit to first 3
            suggested_queries.append(f"What is the distribution of {col}?")
        
        # Add comparison questions
        if len(numeric_cols) >= 2:
            suggested_queries.append(f"Compare {numeric_cols[0]} and {numeric_cols[1]}")
        
        # Add categorical questions
        for col in categorical_cols[:2]:  # Limit to first 2
            suggested_queries.append(f"Show me the breakdown of {col}")
        
        # Add time-based questions if datetime columns exist
        if datetime_cols:
            date_col = datetime_cols[0]
            if numeric_cols:
                suggested_queries.append(f"Show trends in {numeric_cols[0]} over {date_col}")
        
        # Add correlation question if enough numeric columns
        if len(numeric_cols) >= 3:
            suggested_queries.append("Show me correlations between numeric variables")
        
        # Add summary question
        suggested_queries.append("Give me a summary of the dataset")
        
        # Display suggested queries as clickable buttons
        cols = st.columns(2)
        for i, query in enumerate(suggested_queries):
            col_idx = i % 2
            with cols[col_idx]:
                if st.button(query, key=f"suggested_query_{i}"):
                    # Set the query in the text input and rerun
                    st.session_state.query_to_set = query
                    st.rerun()
        
        # Check if we need to set a query from a button click
        if 'query_to_set' in st.session_state:
            query = st.session_state.query_to_set
            del st.session_state.query_to_set
    
    # Automated Insights Tab
    with tab2:
        st.header("Automated Data Insights")
        st.markdown("""
        Our AI automatically analyzes your data to discover meaningful patterns, trends, and insights
        without requiring you to ask specific questions.
        """)
        
        # Generate insights button
        if st.button("Generate Insights", key="generate_insights"):
            with st.spinner("Analyzing data and generating insights..."):
                # Generate various types of insights
                insights = []
                
                # 1. Basic statistics insights
                numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
                categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
                
                insights.append({
                    "type": "summary",
                    "title": "Dataset Overview",
                    "content": f"This dataset contains {data.shape[0]:,} rows and {data.shape[1]:,} columns. "
                               f"There are {len(numeric_cols)} numeric columns, {len(categorical_cols)} categorical columns, "
                               f"and {len(datetime_cols)} datetime columns."
                })
                
                # 2. Missing value insights
                missing_cols = data.columns[data.isna().any()].tolist()
                if missing_cols:
                    total_missing = data.isna().sum().sum()
                    missing_pct = (total_missing / (data.shape[0] * data.shape[1])) * 100
                    
                    insights.append({
                        "type": "data_quality",
                        "title": "Missing Values",
                        "content": f"Found {total_missing:,} missing values ({missing_pct:.2f}% of all data) "
                                   f"across {len(missing_cols)} columns. "
                                   f"The columns with the most missing values are: "
                                   f"{', '.join(data.isna().sum().sort_values(ascending=False).head(3).index.tolist())}"
                    })
                
                # 3. Distribution insights for numeric columns
                for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    mean_val = data[col].mean()
                    median_val = data[col].median()
                    skew_val = data[col].skew()
                    
                    skew_desc = ""
                    if abs(skew_val) > 1:
                        skew_desc = f" The distribution is {'positively' if skew_val > 0 else 'negatively'} skewed (skewness: {skew_val:.2f})."
                    
                    mean_median_diff = abs(mean_val - median_val) / abs(median_val) * 100 if median_val != 0 else 0
                    diff_desc = ""
                    if mean_median_diff > 10:
                        diff_desc = f" There's a notable difference between mean and median ({mean_median_diff:.1f}% difference), indicating potential outliers."
                    
                    insights.append({
                        "type": "distribution",
                        "title": f"Distribution of {col}",
                        "content": f"Average {col} is {mean_val:.2f} (median: {median_val:.2f}).{skew_desc}{diff_desc}",
                        "column": col
                    })
                
                # 4. Categorical insights
                for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                    value_counts = data[col].value_counts()
                    top_category = value_counts.index[0] if not value_counts.empty else "N/A"
                    top_pct = (value_counts.iloc[0] / data[col].count()) * 100 if not value_counts.empty else 0
                    
                    insights.append({
                        "type": "categorical",
                        "title": f"Breakdown of {col}",
                        "content": f"The most common value is '{top_category}' ({top_pct:.1f}% of data). "
                                   f"There are {data[col].nunique()} unique values in this column.",
                        "column": col
                    })
                
                # 5. Correlation insights
                if len(numeric_cols) >= 2:
                    corr_matrix = data[numeric_cols].corr()
                    corr_pairs = []
                    
                    for i in range(len(numeric_cols)):
                        for j in range(i+1, len(numeric_cols)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.6:  # Only strong correlations
                                corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_val))
                    
                    if corr_pairs:
                        # Sort by absolute correlation value
                        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                        
                        corr_text = ""
                        for col1, col2, corr in corr_pairs[:3]:  # Limit to top 3
                            direction = "positive" if corr > 0 else "negative"
                            strength = "strong" if abs(corr) > 0.8 else "moderate"
                            corr_text += f"- {col1} and {col2} have a {strength} {direction} correlation ({corr:.2f})\n"
                        
                        insights.append({
                            "type": "correlation",
                            "title": "Notable Correlations",
                            "content": f"Found {len(corr_pairs)} strong correlations between variables:\n{corr_text}",
                            "columns": [pair[:2] for pair in corr_pairs[:3]]
                        })
                
                # 6. Outlier insights
                outlier_cols = []
                for col in numeric_cols:
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                    outlier_pct = len(outliers) / len(data) * 100
                    
                    if outlier_pct > 5:
                        outlier_cols.append((col, outlier_pct))
                
                if outlier_cols:
                    outlier_text = ""
                    for col, pct in sorted(outlier_cols, key=lambda x: x[1], reverse=True)[:3]:
                        outlier_text += f"- {col}: {pct:.1f}% of values are outliers\n"
                    
                    insights.append({
                        "type": "outliers",
                        "title": "Potential Outliers",
                        "content": f"Detected outliers in {len(outlier_cols)} columns:\n{outlier_text}",
                        "column": outlier_cols[0][0] if outlier_cols else None
                    })
                
                # 7. Time series insights
                if datetime_cols and numeric_cols:
                    date_col = datetime_cols[0]
                    
                    # Select a numeric column with potentially interesting trend
                    selected_num_col = None
                    for col in numeric_cols:
                        if data[col].nunique() > 10:
                            selected_num_col = col
                            break
                    
                    if selected_num_col:
                        insights.append({
                            "type": "time_series",
                            "title": f"Time Trends in {selected_num_col}",
                            "content": f"Analyzing trends in {selected_num_col} over time ({date_col}).",
                            "date_column": date_col,
                            "value_column": selected_num_col
                        })
                
                # 8. Group comparison insights
                if categorical_cols and numeric_cols:
                    cat_col = categorical_cols[0]
                    num_col = numeric_cols[0]
                    
                    # Only if the categorical column has a reasonable number of categories
                    if data[cat_col].nunique() <= 10:
                        insights.append({
                            "type": "group_comparison",
                            "title": f"{num_col} by {cat_col}",
                            "content": f"Comparing {num_col} across different {cat_col} categories.",
                            "group_column": cat_col,
                            "value_column": num_col
                        })
                
                # 9. Feature importance insight if appropriate
                potential_targets = detect_potential_target(data)
                if potential_targets:
                    target = potential_targets[0]['column']
                    target_type = potential_targets[0]['type']
                    
                    insights.append({
                        "type": "predictive",
                        "title": f"Predictive Potential: {target}",
                        "content": f"The column '{target}' appears to be a good target for {target_type} modeling. "
                                   f"Reason: {potential_targets[0]['reason']}",
                        "target_column": target,
                        "model_type": target_type
                    })
                
                # Generate visualizations for insights
                insight_viz = Visualization.generate_insight_visualizations(data, insights)
                
                # Display insights
                st.subheader("Key Insights Discovered")
                
                for i, insight in enumerate(insights):
                    # Create expandable section for each insight
                    with st.expander(f"{insight['title']}", expanded=i < 3):
                        st.markdown(insight['content'])
                        
                        # Display visualization if available
                        viz_key = f"insight_{i}"
                        if viz_key in insight_viz:
                            st.plotly_chart(insight_viz[viz_key], width=True)
                
                # Check if there are multiple visualization types for certain insights
                for i, insight in enumerate(insights):
                    viz_keys = [k for k in insight_viz.keys() if k.startswith(f"insight_{i}_")]
                    if viz_keys:
                        with st.expander(f"Additional Visualizations for {insight['title']}", expanded=False):
                            for key in viz_keys:
                                st.plotly_chart(insight_viz[key], width=True)
    
    # Anomaly Detection Tab
    with tab3:
        st.header("Anomaly Detection")
        st.markdown("""
        Automatically detect unusual patterns and outliers in your data. Anomalies could 
        represent errors, fraud, unexpected behavior, or interesting insights.
        """)
        
        # Anomaly detection configuration
        col1, col2 = st.columns(2)
        
        with col1:
            # Target column selection
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            
            if not numeric_cols:
                st.warning("No numeric columns found for anomaly detection.")
            else:
                target_col = st.selectbox(
                    "Select column to analyze for anomalies:",
                    numeric_cols
                )
                
                # Detection method
                detection_method = st.selectbox(
                    "Detection method:",
                    ["IQR (Box Plot)", "Z-Score", "Isolation Forest"],
                    help="IQR is robust to outliers. Z-Score works well for normally distributed data. Isolation Forest can detect complex anomalies."
                )
                
                # Sensitivity slider
                sensitivity = st.slider(
                    "Sensitivity:",
                    min_value=1.0, max_value=5.0, value=2.0, step=0.1,
                    help="Higher values detect more subtle anomalies but may increase false positives."
                )
                
                # Context columns
                context_cols = st.multiselect(
                    "Context columns (optional):",
                    [col for col in data.columns if col != target_col],
                    help="Additional columns to consider when analyzing anomalies."
                )
        
        with col2:
            # Method explanation
            st.subheader("Detection Method Info")
            
            if detection_method == "IQR (Box Plot)":
                st.markdown("""
                **IQR Method**: Detects values that are far from the median using the interquartile range.
                
                - **How it works**: Values below Q1-1.5×IQR or above Q3+1.5×IQR are considered anomalies
                - **Best for**: Data with skewed distributions or when extreme outliers are present
                - **Sensitivity**: Controls how many IQR multiples to use (lower = more anomalies detected)
                """)
            
            elif detection_method == "Z-Score":
                st.markdown("""
                **Z-Score Method**: Detects values that are unusually far from the mean.
                
                - **How it works**: Values with Z-scores exceeding a threshold are considered anomalies
                - **Best for**: Data that follows approximately normal distribution
                - **Sensitivity**: Controls the Z-score threshold (lower = more anomalies detected)
                """)
            
            elif detection_method == "Isolation Forest":
                st.markdown("""
                **Isolation Forest**: Machine learning algorithm that isolates observations by randomly selecting features.
                
                - **How it works**: Anomalies require fewer random splits to be isolated
                - **Best for**: High-dimensional data and complex anomaly patterns
                - **Sensitivity**: Controls the contamination parameter (higher = more anomalies detected)
                """)
        
        # Run anomaly detection
        if st.button("Detect Anomalies", key="detect_anomalies") and 'target_col' in locals():
            with st.spinner("Detecting anomalies..."):
                # Create a dataframe with selected columns
                analysis_cols = [target_col] + context_cols
                analysis_df = data[analysis_cols].copy()
                
                # Handle missing values
                analysis_df = analysis_df.dropna(subset=[target_col])
                
                # Detect anomalies based on selected method
                if detection_method == "IQR (Box Plot)":
                    # IQR method
                    q1 = analysis_df[target_col].quantile(0.25)
                    q3 = analysis_df[target_col].quantile(0.75)
                    iqr = q3 - q1
                    
                    # Adjust threshold based on sensitivity
                    threshold = 1.5 * (5.0 / sensitivity)
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    
                    # Detect anomalies
                    anomalies = analysis_df[(analysis_df[target_col] < lower_bound) | 
                                          (analysis_df[target_col] > upper_bound)].copy()
                    anomalies['anomaly_score'] = abs((analysis_df[target_col] - analysis_df[target_col].median()) / iqr)
                    
                    # Add anomaly direction
                    anomalies['direction'] = np.where(
                        anomalies[target_col] > upper_bound, 
                        'high', 
                        'low'
                    )
                    
                    method_description = f"IQR Method (threshold: {threshold:.2f} × IQR)"
                
                elif detection_method == "Z-Score":
                    # Z-Score method
                    mean = analysis_df[target_col].mean()
                    std = analysis_df[target_col].std()
                    
                    # Adjust threshold based on sensitivity
                    z_threshold = 3.0 / sensitivity
                    
                    # Calculate Z-scores
                    analysis_df['z_score'] = abs((analysis_df[target_col] - mean) / std)
                    
                    # Detect anomalies
                    anomalies = analysis_df[analysis_df['z_score'] > z_threshold].copy()
                    anomalies['anomaly_score'] = anomalies['z_score']
                    
                    # Add anomaly direction
                    anomalies['direction'] = np.where(
                        anomalies[target_col] > mean, 
                        'high', 
                        'low'
                    )
                    
                    method_description = f"Z-Score Method (threshold: {z_threshold:.2f})"
                
                elif detection_method == "Isolation Forest":
                    try:
                        from sklearn.ensemble import IsolationForest
                        
                        # Prepare data
                        numeric_features = analysis_df.select_dtypes(include=np.number)
                        
                        # Adjust contamination based on sensitivity
                        contamination = max(0.01, min(0.5, 0.1 * sensitivity / 2.0))
                        
                        # Train Isolation Forest
                        model = IsolationForest(
                            contamination=contamination,
                            random_state=42
                        )
                        
                        # Fit and predict
                        anomaly_labels = model.fit_predict(numeric_features)
                        analysis_df['anomaly'] = anomaly_labels
                        
                        # Calculate anomaly scores
                        anomaly_scores = -model.score_samples(numeric_features)
                        analysis_df['anomaly_score'] = anomaly_scores
                        
                        # Detect anomalies (isolation forest returns -1 for anomalies)
                        anomalies = analysis_df[analysis_df['anomaly'] == -1].copy()
                        
                        # Add anomaly direction
                        mean = analysis_df[target_col].mean()
                        anomalies['direction'] = np.where(
                            anomalies[target_col] > mean, 
                            'high', 
                            'low'
                        )
                        
                        method_description = f"Isolation Forest (contamination: {contamination:.2f})"
                    except ImportError:
                        st.error("Isolation Forest requires scikit-learn. Please install it to use this method.")
                        return
                
                # Display results
                if len(anomalies) > 0:
                    # Sort anomalies by score
                    anomalies = anomalies.sort_values('anomaly_score', ascending=False)
                    
                    # Calculate anomaly percentage
                    anomaly_pct = len(anomalies) / len(analysis_df) * 100
                    
                    st.success(f"Detected {len(anomalies)} anomalies ({anomaly_pct:.2f}% of data) using {method_description}")
                    
                    # Display anomalies
                    st.subheader("Anomaly Details")
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Anomalies Detected", f"{len(anomalies):,}")
                    with col2:
                        st.metric("Anomaly %", f"{anomaly_pct:.2f}%")
                    with col3:
                        direction_counts = anomalies['direction'].value_counts()
                        high_pct = (direction_counts.get('high', 0) / len(anomalies) * 100) if len(anomalies) > 0 else 0
                        low_pct = (direction_counts.get('low', 0) / len(anomalies) * 100) if len(anomalies) > 0 else 0
                        st.metric("Direction", f"High: {high_pct:.1f}%, Low: {low_pct:.1f}%")
                    
                    # Visualization of anomalies
                    st.subheader("Anomaly Visualization")
                    
                    # Create scatter plot with anomalies highlighted
                    if detection_method == "IQR (Box Plot)":
                        fig = px.box(
                            analysis_df, y=target_col,
                            title=f"Box Plot with Anomalies: {target_col}"
                        )
                        
                        # Add scatter points for anomalies
                        fig.add_trace(
                            go.Scatter(
                                y=anomalies[target_col],
                                mode='markers',
                                marker=dict(
                                    color='red',
                                    size=8,
                                    symbol='circle'
                                ),
                                name='Anomalies'
                            )
                        )
                        
                        st.plotly_chart(fig, width=True)
                        
                        # Show distribution with anomaly thresholds
                        fig2 = px.histogram(
                            analysis_df, x=target_col,
                            title=f"Distribution with Anomaly Thresholds: {target_col}",
                            marginal="box",
                            color_discrete_sequence=['#0078D7']
                        )
                        
                        # Add vertical lines for bounds
                        fig2.add_vline(x=lower_bound, line_dash="dash", line_color="red", 
                                     annotation_text="Lower Threshold")
                        fig2.add_vline(x=upper_bound, line_dash="dash", line_color="red", 
                                     annotation_text="Upper Threshold")
                        
                        st.plotly_chart(fig2, width=True)
                    
                    else:
                        # Create scatter plot showing anomaly scores
                        fig = px.scatter(
                            analysis_df, x=analysis_df.index, y=target_col,
                            color='anomaly_score' if 'anomaly_score' in analysis_df.columns else None,
                            title=f"Anomaly Scores for {target_col}",
                            color_continuous_scale="Viridis"
                        )
                        
                        # Add markers for anomalies
                        fig.add_trace(
                            go.Scatter(
                                x=anomalies.index,
                                y=anomalies[target_col],
                                mode='markers',
                                marker=dict(
                                    color='red',
                                    size=10,
                                    symbol='circle',
                                    line=dict(
                                        color='black',
                                        width=2
                                    )
                                ),
                                name='Anomalies'
                            )
                        )
                        
                        st.plotly_chart(fig, width=True)
                        
                        # Show distribution of anomaly scores
                        if 'anomaly_score' in analysis_df.columns:
                            fig2 = px.histogram(
                                analysis_df, x='anomaly_score',
                                title="Distribution of Anomaly Scores",
                                color_discrete_sequence=['#0078D7']
                            )
                            
                            # Add vertical line for threshold
                            if detection_method == "Z-Score":
                                fig2.add_vline(x=z_threshold, line_dash="dash", line_color="red", 
                                             annotation_text="Threshold")
                            
                            st.plotly_chart(fig2, width=True)
                    
                    # Context analysis if context columns provided
                    if context_cols:
                        st.subheader("Contextual Analysis")
                        st.markdown("Analyzing anomalies in context of other variables.")
                        
                        for context_col in context_cols[:2]:  # Limit to first 2 context columns
                            # Create contextual visualization
                            if pd.api.types.is_numeric_dtype(data[context_col]):
                                # Scatter plot for numeric context
                                fig = px.scatter(
                                    analysis_df, x=context_col, y=target_col,
                                    title=f"{target_col} vs {context_col}",
                                    opacity=0.7
                                )
                                
                                # Add anomalies
                                fig.add_trace(
                                    go.Scatter(
                                        x=anomalies[context_col],
                                        y=anomalies[target_col],
                                        mode='markers',
                                        marker=dict(
                                            color='red',
                                            size=10,
                                            symbol='circle'
                                        ),
                                        name='Anomalies'
                                    )
                                )
                                
                                st.plotly_chart(fig, width=True)
                            
                            else:
                                # Box plot for categorical context
                                fig = px.box(
                                    analysis_df, x=context_col, y=target_col,
                                    title=f"{target_col} by {context_col}"
                                )
                                
                                # Add anomalies
                                fig.add_trace(
                                    go.Scatter(
                                        x=anomalies[context_col],
                                        y=anomalies[target_col],
                                        mode='markers',
                                        marker=dict(
                                            color='red',
                                            size=8,
                                            symbol='circle'
                                        ),
                                        name='Anomalies'
                                    )
                                )
                                
                                st.plotly_chart(fig, width=True)
                        
                        # Check for patterns in anomalies
                        for context_col in context_cols:
                            if not pd.api.types.is_numeric_dtype(data[context_col]):
                                # Check if any category is overrepresented in anomalies
                                context_counts = analysis_df[context_col].value_counts(normalize=True)
                                anomaly_context_counts = anomalies[context_col].value_counts(normalize=True)
                                
                                # Compare distributions
                                comparison = []
                                for category in anomaly_context_counts.index:
                                    if category in context_counts:
                                        ratio = anomaly_context_counts[category] / context_counts[category]
                                        comparison.append({
                                            'Category': category,
                                            'Overall %': context_counts[category] * 100,
                                            'Anomaly %': anomaly_context_counts[category] * 100,
                                            'Ratio': ratio
                                        })
                                
                                if comparison:
                                    comparison_df = pd.DataFrame(comparison)
                                    comparison_df = comparison_df.sort_values('Ratio', ascending=False)
                                    
                                    st.markdown(f"**Distribution Analysis for {context_col}:**")
                                    st.dataframe(comparison_df)
                                    
                                    # Highlight significant patterns
                                    significant = comparison_df[comparison_df['Ratio'] > 2]
                                    if not significant.empty:
                                        st.info(f"🔍 The category '{significant.iloc[0]['Category']}' in {context_col} is "
                                               f"{significant.iloc[0]['Ratio']:.1f}x overrepresented in anomalies.")
                    
                    # Display anomaly records
                    st.subheader("Anomaly Records")
                    display_cols = [target_col, 'anomaly_score', 'direction'] + context_cols
                    st.dataframe(anomalies[display_cols].head(20))
                    
                    # Explanation based on method
                    with st.expander("Anomaly Detection Explanation"):
                        if detection_method == "IQR (Box Plot)":
                            st.markdown(f"""
                            **Method:** Interquartile Range (IQR)
                            
                            Values are considered anomalies if they fall below Q1-{threshold:.2f}×IQR or above Q3+{threshold:.2f}×IQR, where:
                            - Q1 (25th percentile): {q1:.2f}
                            - Q3 (75th percentile): {q3:.2f}
                            - IQR: {iqr:.2f}
                            - Lower threshold: {lower_bound:.2f}
                            - Upper threshold: {upper_bound:.2f}
                            
                            The anomaly score represents how many IQRs away from the median each value is.
                            """)
                        
                        elif detection_method == "Z-Score":
                            st.markdown(f"""
                            **Method:** Z-Score
                            
                            Values are considered anomalies if their Z-score exceeds {z_threshold:.2f}, where Z-score is:
                            
                            Z = |value - mean| / standard deviation
                            
                            For this data:
                            - Mean: {mean:.2f}
                            - Standard deviation: {std:.2f}
                            - Threshold: {z_threshold:.2f} standard deviations
                            
                            The anomaly score is the Z-score itself, representing how many standard deviations each value is from the mean.
                            """)
                        
                        elif detection_method == "Isolation Forest":
                            st.markdown(f"""
                            **Method:** Isolation Forest
                            
                            Isolation Forest is a machine learning algorithm that isolates anomalies by randomly partitioning the data.
                            Anomalies require fewer partitions to be isolated, as they often have atypical feature values.
                            
                            Parameters:
                            - Contamination: {contamination:.2f} (expected percentage of anomalies)
                            
                            The anomaly score represents the model's confidence that a point is anomalous, based on how easily it was isolated.
                            """)
                else:
                    st.warning(f"No anomalies detected in {target_col} using {detection_method}. Try adjusting the sensitivity or selecting a different method.")
    
    # Forecasting Tab
    with tab4:
        st.header("Time Series Forecasting")
        st.markdown("""
        Predict future values and identify trends in your time series data. This feature works
        best with data that contains timestamps and numeric values to forecast.
        """)
        
        # Check for date/time columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Also include columns that might be dates but not properly typed
        for col in data.columns:
            if col not in datetime_cols and ('date' in col.lower() or 'time' in col.lower()):
                try:
                    # Try to convert a sample to datetime
                    sample = data[col].dropna().iloc[0] if not data[col].dropna().empty else None
                    if sample is not None and isinstance(sample, str):
                        pd.to_datetime(sample)
                        datetime_cols.append(col)
                except:
                    pass
        
        if not datetime_cols:
            st.warning("No date or time columns found in your data. Forecasting requires a time variable.")
        else:
            # Configuration options
            col1, col2 = st.columns(2)
            
            with col1:
                # Date column selection
                date_col = st.selectbox(
                    "Select date/time column:",
                    datetime_cols
                )
                
                # Target column selection
                numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
                
                if not numeric_cols:
                    st.warning("No numeric columns found to forecast.")
                else:
                    target_col = st.selectbox(
                        "Select column to forecast:",
                        numeric_cols
                    )
                    
                    # Number of periods to forecast
                    forecast_periods = st.slider(
                        "Forecast periods:",
                        min_value=1, max_value=50, value=10
                    )
                    
                    # Forecasting method
                    forecast_method = st.selectbox(
                        "Forecasting method:",
                        ["Auto", "Exponential Smoothing", "ARIMA", "Prophet"],
                        help="Auto selects the best method based on your data. Choose a specific method if you prefer."
                    )
            
            with col2:
                # Information about data preparation
                st.subheader("Data Preparation")
                st.markdown("""
                For best forecasting results:
                1. Ensure your date column is properly formatted
                2. Check for missing values in your time series
                3. Consider the appropriate time granularity (daily, monthly, etc.)
                """)
                
                # Date range info if date_col is selected
                if 'date_col' in locals():
                    # Ensure the column is datetime type
                    if data[date_col].dtype != 'datetime64[ns]':
                        try:
                            date_data = pd.to_datetime(data[date_col])
                        except:
                            st.error(f"Could not convert {date_col} to datetime format.")
                            date_data = None
                    else:
                        date_data = data[date_col]
                    
                    if date_data is not None:
                        min_date = date_data.min()
                        max_date = date_data.max()
                        date_range = max_date - min_date
                        
                        st.markdown(f"""
                        **Date Range Information:**
                        - Start date: {min_date.strftime('%Y-%m-%d')}
                        - End date: {max_date.strftime('%Y-%m-%d')}
                        - Span: {date_range.days} days
                        """)
                
                # Method explanation
                st.subheader("Method Information")
                
                if 'forecast_method' in locals():
                    if forecast_method == "Auto":
                        st.markdown("""
                        **Auto Method**: Evaluates multiple forecasting methods and selects the best one.
                        
                        - **How it works**: Tests different models and chooses based on error metrics
                        - **Best for**: When you're unsure which forecasting method to use
                        - **Note**: May take longer to run than other methods
                        """)
                    
                    elif forecast_method == "Exponential Smoothing":
                        st.markdown("""
                        **Exponential Smoothing**: Weighted average method that gives more importance to recent observations.
                        
                        - **How it works**: Forecasts using weighted averages with exponentially decreasing weights
                        - **Best for**: Short-term forecasting with trends but no seasonality
                        - **Note**: Simple yet effective for many time series
                        """)
                    
                    elif forecast_method == "ARIMA":
                        st.markdown("""
                        **ARIMA**: AutoRegressive Integrated Moving Average model.
                        
                        - **How it works**: Combines autoregressive, differencing, and moving average components
                        - **Best for**: Data with trends and complex patterns
                        - **Note**: Requires stationary data (trend and seasonality removed)
                        """)
                    
                    elif forecast_method == "Prophet":
                        st.markdown("""
                        **Prophet**: Facebook's forecasting model designed for business time series.
                        
                        - **How it works**: Decomposable model with trend, seasonality, and holiday effects
                        - **Best for**: Data with strong seasonality and multiple seasonal patterns
                        - **Note**: Handles missing data and outliers well
                        """)
            
            # Run forecast
            if 'target_col' in locals() and st.button("Generate Forecast"):
                with st.spinner("Generating forecast..."):
                    # Ensure date column is datetime type
                    if data[date_col].dtype != 'datetime64[ns]':
                        try:
                            ts_data = data.copy()
                            ts_data[date_col] = pd.to_datetime(ts_data[date_col])
                        except:
                            st.error(f"Could not convert {date_col} to datetime format.")
                            return
                    else:
                        ts_data = data.copy()
                    
                    # Create time series dataframe
                    ts_df = ts_data[[date_col, target_col]].dropna()
                    ts_df = ts_df.sort_values(date_col)
                    
                    if len(ts_df) < 10:
                        st.error("Not enough data points for forecasting. Need at least 10 observations.")
                        return
                    
                    # Determine time frequency
                    try:
                        # Try to infer frequency
                        ts_df.set_index(date_col, inplace=True)
                        inferred_freq = pd.infer_freq(ts_df.index)
                        
                        if inferred_freq is None:
                            # Calculate most common difference
                            diff = ts_df.index.to_series().diff().dropna()
                            most_common_diff = diff.mode().iloc[0] if not diff.mode().empty else None
                            
                            if most_common_diff is not None and most_common_diff.days > 0:
                                days = most_common_diff.days
                                
                                if days == 1:
                                    freq = 'D'  # Daily
                                elif days == 7:
                                    freq = 'W'  # Weekly
                                elif 28 <= days <= 31:
                                    freq = 'M'  # Monthly
                                elif 90 <= days <= 92:
                                    freq = 'Q'  # Quarterly
                                elif 365 <= days <= 366:
                                    freq = 'Y'  # Yearly
                                else:
                                    freq = f'{days}D'  # Custom days
                            else:
                                freq = 'D'  # Default to daily
                        else:
                            freq = inferred_freq
                        st.write("Inferred frequency:", inferred_freq)
                        st.write("Most common difference:", most_common_diff)
                        st.write("Final frequency:", freq)
                        # Apply simple forecasting
                        # This is just demonstrating the UI - in a real implementation, 
                        # we would use proper time series models like ARIMA, exponential smoothing, etc.
                        
                        # Generate simple forecast (for demonstration)
                        last_date = ts_df.index.max()
                        forecast_dates = pd.date_range(start=last_date, periods=forecast_periods+1, freq=freq)[1:]
                        
                        # Create mock forecast values
                        last_values = ts_df[target_col].iloc[-5:].values
                        mean_value = np.mean(last_values)
                        slope = (last_values[-1] - last_values[0]) / 4
                        
                        forecast_values = []
                        current_value = last_values[-1]
                        
                        for i in range(forecast_periods):
                            # Simple trend forecasting with some randomness
                            next_value = current_value + slope + np.random.normal(0, abs(slope/4))
                            forecast_values.append(next_value)
                            current_value = next_value
                        
                        # Create forecast dataframe
                        forecast_df = pd.DataFrame({
                            'date': forecast_dates,
                            'forecast': forecast_values,
                            'lower_bound': [v - abs(v*0.15) for v in forecast_values],
                            'upper_bound': [v + abs(v*0.15) for v in forecast_values]
                        })
                        
                        # Visualize the forecast
                        st.subheader("Forecast Results")
                        
                        # Reset index for plotting
                        plot_df = ts_df.reset_index()
                        
                        # Create interactive time series plot with forecast
                        fig = go.Figure()
                        
                        # Add historical data
                        fig.add_trace(go.Scatter(
                            x=plot_df[date_col], 
                            y=plot_df[target_col],
                            mode='lines',
                            name='Historical Data',
                            line=dict(color='blue')
                        ))
                        
                        # Add forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_df['date'], 
                            y=forecast_df['forecast'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # Add confidence interval
                        fig.add_trace(go.Scatter(
                            x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                            y=forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(231,107,243,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=True,
                            name='95% Confidence Interval'
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"Forecast for {target_col}",
                            xaxis_title="Date",
                            yaxis_title=target_col,
                            legend=dict(x=0.01, y=0.99),
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig, width=True)
                        
                        # Display forecast values
                        st.subheader("Forecast Values")
                        st.dataframe(forecast_df)
                        
                        # Forecast metrics
                        st.subheader("Forecast Metrics")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            last_value = plot_df[target_col].iloc[-1]
                            last_date = plot_df[date_col].iloc[-1]
                            st.metric("Last Known Value", f"{last_value:.2f}", 
                                     f"on {last_date.strftime('%Y-%m-%d')}")
                        
                        with col2:
                            forecast_end = forecast_df['forecast'].iloc[-1]
                            change = (forecast_end - last_value) / last_value * 100
                            st.metric("End of Forecast", f"{forecast_end:.2f}", 
                                     f"{change:+.2f}% change")
                        
                        with col3:
                            forecast_max = forecast_df['forecast'].max()
                            max_change = (forecast_max - last_value) / last_value * 100
                            st.metric("Max Forecast Value", f"{forecast_max:.2f}", 
                                     f"{max_change:+.2f}% from last known")
                        
                        # Forecast explanation
                        with st.expander("Forecast Explanation"):
                            st.markdown(f"""
                            **Forecasting Method:** {forecast_method}
                            
                            **Time Series Properties:**
                            - Frequency: {freq}
                            - Historical data points: {len(plot_df)}
                            - Forecast periods: {forecast_periods}
                            
                            **Forecast Interpretation:**
                            - The forecast shows a {"positive" if slope > 0 else "negative"} trend
                            - Confidence intervals widen as we forecast further into the future, reflecting increased uncertainty
                            - The forecast predicts the value of {target_col} will be around {forecast_end:.2f} by {forecast_df['date'].iloc[-1].strftime('%Y-%m-%d')}
                            
                            **Important Note:**
                            This is a simplified forecast for demonstration purposes. For production use, more sophisticated models with proper validation should be employed.
                            """)
                    
                    except Exception as e:
                        st.error(f"Error generating forecast: {str(e)}")
                        st.exception(e)
    
    # What's next section
    # st.markdown("---")
    # st.markdown("## What's Next?")
    # st.info("👉 Proceed to the **Statistical Analysis** page to run hypothesis tests and analyze relationships in your data.")

if __name__ == "__main__":
    main()
