import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import json
import random
from datetime import datetime

# Add utils to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.visualization import Visualization
from utils.helpers import get_data_summary

st.set_page_config(
    page_title="DataInsights Pro",
    layout="wide"
)

def main():
    st.title("📈 Data Visualization")
    
    # Check if data is uploaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ Please upload a Dataset in the **Data Upload** page.")
        st.stop()
    
    data = st.session_state.data
    
    # Store visualizations in session state if not already present
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = []
    
    # Sidebar with visualization options
    # st.sidebar.header("Visualization Options")
    
    # Create tabs for different visualization approaches
    tab1, tab2, tab3, tab4 = st.tabs([
        "AI-Suggested Charts", 
        "Chart Builder", 
        "Dashboard Creator",
        "Saved Visualizations"
    ])
    
    # AI-Suggested Charts Tab
    with tab1:
        st.header("AI-Suggested Visualizations")
        st.markdown("""
        Based on your data characteristics, our AI suggests the following visualizations 
        that might provide valuable insights. Click on any suggestion to visualize it.
        """)
        
        # Generate visualization suggestions
        with st.spinner("Analyzing your data to suggest visualizations..."):
            suggestions = Visualization.suggest_visualizations(data)
        
        # Display suggestions in an organized way
        if suggestions:
            # Group suggestions by complexity
            simple_viz = [s for s in suggestions if s['complexity'] == 'simple' and s['type'] not in ['dashboard_basic', 'dashboard_advanced']]
            medium_viz = [s for s in suggestions if s['complexity'] == 'medium' and s['type'] not in ['dashboard_basic', 'dashboard_advanced']]
            advanced_viz = [s for s in suggestions if s['complexity'] == 'advanced' and s['type'] not in ['dashboard_basic', 'dashboard_advanced']]
            dashboards = [s for s in suggestions if s['type'] in ['dashboard_basic', 'dashboard_advanced']]
            
            # Display simple visualizations
            if simple_viz:
                st.subheader("Quick Insights")
                cols = st.columns(min(3, len(simple_viz)))
                
                for i, viz in enumerate(simple_viz[:6]):  # Limit to 6 simple viz
                    col_idx = i % len(cols)
                    with cols[col_idx]:
                        if st.button(f"📈 {viz['title']}", key=f"simple_viz_{i}"):
                            # Create visualization
                            try:
                                fig = Visualization.create_visualization(
                                    data, 
                                    viz['type'],
                                    title=viz['title'],
                                    x=viz['columns'][0] if len(viz['columns']) > 0 else None,
                                    y=viz['columns'][1] if len(viz['columns']) > 1 else None,
                                    color=viz['columns'][2] if len(viz['columns']) > 2 else None,
                                    height=500
                                )
                                
                                # Display visualization
                                st.plotly_chart(fig, width=True)
                                
                                # Option to save visualization
                                if st.button("💾 Save This Visualization", key=f"save_simple_{i}"):
                                    if 'visualizations' not in st.session_state:
                                        st.session_state.visualizations = []
                                    
                                    viz_config = {
                                        'type': viz['type'],
                                        'title': viz['title'],
                                        'description': viz['description'],
                                        'columns': viz['columns'],
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    
                                    st.session_state.visualizations.append(viz_config)
                                    st.success("Visualization saved! View it in the 'Saved Visualizations' tab.")
                            except Exception as e:
                                st.error(f"Error creating visualization: {str(e)}")
            
            # Display medium visualizations
            if medium_viz:
                st.subheader("Detailed Analysis")
                cols = st.columns(min(2, len(medium_viz)))
                
                for i, viz in enumerate(medium_viz[:4]):  # Limit to 4 medium viz
                    col_idx = i % len(cols)
                    with cols[col_idx]:
                        if st.button(f"📊 {viz['title']}", key=f"medium_viz_{i}"):
                            # Create visualization
                            try:
                                fig = Visualization.create_visualization(
                                    data, 
                                    viz['type'],
                                    title=viz['title'],
                                    x=viz['columns'][0] if len(viz['columns']) > 0 else None,
                                    y=viz['columns'][1] if len(viz['columns']) > 1 else None,
                                    color=viz['columns'][2] if len(viz['columns']) > 2 else None,
                                    height=500
                                )
                                
                                # Display visualization
                                st.plotly_chart(fig, width=True)
                                
                                # Option to save visualization
                                if st.button("💾 Save This Visualization", key=f"save_medium_{i}"):
                                    if 'visualizations' not in st.session_state:
                                        st.session_state.visualizations = []
                                    
                                    viz_config = {
                                        'type': viz['type'],
                                        'title': viz['title'],
                                        'description': viz['description'],
                                        'columns': viz['columns'],
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    
                                    st.session_state.visualizations.append(viz_config)
                                    st.success("Visualization saved! View it in the 'Saved Visualizations' tab.")
                            except Exception as e:
                                st.error(f"Error creating visualization: {str(e)}")
            
            # Display advanced visualizations
            if advanced_viz:
                st.subheader("Advanced Visualizations")
                cols = st.columns(min(2, len(advanced_viz)))
                
                for i, viz in enumerate(advanced_viz[:4]):  # Limit to 4 advanced viz
                    col_idx = i % len(cols)
                    with cols[col_idx]:
                        if st.button(f"🔬 {viz['title']}", key=f"advanced_viz_{i}"):
                            # Create visualization
                            try:
                                fig = Visualization.create_visualization(
                                    data, 
                                    viz['type'],
                                    title=viz['title'],
                                    x=viz['columns'][0] if len(viz['columns']) > 0 else None,
                                    y=viz['columns'][1] if len(viz['columns']) > 1 else None,
                                    color=viz['columns'][2] if len(viz['columns']) > 2 else None,
                                    columns=viz['columns'] if viz['type'] == 'heatmap' else None,
                                    size=viz['columns'][2] if viz['type'] == 'bubble' and len(viz['columns']) > 2 else None,
                                    height=600
                                )
                                
                                # Display visualization
                                st.plotly_chart(fig, width=True)
                                
                                # Option to save visualization
                                if st.button("💾 Save This Visualization", key=f"save_advanced_{i}"):
                                    if 'visualizations' not in st.session_state:
                                        st.session_state.visualizations = []
                                    
                                    viz_config = {
                                        'type': viz['type'],
                                        'title': viz['title'],
                                        'description': viz['description'],
                                        'columns': viz['columns'],
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    
                                    st.session_state.visualizations.append(viz_config)
                                    st.success("Visualization saved! View it in the 'Saved Visualizations' tab.")
                            except Exception as e:
                                st.error(f"Error creating visualization: {str(e)}")
            
            # Display dashboard suggestions
            if dashboards:
                st.subheader("Dashboard Suggestions")
                for i, dashboard in enumerate(dashboards):
                    if st.button(f"🔍 {dashboard['title']}", key=f"dashboard_{i}"):
                        # Find the component visualizations
                        dashboard_components = []
                        for comp_title in dashboard['components']:
                            # Find matching visualization in suggestions
                            for viz in suggestions:
                                if viz.get('title') == comp_title:
                                    dashboard_components.append({
                                        'type': viz['type'],
                                        'title': viz['title'],
                                        'x': viz['columns'][0] if len(viz['columns']) > 0 else None,
                                        'y': viz['columns'][1] if len(viz['columns']) > 1 else None,
                                        'color': viz['columns'][2] if len(viz['columns']) > 2 else None,
                                        'columns': viz['columns'] if viz['type'] == 'heatmap' else None
                                    })
                        
                        # Create dashboard if components are found
                        if dashboard_components:
                            try:
                                dashboard_fig = Visualization.create_dashboard(data, dashboard_components)
                                st.plotly_chart(dashboard_fig, width=True)
                                
                                # Option to save dashboard
                                if st.button("💾 Save This Dashboard", key=f"save_dashboard_{i}"):
                                    if 'visualizations' not in st.session_state:
                                        st.session_state.visualizations = []
                                    
                                    dashboard_config = {
                                        'type': 'dashboard',
                                        'title': dashboard['title'],
                                        'description': dashboard['description'],
                                        'components': dashboard_components,
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    
                                    st.session_state.visualizations.append(dashboard_config)
                                    st.success("Dashboard saved! View it in the 'Saved Visualizations' tab.")
                            except Exception as e:
                                st.error(f"Error creating dashboard: {str(e)}")
        else:
            st.info("Unable to generate visualization suggestions for this dataset. Try the Chart Builder tab to create custom visualizations.")
    
    # Chart Builder Tab
    with tab2:
        st.header("Custom Chart Builder")
        st.markdown("Create your own visualizations by selecting columns and chart types.")
        
        # Chart type selection
        chart_type = st.selectbox(
            "Select chart type:",
            [
                "Bar Chart", "Line Chart", "Scatter Plot", "Histogram", 
                "Box Plot", "Pie Chart", "Heatmap", "Area Chart", 
                "Violin Plot", "Bubble Chart", "Funnel Chart"
            ]
        )
        
        # Get column types for appropriate selection options
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        datetime_cols.extend([col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()])
        datetime_cols = list(set(datetime_cols))  # Remove duplicates
        
        # Container for chart parameters
        with st.form("chart_form"):
            # Common parameters
            title = st.text_input("Chart Title:", value=f"{chart_type}")
            
            # Chart-specific parameters
            if chart_type == "Bar Chart":
                x_col = st.selectbox("X-axis (Categories):", data.columns.tolist())
                
                y_options = ["Count"]
                if numeric_cols:
                    y_options.extend(numeric_cols)
                
                y_col = st.selectbox("Y-axis:", y_options)
                color_col = st.selectbox("Color by (optional):", ["None"] + categorical_cols)
                orientation = st.selectbox("Orientation:", ["Vertical", "Horizontal"])
                
            elif chart_type == "Line Chart":
                x_col = st.selectbox("X-axis:", datetime_cols + numeric_cols)
                y_col = st.selectbox("Y-axis:", numeric_cols)
                color_col = st.selectbox("Color by (optional):", ["None"] + categorical_cols)
                line_shape = st.selectbox("Line Shape:", ["linear", "spline", "hv", "vh", "hvh", "vhv"])
                
            elif chart_type == "Scatter Plot":
                x_col = st.selectbox("X-axis:", numeric_cols)
                y_col = st.selectbox("Y-axis:", numeric_cols)
                color_col = st.selectbox("Color by (optional):", ["None"] + categorical_cols)
                size_col = st.selectbox("Size by (optional):", ["None"] + numeric_cols)
                add_trendline = st.checkbox("Add Trendline")
                
            elif chart_type == "Histogram":
                x_col = st.selectbox("Value:", numeric_cols)
                n_bins = st.slider("Number of Bins:", min_value=5, max_value=100, value=30)
                color_col = st.selectbox("Color by (optional):", ["None"] + categorical_cols)
                show_kde = st.checkbox("Show Density Curve")
                
            elif chart_type == "Box Plot":
                y_col = st.selectbox("Values:", numeric_cols)
                x_col = st.selectbox("Group by (optional):", ["None"] + categorical_cols)
                color_col = st.selectbox("Color by:", ["None"] + categorical_cols)
                
            elif chart_type == "Pie Chart":
                x_col = st.selectbox("Categories:", categorical_cols)
                values_col = st.selectbox("Values:", ["Count"] + numeric_cols)
                donut = st.checkbox("Donut Chart")
                
            elif chart_type == "Heatmap":
                corr_method = st.selectbox("Correlation Method:", ["Pearson", "Spearman", "Kendall"])
                selected_cols = st.multiselect(
                    "Select columns for correlation:",
                    numeric_cols,
                    default=numeric_cols[:min(10, len(numeric_cols))]
                )
                
            elif chart_type == "Area Chart":
                x_col = st.selectbox("X-axis:", datetime_cols + numeric_cols)
                y_col = st.selectbox("Y-axis:", numeric_cols)
                color_col = st.selectbox("Group by (optional):", ["None"] + categorical_cols)
                stack = st.checkbox("Stacked")
                
            elif chart_type == "Violin Plot":
                x_col = st.selectbox("Group by:", categorical_cols)
                y_col = st.selectbox("Values:", numeric_cols)
                color_col = st.selectbox("Color by:", ["None"] + categorical_cols)
                
            elif chart_type == "Bubble Chart":
                x_col = st.selectbox("X-axis:", numeric_cols)
                y_col = st.selectbox("Y-axis:", numeric_cols)
                size_col = st.selectbox("Bubble Size:", numeric_cols)
                color_col = st.selectbox("Color:", ["None"] + categorical_cols)
                
            elif chart_type == "Funnel Chart":
                x_col = st.selectbox("Categories (Steps):", categorical_cols)
                y_col = st.selectbox("Values:", numeric_cols)
            
            # Submit button
            submitted = st.form_submit_button("Generate Visualization")
        
        # Generate and display chart
        if submitted:
            try:
                # Process parameters
                viz_params = {
                    "title": title,
                    "height": 600
                }
                
                # Convert "None" to None for optional parameters
                if 'color_col' in locals() and color_col == "None":
                    color_col = None
                
                # Add chart-specific parameters
                if chart_type == "Bar Chart":
                    viz_type = "bar"
                    viz_params["x"] = x_col
                    
                    if y_col == "Count":
                        # Use value counts for count
                        chart_data = data[x_col].value_counts().reset_index()
                        chart_data.columns = [x_col, 'count']
                        y_col = 'count'
                    else:
                        chart_data = data
                    
                    viz_params["y"] = y_col
                    viz_params["color"] = color_col
                    viz_params["orientation"] = "h" if orientation == "Horizontal" else "v"
                    
                elif chart_type == "Line Chart":
                    viz_type = "line"
                    viz_params["x"] = x_col
                    viz_params["y"] = y_col
                    viz_params["color"] = color_col
                    viz_params["line_shape"] = line_shape
                    chart_data = data
                
                elif chart_type == "Scatter Plot":
                    viz_type = "scatter"
                    viz_params["x"] = x_col
                    viz_params["y"] = y_col
                    viz_params["color"] = color_col
                    
                    if size_col != "None":
                        viz_params["size"] = size_col
                    
                    if add_trendline:
                        viz_params["trendline"] = "ols"
                    
                    chart_data = data
                
                elif chart_type == "Histogram":
                    viz_type = "histogram"
                    viz_params["x"] = x_col
                    viz_params["nbins"] = n_bins
                    viz_params["color"] = color_col
                    viz_params["density"] = show_kde
                    chart_data = data
                
                elif chart_type == "Box Plot":
                    viz_type = "box"
                    viz_params["y"] = y_col
                    
                    if x_col != "None":
                        viz_params["x"] = x_col
                    
                    viz_params["color"] = color_col
                    chart_data = data
                
                elif chart_type == "Pie Chart":
                    viz_type = "pie"
                    viz_params["x"] = x_col
                    
                    if values_col == "Count":
                        # Use value counts
                        chart_data = data[x_col].value_counts().reset_index()
                        chart_data.columns = [x_col, 'count']
                        viz_params["values"] = 'count'
                    else:
                        chart_data = data
                        viz_params["values"] = values_col
                    
                    if donut:
                        viz_params["hole"] = 0.4
                
                elif chart_type == "Heatmap":
                    viz_type = "heatmap"
                    viz_params["columns"] = selected_cols
                    viz_params["corr_method"] = corr_method.lower()
                    chart_data = data
                
                elif chart_type == "Area Chart":
                    viz_type = "area"
                    viz_params["x"] = x_col
                    viz_params["y"] = y_col
                    viz_params["color"] = color_col
                    viz_params["groupnorm"] = "percent" if stack else None
                    chart_data = data
                
                elif chart_type == "Violin Plot":
                    viz_type = "violin"
                    viz_params["x"] = x_col
                    viz_params["y"] = y_col
                    viz_params["color"] = color_col
                    chart_data = data
                
                elif chart_type == "Bubble Chart":
                    viz_type = "bubble"
                    viz_params["x"] = x_col
                    viz_params["y"] = y_col
                    viz_params["size"] = size_col
                    viz_params["color"] = color_col
                    chart_data = data
                
                elif chart_type == "Funnel Chart":
                    viz_type = "funnel"
                    viz_params["x"] = x_col
                    viz_params["y"] = y_col
                    chart_data = data
                
                # Generate visualization
                fig = Visualization.create_visualization(chart_data, viz_type, **viz_params)
                
                # Display visualization
                if fig:
                    st.plotly_chart(fig, width=True)
                    
                    # Option to save visualization
                    if st.button("💾 Save This Visualization", key="save_custom"):
                        if 'visualizations' not in st.session_state:
                            st.session_state.visualizations = []
                        
                        viz_config = {
                            'type': viz_type,
                            'title': title,
                            'params': viz_params,
                            'chart_type': chart_type,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        st.session_state.visualizations.append(viz_config)
                        st.success("Visualization saved! View it in the 'Saved Visualizations' tab.")
                else:
                    st.error("Failed to create visualization. Please check your parameters.")
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                st.exception(e)
    
    # Dashboard Creator Tab
    with tab3:
        st.header("Custom Dashboard Creator")
        st.markdown("""
        Create a custom dashboard by selecting multiple chart types and arranging them together.
        Add up to 6 charts to your dashboard.
        """)
        
        # Number of charts selection
        n_charts = st.slider("Number of charts in dashboard:", min_value=1, max_value=6, value=4)
        
        dashboard_title = st.text_input("Dashboard Title:", value="Custom Data Dashboard")
        
        # Create form for dashboard configuration
        with st.form("dashboard_form"):
            # Container for chart configurations
            charts_config = []
            
            for row in range((n_charts + 1) // 2):
                cols = st.columns(2)    
                for col_idx in range(2):
                    i = row * 2 + col_idx
                    if i >= n_charts:
                        break
                    with cols[col_idx]:
                        st.subheader(f"Chart {i+1} Configuration")
                        
                        # Chart type selection
                        chart_type = st.selectbox(
                            f"Select chart type for Chart {i+1}:",
                            [
                                "Bar Chart", "Line Chart", "Scatter Plot", "Histogram", 
                                "Box Plot", "Pie Chart", "Area Chart", "Bubble Chart"
                            ],
                            key=f"db_chart_type_{i}"
                        )
                
                        # Chart title
                        chart_title = st.text_input(f"Title for Chart {i+1}:", value=f"Chart {i+1}", key=f"db_title_{i}")
                
                        # Chart-specific parameters
                        if chart_type == "Bar Chart":
                            x_col = st.selectbox("X-axis (Categories):", data.columns.tolist(), key=f"db_x_{i}")
                            
                            y_options = ["Count"]
                            if numeric_cols:
                                y_options.extend(numeric_cols)
                            
                            y_col = st.selectbox("Y-axis:", y_options, key=f"db_y_{i}")
                            color_col = st.selectbox("Color by (optional):", ["None"] + categorical_cols, key=f"db_color_{i}")
                            
                            charts_config.append({
                                'type': 'bar',
                                'title': chart_title,
                                'x': x_col,
                                'y': y_col if y_col != "Count" else None,
                                'color': None if color_col == "None" else color_col,
                                'count': y_col == "Count"
                            })
                        
                        elif chart_type == "Line Chart":
                            x_col = st.selectbox("X-axis:", datetime_cols + numeric_cols, key=f"db_x_{i}")
                            y_col = st.selectbox("Y-axis:", numeric_cols, key=f"db_y_{i}")
                            color_col = st.selectbox("Color by (optional):", ["None"] + categorical_cols, key=f"db_color_{i}")
                            
                            charts_config.append({
                                'type': 'line',
                                'title': chart_title,
                                'x': x_col,
                                'y': y_col,
                                'color': None if color_col == "None" else color_col
                            })
                        
                        elif chart_type == "Scatter Plot":
                            x_col = st.selectbox("X-axis:", numeric_cols, key=f"db_x_{i}")
                            y_col = st.selectbox("Y-axis:", numeric_cols, key=f"db_y_{i}")
                            color_col = st.selectbox("Color by (optional):", ["None"] + categorical_cols, key=f"db_color_{i}")
                            
                            charts_config.append({
                                'type': 'scatter',
                                'title': chart_title,
                                'x': x_col,
                                'y': y_col,
                                'color': None if color_col == "None" else color_col
                            })
                        
                        elif chart_type == "Histogram":
                            x_col = st.selectbox("Value:", numeric_cols, key=f"db_x_{i}")
                            color_col = st.selectbox("Color by (optional):", ["None"] + categorical_cols, key=f"db_color_{i}")
                            
                            charts_config.append({
                                'type': 'histogram',
                                'title': chart_title,
                                'x': x_col,
                                'color': None if color_col == "None" else color_col
                            })
                        
                        elif chart_type == "Box Plot":
                            y_col = st.selectbox("Values:", numeric_cols, key=f"db_y_{i}")
                            x_col = st.selectbox("Group by (optional):", ["None"] + categorical_cols, key=f"db_x_{i}")
                            
                            charts_config.append({
                                'type': 'box',
                                'title': chart_title,
                                'y': y_col,
                                'x': None if x_col == "None" else x_col
                            })
                        
                        elif chart_type == "Pie Chart":
                            x_col = st.selectbox("Categories:", categorical_cols, key=f"db_x_{i}")
                            values_col = st.selectbox("Values:", ["Count"] + numeric_cols, key=f"db_values_{i}")
                            
                            charts_config.append({
                                'type': 'pie',
                                'title': chart_title,
                                'names': x_col,
                                'values': 'count' if values_col == "Count" else values_col,
                                'count': values_col == "Count"
                            })
                        
                        elif chart_type == "Area Chart":
                            x_col = st.selectbox("X-axis:", datetime_cols + numeric_cols, key=f"db_x_{i}")
                            y_col = st.selectbox("Y-axis:", numeric_cols, key=f"db_y_{i}")
                            color_col = st.selectbox("Group by (optional):", ["None"] + categorical_cols, key=f"db_color_{i}")
                            
                            charts_config.append({
                                'type': 'area',
                                'title': chart_title,
                                'x': x_col,
                                'y': y_col,
                                'color': None if color_col == "None" else color_col
                            })
                        
                        elif chart_type == "Bubble Chart":
                            x_col = st.selectbox("X-axis:", numeric_cols, key=f"db_x_{i}")
                            y_col = st.selectbox("Y-axis:", numeric_cols, key=f"db_y_{i}")
                            size_col = st.selectbox("Bubble Size:", numeric_cols, key=f"db_size_{i}")
                            color_col = st.selectbox("Color:", ["None"] + categorical_cols, key=f"db_color_{i}")
                            
                            charts_config.append({
                                'type': 'bubble',
                                'title': chart_title,
                                'x': x_col,
                                'y': y_col,
                                'size': size_col,
                                'color': None if color_col == "None" else color_col
                            })
                
                st.markdown("---")
            
            # Submit button
            submitted = st.form_submit_button("Generate Dashboard")
        
        # Generate and display dashboard
        if submitted:
            try:
                # Prepare dashboard
                with st.spinner("Creating dashboard..."):
                    # Process chart configurations
                    dashboard_components = []
                    
                    for i, config in enumerate(charts_config):
                        chart_type = config['type']
                        
                        # Handle count-based charts
                        if chart_type == 'bar' and config.get('count', False):
                            # Use value counts for count-based bar charts
                            value_counts = data[config['x']].value_counts().reset_index()
                            value_counts.columns = [config['x'], 'count']
                            
                            dashboard_components.append({
                                'type': chart_type,
                                'title': config['title'],
                                'x': config['x'],
                                'y': 'count',
                                'color': config['color']
                            })
                        
                        elif chart_type == 'pie' and config.get('count', False):
                            # Use value counts for count-based pie charts
                            value_counts = data[config['names']].value_counts().reset_index()
                            value_counts.columns = [config['names'], 'count']
                            
                            dashboard_components.append({
                                'type': chart_type,
                                'title': config['title'],
                                'names': config['names'],
                                'values': 'count'
                            })
                        
                        else:
                            # Standard component
                            dashboard_components.append(config)
                    
                    # Create dashboard
                    dashboard_fig = Visualization.create_dashboard(data, dashboard_components)
                    dashboard_fig.update_layout(title_text=dashboard_title)
                    
                    # Display dashboard
                    st.plotly_chart(dashboard_fig, width=True)
                    
                    # Option to save dashboard
                    if st.button("💾 Save This Dashboard", key="save_custom_dashboard"):
                        if 'visualizations' not in st.session_state:
                            st.session_state.visualizations = []
                        
                        dashboard_config = {
                            'type': 'dashboard',
                            'title': dashboard_title,
                            'components': dashboard_components,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        st.session_state.visualizations.append(dashboard_config)
                        st.success("Dashboard saved! View it in the 'Saved Visualizations' tab.")
                
            except Exception as e:
                st.error(f"Error creating dashboard: {str(e)}")
                st.exception(e)
    
    # Saved Visualizations Tab
    with tab4:
        st.header("Saved Visualizations")
        
        if not st.session_state.visualizations:
            st.info("You haven't saved any visualizations yet. Create and save visualizations from the other tabs.")
        else:
            # Display saved visualizations
            st.markdown(f"You have {len(st.session_state.visualizations)} saved visualizations.")
            
            # Option to delete visualizations
            if st.button("Clear All Saved Visualizations"):
                st.session_state.visualizations = []
                st.success("All visualizations cleared.")
                st.rerun()
            
            # List saved visualizations
            for i, viz_config in enumerate(st.session_state.visualizations):
                st.markdown("---")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader(viz_config.get('title', f"Visualization {i+1}"))
                    st.text(f"Created: {viz_config.get('timestamp', 'N/A')}")
                
                with col2:
                    if st.button("Delete", key=f"delete_viz_{i}"):
                        st.session_state.visualizations.pop(i)
                        st.success("Visualization deleted.")
                        st.rerun()
                
                # Display the visualization
                if viz_config.get('type') == 'dashboard':
                    # Recreate dashboard
                    try:
                        dashboard_fig = Visualization.create_dashboard(data, viz_config.get('components', []))
                        dashboard_fig.update_layout(title_text=viz_config.get('title', 'Dashboard'))
                        st.plotly_chart(dashboard_fig, width=True)
                    except Exception as e:
                        st.error(f"Error displaying dashboard: {str(e)}")
                else:
                    # Recreate standard visualization
                    try:
                        viz_type = viz_config.get('type')
                        
                        # Different parameter handling based on source
                        if 'params' in viz_config:
                            # From Chart Builder
                            params = viz_config['params']
                            fig = Visualization.create_visualization(data, viz_type, **params)
                        else:
                            # From AI suggestions
                            columns = viz_config.get('columns', [])
                            fig = Visualization.create_visualization(
                                data, 
                                viz_type,
                                title=viz_config.get('title', ''),
                                x=columns[0] if len(columns) > 0 else None,
                                y=columns[1] if len(columns) > 1 else None,
                                color=columns[2] if len(columns) > 2 else None,
                                columns=columns if viz_type == 'heatmap' else None,
                                height=500
                            )
                        
                        st.plotly_chart(fig, width=True)
                    except Exception as e:
                        st.error(f"Error displaying visualization: {str(e)}")
    
    # What's next section
    st.markdown("---")
    st.markdown("## What's Next?")
    st.info("👉 Proceed to the **Statistical Analysis** page to discover intelligent patterns and anomalies in your data.")

if __name__ == "__main__":
    main()
