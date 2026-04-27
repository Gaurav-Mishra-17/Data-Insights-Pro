import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple, Optional

def auto_suggest_charts(df: pd.DataFrame, column_name: str = None) -> List[Dict[str, Any]]:
    """
    Automatically suggest appropriate chart types based on the data
    
    Args:
        df: The pandas dataframe
        column_name: Optional specific column to focus on
        
    Returns:
        List of chart suggestions with metadata
    """
    suggestions = []
    
    # If specific column is provided, focus on it
    if column_name and column_name in df.columns:
        column_dtype = df[column_name].dtype
        column_nunique = df[column_name].nunique()
        
        # For numeric columns
        if pd.api.types.is_numeric_dtype(column_dtype):
            # Histogram
            suggestions.append({
                'type': 'histogram',
                'title': f'Distribution of {column_name}',
                'columns': [column_name],
                'description': f'Shows the distribution of values in {column_name}'
            })
            
            # Box plot
            suggestions.append({
                'type': 'box',
                'title': f'Box Plot of {column_name}',
                'columns': [column_name],
                'description': f'Shows the statistical distribution including outliers for {column_name}'
            })
            
            # Find categorical columns for potential grouping
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for cat_col in categorical_cols:
                if df[cat_col].nunique() <= 20:  # Only suggest if cardinality is reasonable
                    # Bar chart grouped by category
                    suggestions.append({
                        'type': 'bar',
                        'title': f'Average {column_name} by {cat_col}',
                        'columns': [column_name, cat_col],
                        'description': f'Compare average {column_name} across different {cat_col} categories'
                    })
                    
                    # Box plot grouped by category
                    suggestions.append({
                        'type': 'box',
                        'title': f'Distribution of {column_name} by {cat_col}',
                        'columns': [column_name, cat_col],
                        'description': f'Compare distributions of {column_name} across different {cat_col} categories'
                    })
        
        # For categorical/text columns
        elif pd.api.types.is_string_dtype(column_dtype) or pd.api.types.is_categorical_dtype(column_dtype):
            if column_nunique <= 50:  # Only if reasonable number of categories
                # Bar chart for value counts
                suggestions.append({
                    'type': 'bar',
                    'title': f'Frequency of {column_name} Values',
                    'columns': [column_name],
                    'description': f'Shows the count of occurrences for each value in {column_name}'
                })
                
                # Pie chart if few categories
                if column_nunique <= 10:
                    suggestions.append({
                        'type': 'pie',
                        'title': f'Proportion of {column_name} Values',
                        'columns': [column_name],
                        'description': f'Shows the proportion of each value in {column_name}'
                    })
                
                # Find numeric columns for potential aggregation
                numeric_cols = df.select_dtypes(include=['number']).columns
                for num_col in numeric_cols:
                    suggestions.append({
                        'type': 'bar',
                        'title': f'Sum of {num_col} by {column_name}',
                        'columns': [column_name, num_col],
                        'description': f'Shows the total {num_col} for each {column_name} category'
                    })
        
        # For datetime columns
        elif pd.api.types.is_datetime64_any_dtype(column_dtype):
            # Time series line chart
            numeric_cols = df.select_dtypes(include=['number']).columns
            for num_col in numeric_cols:
                suggestions.append({
                    'type': 'line',
                    'title': f'{num_col} Over Time',
                    'columns': [column_name, num_col],
                    'description': f'Shows how {num_col} varies over time'
                })
                
                # Seasonal decomposition suggestion
                suggestions.append({
                    'type': 'seasonal',
                    'title': f'Seasonal Patterns in {num_col}',
                    'columns': [column_name, num_col],
                    'description': f'Decomposes the {num_col} time series into trend, seasonal, and residual components'
                })
    
    else:
        # General dataset visualization suggestions
        
        # Correlation matrix for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            suggestions.append({
                'type': 'heatmap',
                'title': 'Correlation Matrix',
                'columns': list(numeric_cols),
                'description': 'Displays the correlation between numeric variables'
            })
            
            # Scatter plot matrix if not too many columns
            if len(numeric_cols) <= 5:
                suggestions.append({
                    'type': 'scatter_matrix',
                    'title': 'Scatter Plot Matrix',
                    'columns': list(numeric_cols),
                    'description': 'Creates a matrix of scatter plots to visualize relationships between numeric variables'
                })
        
        # Bar charts for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for cat_col in categorical_cols:
            if df[cat_col].nunique() <= 20:  # Only suggest if cardinality is reasonable
                suggestions.append({
                    'type': 'bar',
                    'title': f'Frequency of {cat_col} Values',
                    'columns': [cat_col],
                    'description': f'Shows the count of occurrences for each value in {cat_col}'
                })
        
        # Recommendations for datetime columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        for datetime_col in datetime_cols:
            # Recommend time-based visualizations for each numeric column
            for num_col in numeric_cols:
                suggestions.append({
                    'type': 'line',
                    'title': f'{num_col} Over Time',
                    'columns': [datetime_col, num_col],
                    'description': f'Shows how {num_col} varies over time'
                })
        
        # Pair plot suggestion for two columns
        if len(numeric_cols) >= 2:
            num_col1 = numeric_cols[0]
            num_col2 = numeric_cols[1]
            suggestions.append({
                'type': 'scatter',
                'title': f'{num_col1} vs {num_col2}',
                'columns': [num_col1, num_col2],
                'description': f'Explores the relationship between {num_col1} and {num_col2}'
            })
    
    return suggestions

def create_chart(df: pd.DataFrame, chart_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a specified chart with the given configuration
    
    Args:
        df: The pandas dataframe
        chart_type: Type of chart to create
        config: Configuration for the chart
        
    Returns:
        Dictionary with chart figure and metadata
    """
    result = {
        'type': chart_type,
        'title': config.get('title', 'Chart'),
        'success': False,
        'error': None,
        'figure': None
    }
    
    try:
        # Extract configurations
        x = config.get('x')
        y = config.get('y')
        color = config.get('color')
        facet = config.get('facet')
        animation = config.get('animation')
        log_x = config.get('log_x', False)
        log_y = config.get('log_y', False)
        orientation = config.get('orientation', 'v')
        barmode = config.get('barmode', 'group')
        trendline = config.get('trendline')
        marginal = config.get('marginal')
        
        # Check column existence
        cols_to_check = [x, y, color, facet, animation]
        for col in cols_to_check:
            if col is not None and col not in df.columns:
                result['error'] = f"Column '{col}' not found in dataframe"
                return result
        
        # Create chart based on type
        if chart_type == 'bar':
            # Handle grouping if both x and y are provided
            if x and y:
                # If y is categorical and x is numeric, swap them and set horizontal
                if (pd.api.types.is_string_dtype(df[y].dtype) or 
                    pd.api.types.is_categorical_dtype(df[y].dtype)) and \
                   pd.api.types.is_numeric_dtype(df[x].dtype):
                    x, y = y, x
                    orientation = 'h'
                
                if orientation == 'h':
                    fig = px.bar(
                        df, y=x, x=y, color=color, facet_col=facet, 
                        animation_frame=animation, log_x=log_y, log_y=log_x,
                        barmode=barmode, title=result['title']
                    )
                else:
                    fig = px.bar(
                        df, x=x, y=y, color=color, facet_col=facet, 
                        animation_frame=animation, log_x=log_x, log_y=log_y,
                        barmode=barmode, title=result['title']
                    )
            else:
                # Simple bar chart for categorical column
                if x and not y:
                    counts = df[x].value_counts().reset_index()
                    counts.columns = ['value', 'count']
                    fig = px.bar(
                        counts, x='value', y='count', title=result['title'],
                        log_y=log_y
                    )
                else:
                    result['error'] = "Bar chart requires at least x column"
                    return result
        
        elif chart_type == 'line':
            if x and y:
                fig = px.line(
                    df, x=x, y=y, color=color, facet_col=facet, 
                    animation_frame=animation, log_x=log_x, log_y=log_y,
                    title=result['title']
                )
                
                # Add trendline if requested
                if trendline:
                    fig.update_layout(showlegend=True)
                    
                    # Only add trendline if both x and y are numeric
                    if pd.api.types.is_numeric_dtype(df[x].dtype) and pd.api.types.is_numeric_dtype(df[y].dtype):
                        mask = df[x].notna() & df[y].notna()
                        
                        if trendline == 'ols':
                            # Simple linear regression
                            from scipy import stats
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                df.loc[mask, x], df.loc[mask, y]
                            )
                            
                            # Create trend line
                            x_range = np.linspace(df[x].min(), df[x].max(), 100)
                            y_trend = slope * x_range + intercept
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=x_range, y=y_trend, mode='lines',
                                    name=f'Trend (R² = {r_value**2:.3f})',
                                    line=dict(color='red', dash='dash')
                                )
                            )
            else:
                result['error'] = "Line chart requires both x and y columns"
                return result
        
        elif chart_type == 'scatter':
            if x and y:
                fig = px.scatter(
                    df, x=x, y=y, color=color, facet_col=facet,
                    animation_frame=animation, log_x=log_x, log_y=log_y,
                    trendline=trendline, marginal_x=marginal, marginal_y=marginal,
                    title=result['title']
                )
            else:
                result['error'] = "Scatter plot requires both x and y columns"
                return result
        
        elif chart_type == 'histogram':
            if x:
                fig = px.histogram(
                    df, x=x, color=color, facet_col=facet,
                    animation_frame=animation, log_x=log_x, log_y=log_y,
                    marginal=marginal, title=result['title']
                )
            else:
                result['error'] = "Histogram requires x column"
                return result
        
        elif chart_type == 'box':
            if x or y:
                fig = px.box(
                    df, x=x, y=y, color=color, facet_col=facet,
                    animation_frame=animation, log_x=log_x, log_y=log_y,
                    title=result['title']
                )
            else:
                result['error'] = "Box plot requires either x or y column"
                return result
        
        elif chart_type == 'violin':
            if x or y:
                fig = px.violin(
                    df, x=x, y=y, color=color, facet_col=facet,
                    animation_frame=animation, log_x=log_x, log_y=log_y,
                    box=True, title=result['title']
                )
            else:
                result['error'] = "Violin plot requires either x or y column"
                return result
        
        elif chart_type == 'pie':
            if x and y:
                fig = px.pie(
                    df, names=x, values=y, color=x,
                    title=result['title']
                )
            elif x and not y:
                # Use value counts if only category is provided
                counts = df[x].value_counts().reset_index()
                counts.columns = ['value', 'count']
                fig = px.pie(
                    counts, names='value', values='count',
                    title=result['title']
                )
            else:
                result['error'] = "Pie chart requires at least x column"
                return result
        
        elif chart_type == 'heatmap':
            # For correlation matrix
            if config.get('correlation', False):
                # Get only numeric columns
                if x and y:
                    # Specific columns provided
                    corr_df = df[[x, y]].corr()
                else:
                    # All numeric columns
                    corr_df = df.select_dtypes(include=['number']).corr()
                
                # Create heatmap
                fig = px.imshow(
                    corr_df, text_auto=True, color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1, title=result['title']
                )
            # For pivot table heatmap
            elif x and y and color:
                # Create pivot table
                pivot_df = df.pivot_table(
                    index=x, columns=y, values=color,
                    aggfunc=config.get('aggfunc', 'mean')
                )
                
                # Create heatmap
                fig = px.imshow(
                    pivot_df, text_auto=True, color_continuous_scale='Viridis',
                    title=result['title']
                )
            else:
                result['error'] = "Heatmap requires proper configuration"
                return result
        
        elif chart_type == 'bubble':
            if x and y and size:
                size = config.get('size')
                fig = px.scatter(
                    df, x=x, y=y, size=size, color=color,
                    facet_col=facet, animation_frame=animation,
                    log_x=log_x, log_y=log_y, title=result['title']
                )
            else:
                result['error'] = "Bubble chart requires x, y, and size columns"
                return result
        
        elif chart_type == 'scatter_matrix':
            dimensions = config.get('dimensions', [])
            if dimensions:
                fig = px.scatter_matrix(
                    df, dimensions=dimensions, color=color,
                    title=result['title']
                )
            else:
                result['error'] = "Scatter matrix requires dimensions"
                return result
        
        elif chart_type == 'parallel_coordinates':
            dimensions = config.get('dimensions', [])
            if dimensions:
                fig = px.parallel_coordinates(
                    df, dimensions=dimensions, color=color,
                    color_continuous_scale=px.colors.diverging.Tealrose,
                    title=result['title']
                )
            else:
                result['error'] = "Parallel coordinates requires dimensions"
                return result
        
        elif chart_type == 'contour':
            if x and y and z:
                z = config.get('z')
                fig = px.density_contour(
                    df, x=x, y=y, z=z, color=color,
                    marginal_x='histogram', marginal_y='histogram',
                    title=result['title']
                )
            else:
                result['error'] = "Contour plot requires x, y, and z columns"
                return result
        
        elif chart_type == 'sunburst':
            path = config.get('path', [])
            if path:
                values = config.get('values')
                fig = px.sunburst(
                    df, path=path, values=values, color=color,
                    title=result['title']
                )
            else:
                result['error'] = "Sunburst chart requires path"
                return result
        
        elif chart_type == 'treemap':
            path = config.get('path', [])
            if path:
                values = config.get('values')
                fig = px.treemap(
                    df, path=path, values=values, color=color,
                    title=result['title']
                )
            else:
                result['error'] = "Treemap requires path"
                return result
        
        elif chart_type == 'funnel':
            if x and y:
                # Sort data for funnel
                sorted_df = df.sort_values(y, ascending=False)
                
                fig = go.Figure(go.Funnel(
                    x=sorted_df[y],
                    y=sorted_df[x],
                    textinfo="value+percent initial"
                ))
                
                fig.update_layout(title=result['title'])
            else:
                result['error'] = "Funnel chart requires both x and y columns"
                return result
        
        elif chart_type == 'area':
            if x and y:
                fig = px.area(
                    df, x=x, y=y, color=color, facet_col=facet,
                    animation_frame=animation, log_x=log_x, log_y=log_y,
                    title=result['title']
                )
            else:
                result['error'] = "Area chart requires both x and y columns"
                return result
        
        elif chart_type == 'density':
            if x:
                hist_data = [df[x].dropna()]
                group_labels = [x]
                
                if color and df[color].nunique() <= 10:
                    # Split by color if categorical with reasonable number of categories
                    hist_data = []
                    group_labels = []
                    
                    for group, group_df in df.groupby(color):
                        if not group_df[x].empty:
                            hist_data.append(group_df[x].dropna())
                            group_labels.append(str(group))
                
                if hist_data:
                    fig = ff.create_distplot(
                        hist_data, group_labels, curve_type='kde',
                        show_hist=False, show_rug=True
                    )
                    fig.update_layout(title=result['title'])
                else:
                    result['error'] = "No valid data for density plot"
                    return result
            else:
                result['error'] = "Density plot requires x column"
                return result
        
        elif chart_type == 'radar':
            if not isinstance(df, pd.DataFrame):
                result['error'] = "Radar chart requires a preprocessed dataframe"
                return result
            
            categories = df.columns.tolist()
            values = df.values.tolist()
            
            fig = go.Figure()
            
            for i, row in enumerate(values):
                fig.add_trace(go.Scatterpolar(
                    r=row,
                    theta=categories,
                    fill='toself',
                    name=f'Series {i+1}'
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True),
                ),
                title=result['title']
            )
        
        elif chart_type == 'waterfall':
            if x and y:
                # Create base for waterfall chart
                measure = ['absolute'] + ['relative'] * (len(df) - 2) + ['total']
                x_data = df[x].tolist() + ['Total']
                y_data = df[y].tolist() + [df[y].sum()]
                
                fig = go.Figure(go.Waterfall(
                    name=result['title'],
                    orientation='v',
                    measure=measure,
                    x=x_data,
                    y=y_data,
                    textposition='outside',
                    text=y_data,
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                ))
                
                fig.update_layout(title=result['title'])
            else:
                result['error'] = "Waterfall chart requires both x and y columns"
                return result
        
        else:
            result['error'] = f"Chart type '{chart_type}' not supported"
            return result
        
        # Common layout adjustments
        fig.update_layout(
            title={
                'text': result['title'],
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        
        # Store the figure in the result
        result['figure'] = fig
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def create_dashboard(charts: List[Dict[str, Any]], title: str = "Dashboard") -> Dict[str, Any]:
    """
    Create a dashboard with multiple charts
    
    Args:
        charts: List of chart dictionaries with figures
        title: Dashboard title
        
    Returns:
        Dictionary with dashboard figure and metadata
    """
    result = {
        'title': title,
        'success': False,
        'error': None,
        'figure': None
    }
    
    try:
        # Filter out failed charts
        valid_charts = [c for c in charts if c.get('success', False) and c.get('figure') is not None]
        
        if not valid_charts:
            result['error'] = "No valid charts to display"
            return result
        
        # Determine grid layout
        n_charts = len(valid_charts)
        if n_charts == 1:
            rows, cols = 1, 1
        elif n_charts == 2:
            rows, cols = 1, 2
        elif n_charts <= 4:
            rows, cols = 2, 2
        elif n_charts <= 6:
            rows, cols = 2, 3
        elif n_charts <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = (n_charts + 2) // 3, 3  # Rounded up to multiple of 3
        
        # Create subplots
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[c.get('title', f"Chart {i+1}") for i, c in enumerate(valid_charts)]
        )
        
        # Add each chart to the grid
        for i, chart in enumerate(valid_charts):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            chart_fig = chart.get('figure')
            
            # Extract traces from the chart figure
            for trace in chart_fig.data:
                fig.add_trace(trace, row=row, col=col)
            
            # Copy layout properties for each subplot
            # This is a simplified approach - not all layout properties will transfer correctly
            for axis_type in ['xaxis', 'yaxis']:
                for property_suffix in ['type', 'title', 'range']:
                    property_name = f"{axis_type}.{property_suffix}"
                    if property_name in chart_fig.layout:
                        fig.update_layout({
                            f"{axis_type}{i+1}.{property_suffix}": chart_fig.layout[property_name]
                        })
        
        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            height=300 * rows,  # Adjust height based on number of rows
            width=400 * cols,   # Adjust width based on number of columns
            showlegend=False,   # Hide legend by default to save space
        )
        
        # Store the figure in the result
        result['figure'] = fig
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def create_comparison_chart(df1: pd.DataFrame, df2: pd.DataFrame, 
                            chart_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a chart comparing two datasets
    
    Args:
        df1: First dataframe
        df2: Second dataframe
        chart_type: Type of chart to create
        config: Configuration for the chart
        
    Returns:
        Dictionary with comparison chart figure and metadata
    """
    result = {
        'type': 'comparison',
        'title': config.get('title', 'Comparison Chart'),
        'success': False,
        'error': None,
        'figure': None
    }
    
    try:
        # Extract configurations
        x = config.get('x')
        y = config.get('y')
        
        # Check column existence
        for df, name in [(df1, 'First dataset'), (df2, 'Second dataset')]:
            if x is not None and x not in df.columns:
                result['error'] = f"{name}: Column '{x}' not found"
                return result
            if y is not None and y not in df.columns:
                result['error'] = f"{name}: Column '{y}' not found"
                return result
        
        if chart_type == 'bar':
            # Create grouped bar chart for comparison
            if x and y:
                # Create figure
                fig = go.Figure()
                
                # Add bars for first dataset
                fig.add_trace(go.Bar(
                    x=df1[x],
                    y=df1[y],
                    name='Dataset 1',
                    marker_color='rgb(55, 83, 109)'
                ))
                
                # Add bars for second dataset
                fig.add_trace(go.Bar(
                    x=df2[x],
                    y=df2[y],
                    name='Dataset 2',
                    marker_color='rgb(26, 118, 255)'
                ))
                
                # Update layout
                fig.update_layout(
                    title=result['title'],
                    xaxis_title=x,
                    yaxis_title=y,
                    barmode='group',
                    bargap=0.15,
                    bargroupgap=0.1
                )
            else:
                result['error'] = "Bar comparison requires both x and y columns"
                return result
        
        elif chart_type == 'line':
            # Create overlaid line chart for comparison
            if x and y:
                # Create figure
                fig = go.Figure()
                
                # Add line for first dataset
                fig.add_trace(go.Scatter(
                    x=df1[x],
                    y=df1[y],
                    name='Dataset 1',
                    mode='lines+markers',
                    line=dict(color='rgb(55, 83, 109)', width=2)
                ))
                
                # Add line for second dataset
                fig.add_trace(go.Scatter(
                    x=df2[x],
                    y=df2[y],
                    name='Dataset 2',
                    mode='lines+markers',
                    line=dict(color='rgb(26, 118, 255)', width=2)
                ))
                
                # Update layout
                fig.update_layout(
                    title=result['title'],
                    xaxis_title=x,
                    yaxis_title=y
                )
            else:
                result['error'] = "Line comparison requires both x and y columns"
                return result
        
        elif chart_type == 'scatter':
            # Create scatter plot with two datasets
            if x and y:
                # Create figure
                fig = go.Figure()
                
                # Add scatter for first dataset
                fig.add_trace(go.Scatter(
                    x=df1[x],
                    y=df1[y],
                    name='Dataset 1',
                    mode='markers',
                    marker=dict(
                        color='rgb(55, 83, 109)',
                        size=10,
                        line=dict(
                            color='rgb(40, 60, 80)',
                            width=1
                        )
                    )
                ))
                
                # Add scatter for second dataset
                fig.add_trace(go.Scatter(
                    x=df2[x],
                    y=df2[y],
                    name='Dataset 2',
                    mode='markers',
                    marker=dict(
                        color='rgb(26, 118, 255)',
                        size=10,
                        line=dict(
                            color='rgb(15, 80, 180)',
                            width=1
                        )
                    )
                ))
                
                # Update layout
                fig.update_layout(
                    title=result['title'],
                    xaxis_title=x,
                    yaxis_title=y
                )
            else:
                result['error'] = "Scatter comparison requires both x and y columns"
                return result
        
        elif chart_type == 'box':
            # Create box plot comparison
            if y:
                # Create figure
                fig = go.Figure()
                
                # Add box for first dataset
                fig.add_trace(go.Box(
                    y=df1[y],
                    name='Dataset 1',
                    marker_color='rgb(55, 83, 109)',
                    boxmean=True
                ))
                
                # Add box for second dataset
                fig.add_trace(go.Box(
                    y=df2[y],
                    name='Dataset 2',
                    marker_color='rgb(26, 118, 255)',
                    boxmean=True
                ))
                
                # Update layout
                fig.update_layout(
                    title=result['title'],
                    yaxis_title=y
                )
            else:
                result['error'] = "Box comparison requires y column"
                return result
        
        elif chart_type == 'histogram':
            # Create overlaid histogram comparison
            if x:
                # Create figure
                fig = go.Figure()
                
                # Add histogram for first dataset
                fig.add_trace(go.Histogram(
                    x=df1[x],
                    name='Dataset 1',
                    marker_color='rgb(55, 83, 109)',
                    opacity=0.75
                ))
                
                # Add histogram for second dataset
                fig.add_trace(go.Histogram(
                    x=df2[x],
                    name='Dataset 2',
                    marker_color='rgb(26, 118, 255)',
                    opacity=0.75
                ))
                
                # Update layout
                fig.update_layout(
                    title=result['title'],
                    xaxis_title=x,
                    barmode='overlay'
                )
            else:
                result['error'] = "Histogram comparison requires x column"
                return result
        
        elif chart_type == 'pie':
            # Create side-by-side pie charts
            if x:
                # Create figure with subplots
                fig = make_subplots(
                    rows=1, cols=2,
                    specs=[[{'type': 'domain'}, {'type': 'domain'}]],
                    subplot_titles=['Dataset 1', 'Dataset 2']
                )
                
                # Add pie for first dataset
                values1 = df1[x].value_counts().values
                labels1 = df1[x].value_counts().index
                
                fig.add_trace(go.Pie(
                    labels=labels1,
                    values=values1,
                    name='Dataset 1'
                ), 1, 1)
                
                # Add pie for second dataset
                values2 = df2[x].value_counts().values
                labels2 = df2[x].value_counts().index
                
                fig.add_trace(go.Pie(
                    labels=labels2,
                    values=values2,
                    name='Dataset 2'
                ), 1, 2)
                
                # Update layout
                fig.update_layout(title=result['title'])
            else:
                result['error'] = "Pie comparison requires x column"
                return result
        
        elif chart_type == 'heatmap':
            # Create side-by-side heatmaps for correlation matrices
            # Get only numeric columns
            numeric_cols1 = df1.select_dtypes(include=['number']).columns
            numeric_cols2 = df2.select_dtypes(include=['number']).columns
            
            if not numeric_cols1.empty and not numeric_cols2.empty:
                # Create correlation matrices
                corr1 = df1[numeric_cols1].corr()
                corr2 = df2[numeric_cols2].corr()
                
                # Create figure with subplots
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=['Dataset 1 Correlation', 'Dataset 2 Correlation']
                )
                
                # Add heatmap for first dataset
                fig.add_trace(go.Heatmap(
                    z=corr1.values,
                    x=corr1.columns,
                    y=corr1.index,
                    colorscale='RdBu_r',
                    zmin=-1, zmax=1
                ), 1, 1)
                
                # Add heatmap for second dataset
                fig.add_trace(go.Heatmap(
                    z=corr2.values,
                    x=corr2.columns,
                    y=corr2.index,
                    colorscale='RdBu_r',
                    zmin=-1, zmax=1
                ), 1, 2)
                
                # Update layout
                fig.update_layout(title=result['title'])
            else:
                result['error'] = "Heatmap comparison requires numeric columns in both datasets"
                return result
        
        else:
            result['error'] = f"Chart type '{chart_type}' not supported for comparison"
            return result
        
        # Store the figure in the result
        result['figure'] = fig
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def create_interactive_chart(df: pd.DataFrame, chart_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an interactive chart with additional controls
    
    Args:
        df: The pandas dataframe
        chart_type: Type of chart to create
        config: Configuration for the chart
        
    Returns:
        Dictionary with interactive chart figure and metadata
    """
    # First create the base chart
    result = create_chart(df, chart_type, config)
    
    # If the base chart creation failed, return immediately
    if not result['success']:
        return result
    
    fig = result['figure']
    
    # Add interactive elements based on chart type
    if chart_type in ['scatter', 'line', 'bar']:
        # Add range slider for x-axis
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='auto'
            )
        )
        
        # Add buttons for showing/hiding trend lines if it's a scatter plot
        if chart_type == 'scatter':
            x = config.get('x')
            y = config.get('y')
            
            if (x and y and 
                pd.api.types.is_numeric_dtype(df[x].dtype) and 
                pd.api.types.is_numeric_dtype(df[y].dtype)):
                
                # Create trend line data
                mask = df[x].notna() & df[y].notna()
                
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df.loc[mask, x], df.loc[mask, y]
                )
                
                x_range = np.linspace(df[x].min(), df[x].max(), 100)
                y_trend = slope * x_range + intercept
                
                # Add the trend line as a hidden trace
                fig.add_trace(
                    go.Scatter(
                        x=x_range, y=y_trend, mode='lines',
                        name=f'Trend (R² = {r_value**2:.3f})',
                        line=dict(color='red', dash='dash'),
                        visible='legendonly'  # Hidden by default
                    )
                )
    
    # For histograms, add buttons to change binning
    elif chart_type == 'histogram':
        x = config.get('x')
        
        if x and pd.api.types.is_numeric_dtype(df[x].dtype):
            fig.update_layout(
                updatemenus=[
                    dict(
                        buttons=list([
                            dict(
                                args=[{'nbinsx': 10}],
                                label="10 Bins",
                                method="relayout"
                            ),
                            dict(
                                args=[{'nbinsx': 20}],
                                label="20 Bins",
                                method="relayout"
                            ),
                            dict(
                                args=[{'nbinsx': 50}],
                                label="50 Bins",
                                method="relayout"
                            )
                        ]),
                        direction="down",
                        showactive=True,
                        x=0.1,
                        y=1.15
                    )
                ]
            )
    
    # For box plots, add toggle for points/notches
    elif chart_type == 'box':
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=[{'boxpoints': 'outliers'}],
                            label="Show Outliers Only",
                            method="restyle"
                        ),
                        dict(
                            args=[{'boxpoints': 'all'}],
                            label="Show All Points",
                            method="restyle"
                        ),
                        dict(
                            args=[{'boxpoints': False}],
                            label="Hide Points",
                            method="restyle"
                        )
                    ]),
                    direction="down",
                    showactive=True,
                    x=0.1,
                    y=1.15
                )
            ]
        )
    
    # Update the figure in the result
    result['figure'] = fig
    
    return result
