import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta
import re

class Visualization:
    """Class for generating data visualizations"""
    
    @staticmethod
    def suggest_visualizations(df):
        """
        Suggest appropriate visualizations based on data types
        
        Args:
            df (pandas.DataFrame): The dataframe to analyze
            
        Returns:
            list: List of suggested visualizations with metadata
        """
        if df is None or df.empty:
            return []
        
        suggestions = []
        
        # Get column types
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Add datetime-like columns that are stored as objects
        for col in df.columns:
            if col in datetime_cols:
                continue
                
            if 'date' in col.lower() or 'time' in col.lower():
                sample = df[col].dropna().head(5)
                try:
                    pd.to_datetime(sample)
                    datetime_cols.append(col)
                    if col in categorical_cols:
                        categorical_cols.remove(col)
                except:
                    pass
        
        # 1. Distribution of numeric columns
        for col in numeric_cols[:5]:  # Limit to first 5 to avoid too many suggestions
            suggestions.append({
                'type': 'histogram',
                'title': f'Distribution of {col}',
                'description': f'View the distribution pattern of {col} values',
                'columns': [col],
                'complexity': 'simple'
            })
        
        # 2. Bar charts for categorical columns
        for col in categorical_cols[:5]:  # Limit to first 5
            if df[col].nunique() < 30:  # Only for reasonably small number of categories
                suggestions.append({
                    'type': 'bar',
                    'title': f'Count of {col}',
                    'description': f'Compare the frequency of each {col} category',
                    'columns': [col],
                    'complexity': 'simple'
                })
        
        # 3. Scatter plots for pairs of numeric columns
        if len(numeric_cols) >= 2:
            # Suggest up to 3 scatter plots
            for i in range(min(3, len(numeric_cols) - 1)):
                for j in range(i + 1, min(i + 4, len(numeric_cols))):
                    suggestions.append({
                        'type': 'scatter',
                        'title': f'{numeric_cols[i]} vs {numeric_cols[j]}',
                        'description': f'Explore relationship between {numeric_cols[i]} and {numeric_cols[j]}',
                        'columns': [numeric_cols[i], numeric_cols[j]],
                        'complexity': 'simple'
                    })
        
        # 4. Time series for datetime columns with numeric values
        if datetime_cols and numeric_cols:
            for date_col in datetime_cols[:2]:  # Limit to first 2 datetime columns
                for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    suggestions.append({
                        'type': 'line',
                        'title': f'{num_col} over {date_col}',
                        'description': f'Track changes in {num_col} over time',
                        'columns': [date_col, num_col],
                        'complexity': 'simple'
                    })
        
        # 5. Box plots for numeric columns grouped by categorical
        if numeric_cols and categorical_cols:
            for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                for cat_col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                    if df[cat_col].nunique() < 10:  # Only for small number of categories
                        suggestions.append({
                            'type': 'box',
                            'title': f'{num_col} by {cat_col}',
                            'description': f'Compare distribution of {num_col} across {cat_col} categories',
                            'columns': [cat_col, num_col],
                            'complexity': 'medium'
                        })
        
        # 6. Correlation heatmap for numeric columns
        if len(numeric_cols) > 2:
            suggestions.append({
                'type': 'heatmap',
                'title': 'Correlation Matrix',
                'description': 'Visualize correlations between numeric variables',
                'columns': numeric_cols[:10],  # Limit to 10 columns for readability
                'complexity': 'advanced'
            })
        
        # 7. Pie charts for categorical columns with few unique values
        for col in categorical_cols:
            if 2 <= df[col].nunique() <= 8:  # Ideal for pie charts
                suggestions.append({
                    'type': 'pie',
                    'title': f'Proportion of {col}',
                    'description': f'See the relative proportion of each {col} category',
                    'columns': [col],
                    'complexity': 'simple'
                })
        
        # 8. Grouped bar charts
        if len(categorical_cols) >= 2:
            for i in range(min(2, len(categorical_cols))):
                for j in range(i + 1, min(i + 2, len(categorical_cols))):
                    if df[categorical_cols[i]].nunique() < 10 and df[categorical_cols[j]].nunique() < 10:
                        suggestions.append({
                            'type': 'bar_grouped',
                            'title': f'{categorical_cols[i]} by {categorical_cols[j]}',
                            'description': f'Compare counts across {categorical_cols[i]} and {categorical_cols[j]}',
                            'columns': [categorical_cols[i], categorical_cols[j]],
                            'complexity': 'medium'
                        })
        
        # 9. Area charts for time series
        if datetime_cols and numeric_cols:
            for date_col in datetime_cols[:1]:  # Limit to first datetime column
                for num_col in numeric_cols[:2]:  # Limit to first 2 numeric columns
                    suggestions.append({
                        'type': 'area',
                        'title': f'{num_col} Trend over {date_col}',
                        'description': f'Visualize trends and cumulative patterns in {num_col} over time',
                        'columns': [date_col, num_col],
                        'complexity': 'medium'
                    })
        
        # 10. Violin plots
        if numeric_cols and categorical_cols:
            for num_col in numeric_cols[:2]:  # Limit to first 2 numeric columns
                for cat_col in categorical_cols[:2]:  # Limit to first 2 categorical columns
                    if df[cat_col].nunique() < 8:  # Only for small number of categories
                        suggestions.append({
                            'type': 'violin',
                            'title': f'Distribution of {num_col} by {cat_col}',
                            'description': f'Detailed distribution comparison of {num_col} across {cat_col} categories',
                            'columns': [cat_col, num_col],
                            'complexity': 'advanced'
                        })
        
        # 11. Bubble charts
        if len(numeric_cols) >= 3:
            suggestions.append({
                'type': 'bubble',
                'title': f'Bubble Chart of {numeric_cols[0]}, {numeric_cols[1]}, and {numeric_cols[2]}',
                'description': f'Compare three metrics: {numeric_cols[0]} (x), {numeric_cols[1]} (y), and {numeric_cols[2]} (size)',
                'columns': [numeric_cols[0], numeric_cols[1], numeric_cols[2]],
                'complexity': 'advanced'
            })
        
        # Add dashboard suggestions if we have enough variety of visualizations
        if len(suggestions) >= 5:
            # Basic dashboard
            dashboard_charts = [
                s for s in suggestions 
                if s['complexity'] == 'simple' and s['type'] in ['bar', 'histogram', 'line', 'pie']
            ][:4]
            
            if len(dashboard_charts) >= 3:
                suggestions.append({
                    'type': 'dashboard_basic',
                    'title': 'Basic Data Overview Dashboard',
                    'description': 'A simple dashboard with key charts for data overview',
                    'components': [d['title'] for d in dashboard_charts],
                    'complexity': 'medium'
                })
            
            # Advanced dashboard if we have more complex charts
            advanced_charts = [
                s for s in suggestions 
                if s['complexity'] in ['medium', 'advanced'] and s['type'] in ['heatmap', 'box', 'violin', 'scatter']
            ][:3]
            
            time_charts = [
                s for s in suggestions 
                if 'over time' in s['description'].lower() or 'trend' in s['description'].lower()
            ][:2]
            
            if len(advanced_charts) >= 2 and time_charts:
                components = [d['title'] for d in advanced_charts + time_charts]
                suggestions.append({
                    'type': 'dashboard_advanced',
                    'title': 'Advanced Analytics Dashboard',
                    'description': 'A comprehensive dashboard with advanced visualizations for deeper analysis',
                    'components': components,
                    'complexity': 'advanced'
                })
        
        return suggestions
    
    @staticmethod
    def create_visualization(df, viz_type, **kwargs):
        """
        Create a visualization based on type and parameters
        
        Args:
            df (pandas.DataFrame): The dataframe to visualize
            viz_type (str): Type of visualization
            **kwargs: Additional parameters
            
        Returns:
            plotly.graph_objects.Figure: The created visualization
        """
        if df is None or df.empty:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Get required parameters
        title = kwargs.get('title', 'Visualization')
        x_col = kwargs.get('x')
        y_col = kwargs.get('y')
        color_col = kwargs.get('color')
        
        # Histogram
        if viz_type == 'histogram':
            if not x_col:
                return None
                
            fig = px.histogram(
                df, x=x_col, 
                title=title,
                color=color_col,
                marginal=kwargs.get('marginal', 'box'),
                nbins=kwargs.get('nbins', 30),
                opacity=kwargs.get('opacity', 0.7),
                histnorm=kwargs.get('histnorm')
            )
            
            # Add density curve if requested
            if kwargs.get('density', False) and df[x_col].nunique() > 10:
                try:
                    from scipy import stats
                    kde_x = np.linspace(df[x_col].min(), df[x_col].max(), 1000)
                    kde = stats.gaussian_kde(df[x_col].dropna())
                    kde_y = kde(kde_x)
                    fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode='lines', name='Density',
                                            line=dict(color='red')))
                except:
                    pass
        
        # Bar chart
        elif viz_type == 'bar':
            if not x_col:
                return None
                
            # For categorical columns, get value counts
            if not y_col:
                value_counts = df[x_col].value_counts().reset_index()
                value_counts.columns = [x_col, 'count']
                
                fig = px.bar(
                    value_counts, x=x_col, y='count',
                    title=title,
                    color=x_col if color_col is None else color_col,
                    labels={'count': 'Count'},
                    text='count' if kwargs.get('show_values', False) else None
                )
            else:
                # If y column specified, use it for values
                fig = px.bar(
                    df, x=x_col, y=y_col,
                    title=title,
                    color=color_col,
                    barmode=kwargs.get('barmode', 'group'),
                    text=y_col if kwargs.get('show_values', False) else None
                )
        
        # Scatter plot
        elif viz_type == 'scatter':
            if not x_col or not y_col:
                return None
                
            fig = px.scatter(
                df, x=x_col, y=y_col,
                title=title,
                color=color_col,
                size=kwargs.get('size'),
                hover_data=kwargs.get('hover_data'),
                trendline=kwargs.get('trendline'),
                opacity=kwargs.get('opacity', 0.7)
            )
        
        # Line chart
        elif viz_type == 'line':
            if not x_col or not y_col:
                return None
                
            # Convert to datetime if it's a string date
            if df[x_col].dtype == 'object':
                try:
                    df = df.copy()
                    df[x_col] = pd.to_datetime(df[x_col])
                except:
                    pass
            
            fig = px.line(
                df, x=x_col, y=y_col,
                title=title,
                color=color_col,
                markers=kwargs.get('markers', True),
                line_shape=kwargs.get('line_shape', 'linear')
            )
        
        # Box plot
        elif viz_type == 'box':
            if not x_col or not y_col:
                return None
                
            fig = px.box(
                df, x=x_col, y=y_col,
                title=title,
                color=color_col,
                notched=kwargs.get('notched', False),
                points=kwargs.get('points', 'outliers')
            )
        
        # Heatmap (correlation matrix)
        elif viz_type == 'heatmap':
            # If columns specified, use only those columns
            columns = kwargs.get('columns')
            if columns:
                corr_df = df[columns].corr()
            else:
                # Otherwise use all numeric columns
                corr_df = df.select_dtypes(include=np.number).corr()
            
            fig = px.imshow(
                corr_df,
                text_auto=kwargs.get('text_auto', True),
                color_continuous_scale=kwargs.get('color_scale', 'RdBu_r'),
                title=title,
                zmin=-1, zmax=1,
                aspect='auto'
            )
        
        # Pie chart
        elif viz_type == 'pie':
            if not x_col:
                return None
                
            # Get value counts
            value_counts = df[x_col].value_counts().reset_index()
            value_counts.columns = [x_col, 'count']
            
            fig = px.pie(
                value_counts, names=x_col, values='count',
                title=title,
                hole=kwargs.get('hole', 0.3),
                color_discrete_sequence=px.colors.qualitative.Set3
            )
        
        # Area chart
        elif viz_type == 'area':
            if not x_col or not y_col:
                return None
                
            # Convert to datetime if it's a string date
            if df[x_col].dtype == 'object':
                try:
                    df = df.copy()
                    df[x_col] = pd.to_datetime(df[x_col])
                except:
                    pass
            
            fig = px.area(
                df, x=x_col, y=y_col,
                title=title,
                color=color_col,
                line_shape=kwargs.get('line_shape', 'linear')
            )
        
        # Violin plot
        elif viz_type == 'violin':
            if not x_col or not y_col:
                return None
                
            fig = px.violin(
                df, x=x_col, y=y_col,
                title=title,
                color=color_col,
                box=kwargs.get('box', True),
                points=kwargs.get('points', 'outliers')
            )
        
        # Bubble chart
        elif viz_type == 'bubble':
            if not x_col or not y_col or not kwargs.get('size'):
                return None
                
            fig = px.scatter(
                df, x=x_col, y=y_col,
                title=title,
                color=color_col,
                size=kwargs.get('size'),
                hover_data=kwargs.get('hover_data'),
                opacity=kwargs.get('opacity', 0.7)
            )
        
        # Grouped bar chart
        elif viz_type == 'bar_grouped':
            if not x_col or not color_col:
                return None
                
            # Create cross-tabulation
            crosstab = pd.crosstab(df[x_col], df[color_col])
            
            # Melt the crosstab for plotting
            melted = crosstab.reset_index().melt(id_vars=x_col, var_name=color_col, value_name='count')
            
            fig = px.bar(
                melted, x=x_col, y='count', color=color_col,
                title=title,
                barmode='group',
                text='count' if kwargs.get('show_values', False) else None
            )
        
        # Sunburst chart
        elif viz_type == 'sunburst':
            path = kwargs.get('path')
            if not path:
                return None
                
            fig = px.sunburst(
                df, path=path,
                values=kwargs.get('values'),
                title=title,
                color=color_col,
                maxdepth=kwargs.get('maxdepth', -1)
            )
        
        # Map visualization
        elif viz_type == 'map':
            lat_col = kwargs.get('lat')
            lon_col = kwargs.get('lon')
            if not lat_col or not lon_col:
                return None
                
            fig = px.scatter_mapbox(
                df, lat=lat_col, lon=lon_col,
                title=title,
                color=color_col,
                size=kwargs.get('size'),
                hover_name=kwargs.get('hover_name'),
                zoom=kwargs.get('zoom', 3),
                mapbox_style='carto-positron'
            )
        
        # Stacked area chart
        elif viz_type == 'area_stacked':
            if not x_col or not y_col or not color_col:
                return None
                
            # Convert to datetime if it's a string date
            if df[x_col].dtype == 'object':
                try:
                    df = df.copy()
                    df[x_col] = pd.to_datetime(df[x_col])
                except:
                    pass
            
            fig = px.area(
                df, x=x_col, y=y_col,
                title=title,
                color=color_col,
                line_shape=kwargs.get('line_shape', 'linear'),
                groupnorm=kwargs.get('groupnorm')
            )
        
        # Radar chart
        elif viz_type == 'radar':
            if not kwargs.get('categories') or not kwargs.get('values'):
                return None
                
            categories = kwargs.get('categories')
            values = kwargs.get('values')
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=kwargs.get('name', 'Radar Chart')
            ))
            
            fig.update_layout(
                title=title,
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, kwargs.get('max_value', max(values) * 1.2)]
                    )
                )
            )
        
        # Funnel chart
        elif viz_type == 'funnel':
            if not x_col or not y_col:
                return None
                
            # Sort by values for funnel effect
            df_sorted = df.sort_values(y_col, ascending=False)
            
            fig = go.Figure(go.Funnel(
                y=df_sorted[x_col],
                x=df_sorted[y_col],
                textinfo=kwargs.get('textinfo', 'value+percent initial')
            ))
            
            fig.update_layout(title=title)
        
        # Default to None if visualization type not found
        else:
            return None
        
        # Update layout with common settings
        fig.update_layout(
            title_x=0.5,
            margin=dict(l=20, r=20, t=50, b=20),
            height=kwargs.get('height', 400),
            width=kwargs.get('width', None)
        )
        
        return fig
    
    @staticmethod
    def create_dashboard(df, components):
        """
        Create a multi-chart dashboard
        
        Args:
            df (pandas.DataFrame): The dataframe to visualize
            components (list): List of visualization specifications
            
        Returns:
            plotly.graph_objects.Figure: The created dashboard
        """
        if df is None or df.empty or not components:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for dashboard",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Determine grid layout based on number of components
        n_components = len(components)
        
        if n_components <= 2:
            rows, cols = 1, n_components
        elif n_components <= 4:
            rows, cols = 2, 2
        elif n_components <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        # Create subplot grid
        subplot_titles = [comp.get('title', f'Chart {i+1}') for i, comp in enumerate(components)]
        fig = make_subplots(
            rows=rows, cols=cols, 
            subplot_titles=subplot_titles,
            specs=[[{"type": "xy" if comp.get('type') not in ['pie', 'sunburst'] else comp.get('type')} 
                   for comp in components[:cols]] for _ in range(rows)]
        )
        
        # Add each component to dashboard
        for i, comp in enumerate(components):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            viz_type = comp.get('type')
            
            # Create the individual visualization
            sub_fig = Visualization.create_visualization(df, viz_type, **comp)
            
            if sub_fig:
                # Add traces from sub_fig to the main figure
                for trace in sub_fig.data:
                    fig.add_trace(trace, row=row, col=col)
                
                # Update x-axis title if present
                if hasattr(sub_fig.layout, "xaxis") and hasattr(sub_fig.layout.xaxis, "title") and hasattr(sub_fig.layout.xaxis.title, "text"):
                    fig.update_xaxes(title_text=sub_fig.layout.xaxis.title.text, row=row, col=col)
                # Update y-axis title if present
                if hasattr(sub_fig.layout, "yaxis") and hasattr(sub_fig.layout.yaxis, "title") and hasattr(sub_fig.layout.yaxis.title, "text"):
                    fig.update_yaxes(title_text=sub_fig.layout.yaxis.title.text, row=row, col=col)
                # Optionally, update axis range and tickformat if needed
                if hasattr(sub_fig.layout, "xaxis") and hasattr(sub_fig.layout.xaxis, "range"):
                    fig.update_xaxes(range=sub_fig.layout.xaxis.range, row=row, col=col)
                if hasattr(sub_fig.layout, "xaxis") and hasattr(sub_fig.layout.xaxis, "tickformat"):
                    fig.update_xaxes(tickformat=sub_fig.layout.xaxis.tickformat, row=row, col=col)
                if hasattr(sub_fig.layout, "yaxis") and hasattr(sub_fig.layout.yaxis, "range"):
                    fig.update_yaxes(range=sub_fig.layout.yaxis.range, row=row, col=col)
                if hasattr(sub_fig.layout, "yaxis") and hasattr(sub_fig.layout.yaxis, "tickformat"):
                    fig.update_yaxes(tickformat=sub_fig.layout.yaxis.tickformat, row=row, col=col)
        
        # Update overall layout
        fig.update_layout(
            height=250 * rows,
            title_text="Data Analytics Dashboard",
            showlegend=False,
            title_x=0.5,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig
    
    @staticmethod
    def generate_insight_visualizations(df, insights):
        """
        Generate visualizations for specific insights
        
        Args:
            df (pandas.DataFrame): The dataframe to visualize
            insights (list): List of insights
            
        Returns:
            dict: Dictionary of figures for each insight
        """
        if df is None or df.empty or not insights:
            return {}
        
        viz_dict = {}
        
        for i, insight in enumerate(insights):
            insight_type = insight.get('type')
            
            # Distribution insight
            if insight_type == 'distribution':
                column = insight.get('column')
                if not column or column not in df.columns:
                    continue
                
                if pd.api.types.is_numeric_dtype(df[column]):
                    # Numeric distribution
                    fig = px.histogram(
                        df, x=column,
                        title=f"Distribution of {column}",
                        marginal="box",
                        histnorm='probability density'
                    )
                    
                    # Add KDE curve if we have scipy
                    try:
                        from scipy import stats
                        kde_x = np.linspace(df[column].min(), df[column].max(), 1000)
                        kde = stats.gaussian_kde(df[column].dropna())
                        kde_y = kde(kde_x)
                        fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode='lines', name='Density',
                                                line=dict(color='red')))
                    except:
                        pass
                else:
                    # Categorical distribution
                    value_counts = df[column].value_counts().reset_index()
                    value_counts.columns = [column, 'count']
                    
                    fig = px.bar(
                        value_counts, x=column, y='count',
                        title=f"Distribution of {column}",
                        color=column
                    )
                
                viz_dict[f'insight_{i}'] = fig
            
            # Correlation insight
            elif insight_type == 'correlation':
                columns = insight.get('columns', [])
                if not columns or not all(col in df.columns for col in columns):
                    continue
                
                # Create scatter plot if it's two numeric columns
                if len(columns) == 2 and all(pd.api.types.is_numeric_dtype(df[col]) for col in columns):
                    fig = px.scatter(
                        df, x=columns[0], y=columns[1],
                        title=f"Correlation: {columns[0]} vs {columns[1]}",
                        trendline="ols"
                    )
                    
                    viz_dict[f'insight_{i}'] = fig
                    
                # Create heatmap for multiple columns
                elif len(columns) > 2:
                    corr_df = df[columns].corr()
                    
                    fig = px.imshow(
                        corr_df,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        title=f"Correlation Matrix for Selected Variables",
                        zmin=-1, zmax=1
                    )
                    
                    viz_dict[f'insight_{i}'] = fig
            
            # Outlier insight
            elif insight_type == 'outliers':
                column = insight.get('column')
                if not column or column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
                    continue
                
                # Create box plot
                fig = px.box(
                    df, y=column,
                    title=f"Outliers in {column}"
                )
                
                # Add histogram on the side
                fig2 = px.histogram(
                    df, x=column,
                    title=f"Distribution with Outliers: {column}",
                    marginal="box"
                )
                
                # Add markers for outlier boundaries
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                fig2.add_vline(x=lower_bound, line_dash="dash", line_color="red")
                fig2.add_vline(x=upper_bound, line_dash="dash", line_color="red")
                
                viz_dict[f'insight_{i}_box'] = fig
                viz_dict[f'insight_{i}_hist'] = fig2
            
            # Time series insight
            elif insight_type == 'time_series':
                date_col = insight.get('date_column')
                value_col = insight.get('value_column')
                
                if not date_col or not value_col or date_col not in df.columns or value_col not in df.columns:
                    continue
                
                # Convert to datetime if needed
                plot_df = df.copy()
                if plot_df[date_col].dtype != 'datetime64[ns]':
                    try:
                        plot_df[date_col] = pd.to_datetime(plot_df[date_col])
                    except:
                        continue
                
                # Sort by date
                plot_df = plot_df.sort_values(date_col)
                
                # Create line plot
                fig = px.line(
                    plot_df, x=date_col, y=value_col,
                    title=f"Time Series: {value_col} over {date_col}",
                    markers=True
                )
                
                viz_dict[f'insight_{i}'] = fig
            
            # Group comparison insight
            elif insight_type == 'group_comparison':
                group_col = insight.get('group_column')
                value_col = insight.get('value_column')
                
                if not group_col or not value_col or group_col not in df.columns or value_col not in df.columns:
                    continue
                
                if not pd.api.types.is_numeric_dtype(df[value_col]):
                    continue
                
                # Create grouped bar chart or box plot
                if df[group_col].nunique() < 15:  # For reasonable number of groups
                    # Bar chart of means
                    means = df.groupby(group_col)[value_col].mean().reset_index()
                    fig = px.bar(
                        means, x=group_col, y=value_col,
                        title=f"Mean {value_col} by {group_col}",
                        color=group_col
                    )
                    
                    # Box plot for distributions
                    fig2 = px.box(
                        df, x=group_col, y=value_col,
                        title=f"Distribution of {value_col} by {group_col}",
                        color=group_col
                    )
                    
                    viz_dict[f'insight_{i}_bar'] = fig
                    viz_dict[f'insight_{i}_box'] = fig2
            
            # Trend insight
            elif insight_type == 'trend':
                x_col = insight.get('x_column')
                y_col = insight.get('y_column')
                
                if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
                    continue
                
                # Create line or scatter plot with trend line
                if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                    fig = px.scatter(
                        df, x=x_col, y=y_col,
                        title=f"Trend: {y_col} vs {x_col}",
                        trendline="ols"
                    )
                    
                    viz_dict[f'insight_{i}'] = fig
            
            # Composition insight
            elif insight_type == 'composition':
                column = insight.get('column')
                if not column or column not in df.columns:
                    continue
                
                # Create pie chart for categorical columns with reasonable cardinality
                if df[column].nunique() < 15:
                    value_counts = df[column].value_counts().reset_index()
                    value_counts.columns = [column, 'count']
                    
                    fig = px.pie(
                        value_counts, names=column, values='count',
                        title=f"Composition of {column}",
                        hole=0.3
                    )
                    
                    viz_dict[f'insight_{i}'] = fig
        
        return viz_dict
    
    @staticmethod
    def add_custom_annotations(fig, annotations=None):
        """
        Add custom annotations to a plotly figure
        
        Args:
            fig (plotly.graph_objects.Figure): Figure to annotate
            annotations (list): List of annotation specifications
            
        Returns:
            plotly.graph_objects.Figure: The annotated figure
        """
        if not fig or not annotations:
            return fig
        
        for annotation in annotations:
            x = annotation.get('x')
            y = annotation.get('y')
            text = annotation.get('text', '')
            
            if x is None or y is None:
                continue
            
            fig.add_annotation(
                x=x,
                y=y,
                text=text,
                showarrow=annotation.get('arrow', True),
                arrowhead=annotation.get('arrowhead', 2),
                arrowsize=annotation.get('arrowsize', 1),
                arrowwidth=annotation.get('arrowwidth', 2),
                arrowcolor=annotation.get('arrowcolor', '#636363'),
                bgcolor=annotation.get('bgcolor', 'rgba(255, 255, 255, 0.8)'),
                bordercolor=annotation.get('bordercolor', '#c7c7c7'),
                borderwidth=annotation.get('borderwidth', 1),
                borderpad=annotation.get('borderpad', 4),
                font=annotation.get('font', dict(size=12))
            )
        
        return fig
