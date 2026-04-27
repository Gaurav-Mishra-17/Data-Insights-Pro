import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import re

def get_column_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Get comprehensive statistics for a specific column
    
    Args:
        df: The pandas dataframe
        column: The column name to analyze
        
    Returns:
        Dictionary with column statistics
    """
    stats = {}
    
    # Get basic information
    stats['name'] = column
    stats['dtype'] = str(df[column].dtype)
    stats['count'] = int(df[column].count())
    stats['missing'] = int(df[column].isna().sum())
    stats['missing_percent'] = round(df[column].isna().mean() * 100, 2)
    stats['unique'] = int(df[column].nunique())
    
    # Type-specific statistics
    if pd.api.types.is_numeric_dtype(df[column].dtype):
        # Numeric column
        stats['min'] = float(df[column].min()) if not pd.isna(df[column].min()) else None
        stats['max'] = float(df[column].max()) if not pd.isna(df[column].max()) else None
        stats['mean'] = float(df[column].mean()) if not pd.isna(df[column].mean()) else None
        stats['median'] = float(df[column].median()) if not pd.isna(df[column].median()) else None
        stats['std'] = float(df[column].std()) if not pd.isna(df[column].std()) else None
        stats['skewness'] = float(df[column].skew()) if not pd.isna(df[column].skew()) else None
        stats['kurtosis'] = float(df[column].kurtosis()) if not pd.isna(df[column].kurtosis()) else None
        
        # Quantiles
        stats['quantiles'] = {
            '5%': float(df[column].quantile(0.05)),
            '25%': float(df[column].quantile(0.25)), 
            '50%': float(df[column].quantile(0.5)),
            '75%': float(df[column].quantile(0.75)),
            '95%': float(df[column].quantile(0.95))
        }
        
        # Zero and negative values
        stats['zeros'] = int((df[column] == 0).sum())
        stats['negatives'] = int((df[column] < 0).sum())
        
        # Top and bottom values
        stats['top_5_values'] = df[column].nlargest(5).tolist()
        stats['bottom_5_values'] = df[column].nsmallest(5).tolist()
    
    elif pd.api.types.is_string_dtype(df[column].dtype) or pd.api.types.is_categorical_dtype(df[column].dtype):
        # Categorical/string column
        value_counts = df[column].value_counts().head(10).to_dict()
        stats['value_counts'] = value_counts
        
        if df[column].nunique() > 0:
            stats['most_common'] = df[column].value_counts().index[0]
            stats['most_common_count'] = int(df[column].value_counts().iloc[0])
            stats['most_common_percent'] = round(df[column].value_counts().iloc[0] / df[column].count() * 100, 2)
        
        # Length statistics for string columns
        if pd.api.types.is_string_dtype(df[column].dtype):
            str_lens = df[column].dropna().astype(str).str.len()
            if not str_lens.empty:
                stats['min_length'] = int(str_lens.min())
                stats['max_length'] = int(str_lens.max())
                stats['mean_length'] = float(str_lens.mean())
    
    elif pd.api.types.is_datetime64_any_dtype(df[column].dtype):
        # Datetime column
        stats['min'] = df[column].min().strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(df[column].min()) else None
        stats['max'] = df[column].max().strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(df[column].max()) else None
        
        if not pd.isna(df[column].min()) and not pd.isna(df[column].max()):
            time_range = df[column].max() - df[column].min()
            stats['range_days'] = time_range.days
        
        # Year, month, day of week distributions
        year_counts = df[column].dt.year.value_counts().sort_index().head(10).to_dict()
        month_counts = df[column].dt.month.value_counts().sort_index().to_dict()
        dow_counts = df[column].dt.dayofweek.value_counts().sort_index().to_dict()
        
        stats['year_counts'] = year_counts
        stats['month_counts'] = month_counts
        stats['day_of_week_counts'] = {str(k): v for k, v in dow_counts.items()}
    
    return stats

def get_correlations(df: pd.DataFrame, method: str = 'pearson') -> Dict[str, Dict[str, float]]:
    """
    Get correlation matrix for numeric columns
    
    Args:
        df: The pandas dataframe
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        
    Returns:
        Dictionary with correlation values
    """
    # Get only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty or numeric_df.shape[1] < 2:
        return {}
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr(method=method)
    
    # Convert to dictionary
    correlation_dict = {}
    for col1 in corr_matrix.columns:
        correlation_dict[col1] = {}
        for col2 in corr_matrix.columns:
            if col1 != col2:  # Skip self-correlations
                correlation_dict[col1][col2] = round(corr_matrix.loc[col1, col2], 3)
    
    return correlation_dict

def get_top_correlations(df: pd.DataFrame, method: str = 'pearson', n: int = 10) -> List[Dict[str, Any]]:
    """
    Get top N correlations between numeric columns
    
    Args:
        df: The pandas dataframe
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        n: Number of top correlations to return
        
    Returns:
        List of dictionaries with top correlations
    """
    # Get only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty or numeric_df.shape[1] < 2:
        return []
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr(method=method)
    
    # Convert to tidy format (excluding self-correlations)
    tidy_corr = corr_matrix.stack().reset_index()
    tidy_corr.columns = ['variable_1', 'variable_2', 'correlation']
    tidy_corr = tidy_corr[tidy_corr['variable_1'] != tidy_corr['variable_2']]
    
    # Remove duplicates (e.g., A-B and B-A)
    tidy_corr['sorted_vars'] = tidy_corr.apply(
        lambda row: '-'.join(sorted([row['variable_1'], row['variable_2']])), axis=1
    )
    tidy_corr = tidy_corr.drop_duplicates('sorted_vars')
    
    # Sort by absolute correlation and get top N
    tidy_corr['abs_correlation'] = tidy_corr['correlation'].abs()
    tidy_corr = tidy_corr.sort_values('abs_correlation', ascending=False).head(n)
    
    # Convert to list of dictionaries
    result = []
    for _, row in tidy_corr.iterrows():
        result.append({
            'variable_1': row['variable_1'],
            'variable_2': row['variable_2'],
            'correlation': round(row['correlation'], 3)
        })
    
    return result

def filter_data(df: pd.DataFrame, filters: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Filter dataframe based on specified conditions
    
    Args:
        df: The pandas dataframe
        filters: Dictionary of filter conditions
            Format: {
                'column_name': {
                    'type': 'numeric'|'category'|'datetime',
                    'operator': 'equals'|'not_equals'|'greater'|'less'|'between'|'contains',
                    'value': value or [min, max] for 'between'
                }
            }
        
    Returns:
        Filtered dataframe
    """
    filtered_df = df.copy()
    
    for column, filter_dict in filters.items():
        if column not in filtered_df.columns:
            continue
        
        filter_type = filter_dict.get('type')
        operator = filter_dict.get('operator')
        value = filter_dict.get('value')
        
        if filter_type == 'numeric':
            if operator == 'equals':
                filtered_df = filtered_df[filtered_df[column] == value]
            elif operator == 'not_equals':
                filtered_df = filtered_df[filtered_df[column] != value]
            elif operator == 'greater':
                filtered_df = filtered_df[filtered_df[column] > value]
            elif operator == 'less':
                filtered_df = filtered_df[filtered_df[column] < value]
            elif operator == 'between' and isinstance(value, list) and len(value) == 2:
                filtered_df = filtered_df[(filtered_df[column] >= value[0]) & 
                                         (filtered_df[column] <= value[1])]
        
        elif filter_type == 'category':
            if operator == 'equals':
                filtered_df = filtered_df[filtered_df[column] == value]
            elif operator == 'not_equals':
                filtered_df = filtered_df[filtered_df[column] != value]
            elif operator == 'contains':
                filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(value, na=False)]
            elif operator == 'in' and isinstance(value, list):
                filtered_df = filtered_df[filtered_df[column].isin(value)]
        
        elif filter_type == 'datetime':
            # Convert value to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(filtered_df[column].dtype):
                filtered_df[column] = pd.to_datetime(filtered_df[column], errors='coerce')
            
            if operator == 'equals':
                value_dt = pd.to_datetime(value)
                filtered_df = filtered_df[filtered_df[column].dt.date == value_dt.date()]
            elif operator == 'before':
                value_dt = pd.to_datetime(value)
                filtered_df = filtered_df[filtered_df[column] < value_dt]
            elif operator == 'after':
                value_dt = pd.to_datetime(value)
                filtered_df = filtered_df[filtered_df[column] > value_dt]
            elif operator == 'between' and isinstance(value, list) and len(value) == 2:
                start_dt = pd.to_datetime(value[0])
                end_dt = pd.to_datetime(value[1])
                filtered_df = filtered_df[(filtered_df[column] >= start_dt) & 
                                         (filtered_df[column] <= end_dt)]
    
    return filtered_df

def group_data(df: pd.DataFrame, 
                group_by: List[str], 
                agg_columns: List[str], 
                agg_functions: List[str]) -> pd.DataFrame:
    """
    Group dataframe by columns and calculate aggregations
    
    Args:
        df: The pandas dataframe
        group_by: List of columns to group by
        agg_columns: List of columns to aggregate
        agg_functions: List of aggregation functions to apply
        
    Returns:
        Grouped dataframe
    """
    # Check if all group_by columns exist
    for col in group_by:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
    
    # Check if all agg_columns exist
    for col in agg_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
            
    # Create aggregation dictionary
    agg_dict = {}
    for col in agg_columns:
        agg_dict[col] = []
        for func in agg_functions:
            if func in ['sum', 'mean', 'min', 'max', 'count', 'std', 'median']:
                agg_dict[col].append(func)
            
    # Group by and aggregate
    grouped_df = df.groupby(group_by, as_index=False).agg(agg_dict)
    
    # Flatten multi-level columns if they exist
    if isinstance(grouped_df.columns, pd.MultiIndex):
        grouped_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                             for col in grouped_df.columns]
    
    return grouped_df

def temporal_analysis(df: pd.DataFrame, 
                      date_column: str, 
                      value_column: str, 
                      freq: str = 'M',
                      agg_func: str = 'mean') -> pd.DataFrame:
    """
    Perform temporal analysis on time series data
    
    Args:
        df: The pandas dataframe
        date_column: The datetime column to group by
        value_column: The column with values to aggregate
        freq: Frequency for resampling ('D'=daily, 'W'=weekly, 'M'=monthly, 'Q'=quarterly, 'Y'=yearly)
        agg_func: Aggregation function ('mean', 'sum', 'min', 'max', 'count')
        
    Returns:
        Dataframe with temporal analysis results
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column].dtype):
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Set date as index
    ts_df = df.set_index(date_column)
    
    # Resample by frequency and aggregate
    if agg_func == 'mean':
        result = ts_df[value_column].resample(freq).mean()
    elif agg_func == 'sum':
        result = ts_df[value_column].resample(freq).sum()
    elif agg_func == 'min':
        result = ts_df[value_column].resample(freq).min()
    elif agg_func == 'max':
        result = ts_df[value_column].resample(freq).max()
    elif agg_func == 'count':
        result = ts_df[value_column].resample(freq).count()
    else:
        result = ts_df[value_column].resample(freq).mean()
    
    # Reset index to get date as a column
    result = result.reset_index()
    result.columns = [date_column, f"{value_column}_{agg_func}"]
    
    return result

def calculate_summary_by_group(df: pd.DataFrame, 
                               group_column: str, 
                               value_column: str) -> pd.DataFrame:
    """
    Calculate summary statistics grouped by a categorical column
    
    Args:
        df: The pandas dataframe
        group_column: The categorical column to group by
        value_column: The numeric column to calculate statistics for
        
    Returns:
        Dataframe with summary statistics by group
    """
    # Group by the categorical column and calculate statistics
    summary = df.groupby(group_column)[value_column].agg([
        'count',
        'mean',
        'median',
        'std',
        'min',
        'max'
    ]).reset_index()
    
    # Calculate quartiles
    quartiles = df.groupby(group_column)[value_column].quantile([0.25, 0.75]).unstack()
    quartiles.columns = ['q1', 'q3']
    quartiles = quartiles.reset_index()
    
    # Merge with summary
    result = pd.merge(summary, quartiles, on=group_column)
    
    return result

def detect_anomalies(df: pd.DataFrame, 
                     column: str, 
                     method: str = 'iqr',
                     threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect anomalies in a numeric column
    
    Args:
        df: The pandas dataframe
        column: The numeric column to analyze
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for anomaly detection (1.5 for IQR, 3.0 for z-score)
        
    Returns:
        Dataframe of anomalous rows
    """
    if not pd.api.types.is_numeric_dtype(df[column].dtype):
        raise ValueError(f"Column '{column}' must be numeric")
    
    if method == 'iqr':
        # IQR method
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        # Find anomalies
        anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)].copy()
        
        # Add anomaly score - how many IQRs away from the median
        median = df[column].median()
        anomalies['anomaly_score'] = anomalies[column].apply(
            lambda x: abs(x - median) / iqr
        )
        
    elif method == 'zscore':
        # Z-score method
        mean = df[column].mean()
        std = df[column].std()
        
        # Calculate z-scores
        z_scores = abs((df[column] - mean) / std)
        
        # Find anomalies
        anomalies = df[z_scores > threshold].copy()
        
        # Add the z-score as anomaly score
        anomalies['anomaly_score'] = z_scores[z_scores > threshold]
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")
    
    # Sort by anomaly score descending
    if not anomalies.empty:
        anomalies = anomalies.sort_values('anomaly_score', ascending=False)
    
    return anomalies

def parse_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Parse a natural language query to extract data insights
    
    Args:
        df: The pandas dataframe
        query: Natural language query string
        
    Returns:
        Result dataframe based on query
    """
    query = query.lower()
    result = None
    
    # Check for top/bottom N pattern
    top_pattern = re.search(r'top\s+(\d+)', query)
    bottom_pattern = re.search(r'bottom\s+(\d+)', query)
    
    # Find columns that might be in the query
    mentioned_columns = []
    for col in df.columns:
        if col.lower() in query:
            mentioned_columns.append(col)
    
    # If no columns found, return empty dataframe
    if not mentioned_columns:
        return pd.DataFrame()
    
    # Find numeric columns for analysis
    numeric_columns = [col for col in mentioned_columns 
                      if pd.api.types.is_numeric_dtype(df[col].dtype)]
    
    # Find categorical/groupby columns
    category_columns = [col for col in mentioned_columns 
                       if col not in numeric_columns]
    
    # Determine if we need to group by some column
    group_by_col = None
    for col in category_columns:
        if f"by {col.lower()}" in query or f"per {col.lower()}" in query:
            group_by_col = col
            break
    
    # Determine which operation to perform (sum, average, count, etc.)
    operation = 'sum'  # default
    if 'average' in query or 'mean' in query:
        operation = 'mean'
    elif 'count' in query:
        operation = 'count'
    elif 'maximum' in query or 'highest' in query:
        operation = 'max'
    elif 'minimum' in query or 'lowest' in query:
        operation = 'min'
    
    # Determine which value column to use
    value_col = None
    value_keywords = ['sum of', 'total', 'average', 'mean', 'count', 'maximum', 'minimum']
    
    for col in numeric_columns:
        for keyword in value_keywords:
            if f"{keyword} {col.lower()}" in query:
                value_col = col
                break
        if value_col:
            break
    
    # If no value column explicitly mentioned, take the first numeric column
    if not value_col and numeric_columns:
        value_col = numeric_columns[0]
    
    # If we have a group by column and value column, perform the aggregation
    if group_by_col and value_col:
        result = df.groupby(group_by_col)[value_col].agg(operation).reset_index()
        result.columns = [group_by_col, f"{operation}_{value_col}"]
        
        # Apply top/bottom N filter if requested
        if top_pattern:
            n = int(top_pattern.group(1))
            result = result.sort_values(f"{operation}_{value_col}", ascending=False).head(n)
        elif bottom_pattern:
            n = int(bottom_pattern.group(1))
            result = result.sort_values(f"{operation}_{value_col}", ascending=True).head(n)
    
    # If no groupby but value column exists, return basic stats
    elif value_col:
        if 'distribution' in query or 'histogram' in query:
            # Return distribution data
            bin_count = 10  # default
            bin_match = re.search(r'(\d+)\s+bins', query)
            if bin_match:
                bin_count = int(bin_match.group(1))
            
            bins = pd.cut(df[value_col], bins=bin_count)
            result = pd.DataFrame({
                'bin': bins.cat.categories.astype(str),
                'count': bins.value_counts().sort_index().values
            })
        else:
            # Return basic statistics
            result = pd.DataFrame({
                'statistic': ['count', 'mean', 'median', 'std', 'min', 'max'],
                'value': [
                    df[value_col].count(),
                    df[value_col].mean(),
                    df[value_col].median(),
                    df[value_col].std(),
                    df[value_col].min(),
                    df[value_col].max()
                ]
            })
    
    return result if result is not None else pd.DataFrame()
