import pandas as pd
import numpy as np
import re
from datetime import datetime

def get_data_summary(df):
    """
    Generate a comprehensive summary of a dataframe
    
    Args:
        df (pandas.DataFrame): The dataframe to analyze
        
    Returns:
        dict: Dictionary containing summary information
    """
    if df is None or df.empty:
        return {}
    
    # Basic info
    n_rows, n_cols = df.shape
    memory_usage = df.memory_usage(deep=True).sum()
    
    # Format memory usage
    if memory_usage < 1024:
        memory_str = f"{memory_usage} bytes"
    elif memory_usage < 1024**2:
        memory_str = f"{memory_usage/1024:.2f} KB"
    elif memory_usage < 1024**3:
        memory_str = f"{memory_usage/(1024**2):.2f} MB"
    else:
        memory_str = f"{memory_usage/(1024**3):.2f} GB"
    
    # Column types
    dtypes = df.dtypes.value_counts().to_dict()
    dtypes = {str(k): v for k, v in dtypes.items()}
    
    # Missing values
    total_missing = df.isna().sum().sum()
    missing_percent = (total_missing / (n_rows * n_cols)) * 100
    
    # Columns with missing values
    cols_with_missing = {}
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            cols_with_missing[col] = {
                'count': int(missing),
                'percent': float((missing / n_rows) * 100)
            }
    
    # Duplicate rows
    duplicate_count = df.duplicated().sum()
    
    # Numeric column statistics
    numeric_stats = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        stats = df[col].describe().to_dict()
        stats['skew'] = float(df[col].skew())
        stats['kurtosis'] = float(df[col].kurtosis())
        # Convert numpy types to native Python types for JSON serialization
        numeric_stats[col] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else int(v) if isinstance(v, (np.int32, np.int64)) else v 
                             for k, v in stats.items()}
    
    # Categorical column statistics
    categorical_stats = {}
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        value_counts = df[col].value_counts().head(10).to_dict()
        unique_count = df[col].nunique()
        top_value = df[col].mode()[0] if not df[col].mode().empty else None
        stats = {
            'unique_count': unique_count,
            'top_value': str(top_value) if top_value is not None else None,
            'top_value_count': int(df[col].value_counts().iloc[0]) if not df[col].value_counts().empty else 0,
            'sample_values': list(df[col].dropna().sample(min(5, df[col].count())).astype(str).values)
        }
        categorical_stats[col] = stats
    
    # Date column detection and statistics
    date_stats = {}
    potential_date_cols = df.select_dtypes(include=['object', 'datetime64']).columns
    
    for col in potential_date_cols:
        if df[col].dtype == 'datetime64[ns]':
            # Already a datetime column
            dates = df[col].dropna()
            if not dates.empty:
                date_stats[col] = {
                    'min_date': dates.min().strftime('%Y-%m-%d'),
                    'max_date': dates.max().strftime('%Y-%m-%d'),
                    'range_days': (dates.max() - dates.min()).days
                }
        else:
            # Try to detect if this is a date column
            # Only check if fewer than 100 unique values to avoid performance issues
            if df[col].nunique() < 100:
                sample = df[col].dropna().astype(str).sample(min(100, df[col].count())).values
                date_patterns = [
                    # Common date formats
                    r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
                    r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YY[YY]
                    r'\d{1,2}-\d{1,2}-\d{2,4}',  # DD-MM-YYYY
                    r'\d{1,2}\s+[A-Za-z]{3,}\s+\d{2,4}'  # DD Month YYYY
                ]
                
                match_count = 0
                for s in sample:
                    if any(re.match(pattern, s) for pattern in date_patterns):
                        match_count += 1
                
                # If more than 75% match a date pattern, consider it a date column
                if match_count > len(sample) * 0.75:
                    date_stats[col] = {
                        'likely_date_column': True,
                        'format': 'unknown'
                    }
    
    # Prepare final summary
    summary = {
        'rows': n_rows,
        'columns': n_cols,
        'memory_usage': memory_str,
        'dtypes': dtypes,
        'missing_values': {
            'total': int(total_missing),
            'percent': float(missing_percent),
            'columns': cols_with_missing
        },
        'duplicates': {
            'count': int(duplicate_count),
            'percent': float((duplicate_count / n_rows) * 100) if n_rows > 0 else 0
        },
        'numeric_stats': numeric_stats,
        'categorical_stats': categorical_stats,
        'date_stats': date_stats
    }
    
    return summary

def format_large_number(num):
    """Format large numbers with K, M, B suffixes."""
    if num < 1000:
        return str(num)
    elif num < 1000000:
        return f"{num/1000:.1f}K"
    elif num < 1000000000:
        return f"{num/1000000:.1f}M"
    else:
        return f"{num/1000000000:.1f}B"

def generate_sample_query(df):
    """
    Generate a natural language query example based on the dataset columns
    
    Args:
        df (pandas.DataFrame): The dataframe to analyze
        
    Returns:
        str: An example query
    """
    if df is None or df.empty:
        return "Ask a question about your data"
        
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols][:3]
    
    if not numeric_cols:
        return "Ask a question about your data"
    
    # Generate different types of example queries
    examples = []
    
    # Top N by metric
    if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
        examples.append(f"Show top 5 {categorical_cols[0]} by {numeric_cols[0]}")
    
    # Average of a metric
    if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
        examples.append(f"What is the average {numeric_cols[0]} by {categorical_cols[0]}?")
    
    # Comparison
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        examples.append(f"Compare {numeric_cols[0]} across different {categorical_cols[0]}")
    
    # Trend if time column is detected
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or df[col].dtype == 'datetime64[ns]']
    if date_cols and numeric_cols:
        examples.append(f"Show trend of {numeric_cols[0]} over {date_cols[0]}")
    
    if not examples:
        return "Ask a question about your data"
    
    return np.random.choice(examples)

def detect_potential_target(df):
    """
    Suggest potential target variables for ML modeling
    
    Args:
        df (pandas.DataFrame): The dataframe to analyze
        
    Returns:
        list: List of potential target columns with reasons
    """
    if df is None or df.empty:
        return []
        
    potential_targets = []
    
    # Look for columns that might be targets based on name
    target_keywords = ['target', 'class', 'label', 'outcome', 'result', 'status', 
                      'response', 'dependent', 'output', 'category', 'diagnos',
                      'risk', 'fraud', 'default', 'churn', 'conversion', 'click',
                      'purchase', 'revenue', 'sales', 'income', 'profit', 'rating',
                      'score', 'satisfaction', 'sentiment']
    
    for col in df.columns:
        # Check for keyword match in column name
        col_lower = col.lower()
        keyword_match = [keyword for keyword in target_keywords if keyword in col_lower]
        
        if keyword_match:
            # Check column properties to see if it's likely to be a target
            if df[col].dtype == 'bool' or (df[col].nunique() == 2):
                potential_targets.append({
                    'column': col,
                    'reason': f"Binary column with name containing '{keyword_match[0]}', suggesting a classification target",
                    'type': 'classification',
                    'likelihood': 'high'
                })
            elif df[col].nunique() < 10 and (df[col].dtype == 'object' or df[col].dtype.name == 'category'):
                potential_targets.append({
                    'column': col,
                    'reason': f"Categorical column with name containing '{keyword_match[0]}', suggesting a multi-class target",
                    'type': 'classification',
                    'likelihood': 'medium'
                })
            elif np.issubdtype(df[col].dtype, np.number) and df[col].nunique() > 10:
                potential_targets.append({
                    'column': col,
                    'reason': f"Numeric column with name containing '{keyword_match[0]}', suggesting a regression target",
                    'type': 'regression',
                    'likelihood': 'medium'
                })
    
    # If no keyword matches, look for other indicators
    if not potential_targets:
        # Binary columns might be classification targets
        binary_cols = [col for col in df.columns 
                      if df[col].nunique() == 2 and col.lower() not in ['id', 'user_id', 'customer_id']]
        
        for col in binary_cols:
            potential_targets.append({
                'column': col,
                'reason': "Binary column with exactly 2 unique values, could be a classification target",
                'type': 'classification',
                'likelihood': 'low'
            })
        
        # Columns with low number of categories might be classification targets
        categorical_cols = [col for col in df.columns 
                           if df[col].nunique() < 10 and 
                           df[col].nunique() > 2 and 
                           (df[col].dtype == 'object' or df[col].dtype.name == 'category') and
                           col.lower() not in ['id', 'user_id', 'customer_id']]
        
        for col in categorical_cols[:2]:  # Limit to top 2 candidates
            potential_targets.append({
                'column': col,
                'reason': f"Categorical column with {df[col].nunique()} classes, could be a multi-class target",
                'type': 'classification',
                'likelihood': 'low'
            })
        
        # Continuous numeric columns might be regression targets
        numeric_cols = [col for col in df.select_dtypes(include=np.number).columns
                       if df[col].nunique() > 10 and 
                       col.lower() not in ['id', 'user_id', 'customer_id']]
        
        for col in numeric_cols[:2]:  # Limit to top 2 candidates
            potential_targets.append({
                'column': col,
                'reason': "Numeric column with many unique values, could be a regression target",
                'type': 'regression',
                'likelihood': 'low'
            })
    
    return potential_targets

def suggest_feature_engineering(df):
    """
    Suggest feature engineering transformations based on dataset characteristics
    
    Args:
        df (pandas.DataFrame): The dataframe to analyze
        
    Returns:
        dict: Dictionary with suggested transformations for different columns
    """
    if df is None or df.empty:
        return {}
    
    suggestions = {}
    
    # Check for datetime columns
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    date_cols.extend([col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()])
    date_cols = list(set(date_cols))  # Remove duplicates
    
    if date_cols:
        for col in date_cols:
            suggestions[col] = {
                'type': 'datetime',
                'transformations': [
                    'Extract day of week',
                    'Extract month',
                    'Extract year',
                    'Extract quarter',
                    'Create is_weekend feature',
                    'Calculate days since specific date'
                ]
            }
    
    # Check for skewed numeric distributions that might benefit from transformation
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].skew() > 1.5:  # Highly positive skew
            suggestions[col] = {
                'type': 'numeric_transform',
                'skew': float(df[col].skew()),
                'transformations': [
                    'Log transformation',
                    'Square root transformation',
                    'Box-Cox transformation'
                ]
            }
        elif df[col].skew() < -1.5:  # Highly negative skew
            suggestions[col] = {
                'type': 'numeric_transform',
                'skew': float(df[col].skew()),
                'transformations': [
                    'Square transformation',
                    'Cube transformation',
                    'Exponential transformation'
                ]
            }
    
    # Check for categorical columns with high cardinality
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count > 10:
            suggestions[col] = {
                'type': 'high_cardinality',
                'unique_values': unique_count,
                'transformations': [
                    'Frequency encoding',
                    'Target encoding (if for ML)',
                    'Grouping rare categories',
                    'Dimensionality reduction (e.g., PCA on one-hot encoded values)'
                ]
            }
    
    # Check for text fields
    text_cols = []
    for col in categorical_cols:
        # Sample the column to check for text
        sample = df[col].dropna().astype(str).sample(min(100, df[col].count()))
        # If average length > 20 characters, it's likely text
        if sample.str.len().mean() > 20:
            text_cols.append(col)
    
    for col in text_cols:
        suggestions[col] = {
            'type': 'text',
            'transformations': [
                'Text length as a feature',
                'Count of special characters',
                'TF-IDF vectorization',
                'Word embeddings',
                'Sentiment analysis'
            ]
        }
    
    # Check for potential interaction terms
    if len(numeric_cols) >= 2:
        suggestions['interactions'] = {
            'type': 'interaction',
            'transformations': [
                'Multiplication of related numeric features',
                'Ratio of related numeric features',
                'Polynomial features for regression problems'
            ]
        }
    
    return suggestions
