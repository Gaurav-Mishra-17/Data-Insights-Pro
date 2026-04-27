import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import io

def load_data(uploaded_file) -> Tuple[pd.DataFrame, str, str]:
    """
    Load data from an uploaded file and infer its type
    
    Args:
        uploaded_file: The uploaded file object from Streamlit
        
    Returns:
        Tuple containing the dataframe, filename, and file type
    """
    file_name = uploaded_file.name
    file_type = file_name.split('.')[-1].lower()
    
    try:
        if file_type == 'csv':
            # Try different encodings and delimiters
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8', keep_default_na=False)
            except:
                try:
                    df = pd.read_csv(uploaded_file, encoding='latin1', keep_default_na=False)
                except:
                    try:
                        df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8', keep_default_na=False)
                    except:
                        df = pd.read_csv(uploaded_file, sep=';', encoding='latin1', keep_default_na=False)
        elif file_type in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file, keep_default_na=False)
        elif file_type == 'json':
            df = pd.read_json(uploaded_file, keep_default_na=False)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
        return df, file_name, file_type
        
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about the dataframe
    
    Args:
        df: The pandas dataframe to analyze
        
    Returns:
        A dictionary with data information
    """
    info = {}
    
    # Basic information
    info['num_rows'] = df.shape[0]
    info['num_columns'] = df.shape[1]
    info['columns'] = list(df.columns)
    info['dtypes'] = {col: str(df[col].dtype) for col in df.columns}
    
    # Missing values
    info['missing_values'] = {col: int(df[col].isna().sum()) for col in df.columns}
    info['missing_percentage'] = {col: round(df[col].isna().mean() * 100, 2) for col in df.columns}
    
    # Duplicates
    info['duplicate_rows'] = int(df.duplicated().sum())
    info['duplicate_percentage'] = round((df.duplicated().sum() / len(df)) * 100, 2)
    
    # Memory usage
    info['memory_usage'] = df.memory_usage(deep=True).sum()
    
    # Column types categorization
    info['numeric_columns'] = list(df.select_dtypes(include=['int64', 'float64']).columns)
    info['categorical_columns'] = list(df.select_dtypes(include=['object', 'category']).columns)
    info['datetime_columns'] = list(df.select_dtypes(include=['datetime64']).columns)
    
    # Auto-detect datetime columns that might be stored as strings
    potential_datetime_cols = []
    for col in info['categorical_columns']:
        if df[col].nunique() < len(df) * 0.7:  # Only check if reasonably unique
            try:
                pd.to_datetime(df[col], errors='raise')
                potential_datetime_cols.append(col)
            except:
                pass
    info['potential_datetime_columns'] = potential_datetime_cols
    
    # Basic statistics for numeric columns
    if info['numeric_columns']:
        info['numeric_stats'] = df[info['numeric_columns']].describe().to_dict()
    
    # Value counts for categorical columns (limited to top 10)
    info['categorical_stats'] = {}
    for col in info['categorical_columns']:
        if len(df[col].unique()) < 50:  # Only if reasonable number of categories
            info['categorical_stats'][col] = df[col].value_counts().head(10).to_dict()
    
    # Detect potential outliers in numeric columns
    info['potential_outliers'] = {}
    for col in info['numeric_columns']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
        info['potential_outliers'][col] = {
            'count': int(outliers),
            'percentage': round((outliers / len(df)) * 100, 2)
        }
        
    return info

def infer_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer and convert columns to their appropriate data types
    
    Args:
        df: The pandas dataframe to process
        
    Returns:
        The dataframe with adjusted data types
    """
    df_copy = df.copy()
    
    # Try to convert object columns to numeric
    for col in df_copy.select_dtypes(include=['object']).columns:
        # Skip columns with high cardinality relative to the dataset size
        if df_copy[col].nunique() > min(len(df_copy) * 0.5, 100):
            continue
            
        # Try to convert to numeric
        try:
            numeric_values = pd.to_numeric(df_copy[col], errors='coerce')
            # If most values converted successfully, use the numeric version
            if numeric_values.notna().sum() > 0.8 * df_copy[col].count():
                df_copy[col] = numeric_values
        except:
            pass
            
        # Try to convert to datetime
        try:
            datetime_values = pd.to_datetime(df_copy[col], errors='coerce')
            # If most values converted successfully, use the datetime version
            if datetime_values.notna().sum() > 0.8 * df_copy[col].count():
                df_copy[col] = datetime_values
        except:
            pass
            
    # Convert low-cardinality string columns to categories
    for col in df_copy.select_dtypes(include=['object']).columns:
        if df_copy[col].nunique() < min(len(df_copy) * 0.2, 50):
            df_copy[col] = df_copy[col].astype('category')
            
    return df_copy

def get_column_sample(df: pd.DataFrame, column: str, n=5) -> List:
    """
    Get a sample of values from a dataframe column
    
    Args:
        df: The pandas dataframe
        column: The column name
        n: Number of sample values to return
        
    Returns:
        List of sample values
    """
    return df[column].dropna().sample(min(n, df[column].count())).tolist()

def generate_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive data profile with even more statistics
    
    Args:
        df: The pandas dataframe to analyze
        
    Returns:
        A dictionary with detailed profile information
    """
    profile = {}
    
    # Get basic info
    profile['basic'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'duplicated_rows': df.duplicated().sum()
    }
    
    # Column information
    profile['columns'] = {}
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'missing_count': int(df[col].isna().sum()),
            'missing_percentage': round(df[col].isna().mean() * 100, 2),
            'unique_count': int(df[col].nunique()),
            'unique_percentage': round((df[col].nunique() / len(df)) * 100, 2)
        }
        
        # Add type-specific statistics
        if np.issubdtype(df[col].dtype, np.number):
            col_info.update({
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                'skew': float(df[col].skew()) if not pd.isna(df[col].skew()) else None,
                'kurtosis': float(df[col].kurtosis()) if not pd.isna(df[col].kurtosis()) else None
            })
            
            # Check for zeros
            col_info['zero_count'] = int((df[col] == 0).sum())
            
            # Calculate quartiles
            col_info['quartiles'] = {
                'q1': float(df[col].quantile(0.25)),
                'q2': float(df[col].quantile(0.5)),
                'q3': float(df[col].quantile(0.75))
            }
            
            # Detect outliers
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            col_info['outliers'] = {
                'count': len(outliers),
                'percentage': round((len(outliers) / len(df)) * 100, 2),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
        
        elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # Get value counts for categorical data
            value_counts = df[col].value_counts().head(10).to_dict()
            col_info['value_counts'] = value_counts
            
            # Get string length statistics if object type
            if df[col].dtype == 'object':
                str_lens = df[col].dropna().astype(str).str.len()
                if not str_lens.empty:
                    col_info['string_length'] = {
                        'min': int(str_lens.min()),
                        'max': int(str_lens.max()),
                        'mean': float(str_lens.mean())
                    }
        
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Get datetime statistics
            col_info['datetime'] = {
                'min': df[col].min().strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(df[col].min()) else None,
                'max': df[col].max().strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(df[col].max()) else None
            }
            
            # Calculate date ranges
            if not pd.isna(df[col].min()) and not pd.isna(df[col].max()):
                date_range = df[col].max() - df[col].min()
                col_info['datetime']['range_days'] = date_range.days
        
        profile['columns'][col] = col_info
    
    # Correlation matrix for numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        profile['correlations'] = numeric_df.corr().to_dict()
    
    return profile

def export_data(df: pd.DataFrame, format_type: str) -> io.BytesIO:
    """
    Export dataframe to different formats
    
    Args:
        df: The pandas dataframe to export
        format_type: The export format (csv, excel, json)
        
    Returns:
        BytesIO object with the exported data
    """
    buffer = io.BytesIO()
    
    if format_type == 'csv':
        df.to_csv(buffer, index=False)
    elif format_type == 'excel':
        df.to_excel(buffer, index=False)
    elif format_type == 'json':
        buffer.write(df.to_json(orient='records').encode())
    else:
        raise ValueError(f"Unsupported export format: {format_type}")
        
    buffer.seek(0)
    return buffer
