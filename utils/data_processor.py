import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from dateutil.parser import parse
import re
from datetime import datetime
import streamlit as st

class DataProcessor:
    """Class for data processing and cleaning operations"""
    
    @staticmethod
    def calculate_quality_score(df):
        """
        Calculate a data quality score from 0-100
        
        Args:
            df (pandas.DataFrame): The dataframe to analyze
            
        Returns:
            int: Quality score from 0-100
        """
        if df is None or df.empty:
            return 0
        
        # Initialize score
        score = 100
        
        # Penalize for missing values
        missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        missing_penalty = min(30, missing_pct * 3)  # Up to 30 points penalty
        score -= missing_penalty
        
        # Penalize for duplicate rows
        duplicate_pct = (df.duplicated().sum() / df.shape[0]) * 100
        duplicate_penalty = min(15, duplicate_pct * 3)  # Up to 15 points penalty
        score -= duplicate_penalty
        
        # Penalize for high cardinality categorical columns (potential issues)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        high_card_cols = 0
        for col in cat_cols:
            if df[col].nunique() > 100 and df[col].nunique() > 0.5 * len(df):
                high_card_cols += 1
        
        high_card_penalty = min(10, high_card_cols * 2)  # Up to 10 points penalty
        score -= high_card_penalty
        
        # Penalize for columns with high percentage of outliers
        numeric_cols = df.select_dtypes(include=np.number).columns
        outlier_cols = 0
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_pct = ((df[col] < lower_bound) | (df[col] > upper_bound)).mean() * 100
            if outlier_pct > 10:
                outlier_cols += 1
        
        outlier_penalty = min(15, outlier_cols * 3)  # Up to 15 points penalty
        score -= outlier_penalty
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, round(score)))

        # Check for incorrect data types
        dtype_issues = 0
        
        # Check for numeric columns stored as objects
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert sample to numeric
                sample = df[col].dropna().head(100)
                numeric_count = 0
                for val in sample:
                    try:
                        float(val)
                        numeric_count += 1
                    except:
                        pass
                
                # If more than 80% can be converted to numeric, it's likely a numeric column
                if numeric_count > 0.8 * len(sample) and len(sample) > 0:
                    dtype_issues += 1
        
        # Check for date columns stored as objects
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to parse sample as dates
                sample = df[col].dropna().head(100)
                date_count = 0
                for val in sample:
                    try:
                        parse(str(val))
                        date_count += 1
                    except:
                        pass
                
                # If more than 80% can be parsed as dates, it's likely a date column
                if date_count > 0.8 * len(sample) and len(sample) > 0:
                    dtype_issues += 1
        
        dtype_penalty = min(15, dtype_issues * 3)  # Up to 15 points penalty
        score -= dtype_penalty
        
        # Penalize for skewed distributions in numeric columns
        skewed_cols = 0
        for col in numeric_cols:
            if abs(df[col].skew()) > 3:
                skewed_cols += 1
        
        skew_penalty = min(5, skewed_cols)  # Up to 5 points penalty
        score -= skew_penalty
        
        # Penalize for inconsistent text formatting
        text_issues = 0
        for col in cat_cols:
            if df[col].nunique() < 10:  # Only check low-cardinality columns
                values = df[col].dropna().unique()
                # Check for case inconsistency
                if any(str(v).lower() != str(v) for v in values) and any(str(v).lower() == str(v).lower() for v in values):
                    text_issues += 1
                
                # Check for leading/trailing whitespace
                if any(str(v).strip() != str(v) for v in values):
                    text_issues += 1
        
        text_penalty = min(10, text_issues * 2)  # Up to 10 points penalty
        score -= text_penalty
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, round(score)))
        
        return score
        
    @staticmethod
    def detect_issues(df):
        """
        Detect potential data quality issues in the dataset
        
        Args:
            df (pandas.DataFrame): The dataframe to analyze
            
        Returns:
            list: List of detected issues
        """
        if df is None or df.empty:
            return ["Empty dataset"]
        
        issues = []
        
        # Check for missing values
        missing_counts = df.isna().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if not missing_cols.empty:
            if len(missing_cols) > 5:
                issues.append(f"Missing values detected in {len(missing_cols)} columns")
            else:
                cols_str = ", ".join([f"{col} ({missing_counts[col]} missing)" for col in missing_cols.index])
                issues.append(f"Missing values detected in columns: {cols_str}")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            dup_pct = (duplicates / len(df)) * 100
            issues.append(f"Found {duplicates} duplicate rows ({dup_pct:.1f}% of data)")
        
        # Check for potential outliers in numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        outlier_cols = []
        
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_pct = ((df[col] < lower_bound) | (df[col] > upper_bound)).mean() * 100
            if outlier_pct:  # If there are outliers
                outlier_cols.append((col, outlier_pct))
        
        if outlier_cols:
            if len(outlier_cols) > 0:
                issues.append(f"Potential outliers detected in {len(outlier_cols)} numeric columns")
            else:
                cols_str = ", ".join([f"{col} ({pct:.1f}%)" for col, pct in outlier_cols])
                issues.append(f"Potential outliers detected in columns: {cols_str}")
        
        # Check for incorrect data types
        dtype_issues = []
        
        # Check for numeric columns stored as objects
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert sample to numeric
                sample = df[col].dropna().head(100)
                numeric_count = 0
                for val in sample:
                    try:
                        float(val)
                        numeric_count += 1
                    except:
                        pass
                
                # If more than 80% can be converted to numeric, it's likely a numeric column
                if numeric_count > 0.8 * len(sample) and len(sample) > 0:
                    dtype_issues.append(f"{col} (likely numeric)")
        
        # Check for date columns stored as objects
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to parse sample as dates
                sample = df[col].dropna().head(100)
                date_count = 0
                for val in sample:
                    try:
                        parse(str(val))
                        date_count += 1
                    except:
                        pass
                
                # If more than 80% can be parsed as dates, it's likely a date column
                if date_count > 0.8 * len(sample) and len(sample) > 0:
                    dtype_issues.append(f"{col} (likely date/time)")
        
        if dtype_issues:
            if len(dtype_issues) > 3:
                issues.append(f"Potential data type issues in {len(dtype_issues)} columns")
            else:
                cols_str = ", ".join(dtype_issues)
                issues.append(f"Potential data type issues in columns: {cols_str}")
        
        # Check for columns with high cardinality
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        high_card_cols = []
        
        for col in cat_cols:
            unique_pct = (df[col].nunique() / len(df)) * 100
            if unique_pct > 50 and df[col].nunique() > 100:
                high_card_cols.append((col, df[col].nunique()))
        
        if high_card_cols:
            if len(high_card_cols) > 3:
                issues.append(f"High cardinality detected in {len(high_card_cols)} categorical columns")
            else:
                cols_str = ", ".join([f"{col} ({count} unique values)" for col, count in high_card_cols])
                issues.append(f"High cardinality detected in columns: {cols_str}")
        
        # Check for highly skewed numeric distributions
        skewed_cols = []
        for col in numeric_cols:
            skew_val = df[col].skew()
            if abs(skew_val) > 3:
                skewed_cols.append((col, skew_val))
        
        if skewed_cols:
            if len(skewed_cols) > 3:
                issues.append(f"Highly skewed distributions in {len(skewed_cols)} numeric columns")
            else:
                cols_str = ", ".join([f"{col} (skew={skew:.1f})" for col, skew in skewed_cols])
                issues.append(f"Highly skewed distributions in columns: {cols_str}")
        
        # Check for inconsistent text formatting
        text_issue_cols = []
        for col in cat_cols:
            if df[col].nunique() < 15:  # Only check low-cardinality columns
                values = df[col].dropna().astype(str).unique()
                
                # Check for case inconsistency
                lower_values = [v.lower() for v in values]
                if len(set(lower_values)) < len(values):
                    text_issue_cols.append(col)
                    continue
                
                # Check for leading/trailing whitespace
                if any(v.strip() != v for v in values):
                    text_issue_cols.append(col)
                    continue
        
        if text_issue_cols:
            if len(text_issue_cols) > 3:
                issues.append(f"Inconsistent text formatting in {len(text_issue_cols)} categorical columns")
            else:
                cols_str = ", ".join(text_issue_cols)
                issues.append(f"Inconsistent text formatting in columns: {cols_str}")
        
        return issues

    @staticmethod
    def suggest_data_type(df, column):
        """
        Suggest an appropriate data type for a column
        
        Args:
            df (pandas.DataFrame): The dataframe
            column (str): The column name
            
        Returns:
            str: Suggested data type
        """
        if column not in df.columns:
            return "unknown"
        
        current_type = str(df[column].dtype)
        values = df[column].dropna()
        
        if len(values) == 0:
            return current_type
        
        # Check if already a datetime
        if pd.api.types.is_datetime64_dtype(df[column]):
            return "datetime64"
        
        # Check if boolean
        if set(values.unique()) <= {0, 1, True, False}:
            return "boolean"
        
        # Check if categorical
        if df[column].nunique() <= 20 and (current_type == 'object' or current_type == 'category'):
            return "category"
        
        # Check if date (if it's an object)
        if current_type == 'object':
            # Try to parse sample as dates
            sample = values.head(min(100, len(values)))
            date_count = 0
            for val in sample:
                try:
                    parse(str(val))
                    date_count += 1
                except:
                    pass
            
            # If more than 80% can be parsed as dates, suggest datetime
            if date_count > 0.8 * len(sample):
                return "datetime64"
        
        # Check if numeric (if it's an object)
        if current_type == 'object':
            # Try to convert sample to numeric
            sample = values.head(min(100, len(values)))
            numeric_count = 0
            float_needed = False
            
            for val in sample:
                try:
                    num = float(val)
                    numeric_count += 1
                    if num != int(num):
                        float_needed = True
                except:
                    pass
            
            # If more than 80% can be converted to numeric, suggest numeric type
            if numeric_count > 0.8 * len(sample):
                return "float64" if float_needed else "int64"
        
        # Check if integer could be float
        if current_type.startswith('int'):
            # Check if there are missing values (ints don't support NaN)
            if df[column].isna().any():
                return "float64"
        
        # Default to current type
        return current_type
    
    @staticmethod
    def clean_column_names(df):
        """
        Clean and standardize column names
        
        Args:
            df (pandas.DataFrame): The dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with cleaned column names
        """
        df = df.copy()
        
        # Function to clean individual column name
        def clean_name(name):
            # Convert to string
            name = str(name)
            # Replace spaces and special chars with underscore
            name = re.sub(r'[^\w\s]', '', name)
            name = re.sub(r'\s+', '_', name)
            # Convert to lowercase
            name = name.lower()
            # Remove leading/trailing underscores
            name = name.strip('_')
            # Ensure name is not empty and doesn't start with a number
            if name == '':
                name = 'column'
            if name[0].isdigit():
                name = 'col_' + name
            return name
        
        # Clean all column names
        df.columns = [clean_name(col) for col in df.columns]
        
        # Handle duplicate column names by adding suffix
        if len(df.columns) != len(set(df.columns)):
            cols = {}
            for i, col in enumerate(df.columns):
                if col in cols:
                    cols[col] += 1
                    df.columns.values[i] = f"{col}_{cols[col]}"
                else:
                    cols[col] = 0
        
        return df
    
    @staticmethod
    def handle_missing_values(df, strategy='auto'):
        """
        Handle missing values in the dataset
        
        Args:
            df (pandas.DataFrame): The dataframe
            strategy (str): Strategy for handling missing values ('auto', 'drop', 'impute')
            
        Returns:
            pandas.DataFrame: Dataframe with handled missing values
        """
        if df is None or df.empty:
            return df
        
        df = df.copy()
        
        # If no missing values, return the original dataframe
        if not df.isna().any().any():
            return df
        
        # Auto strategy determines the best approach based on data
        if strategy == 'auto':
            missing_cols = df.columns[df.isna().any()].tolist()
            
            for col in missing_cols:
                missing_pct = df[col].isna().mean() * 100
                
                # If more than 50% missing, drop the column
                if missing_pct > 50:
                    df = df.drop(columns=[col])
                
                # Otherwise impute the missing values
                else:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # For numeric columns, use median for imputation
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        # For non-numeric, use most frequent value
                        most_frequent = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                        df[col] = df[col].fillna(most_frequent)
        
        # Drop strategy removes rows with any missing values
        elif strategy == 'drop':
            df = df.dropna()
        
        # Impute strategy fills missing values
        elif strategy == 'impute':
            numeric_cols = df.select_dtypes(include=np.number).columns
            categorical_cols = df.select_dtypes(exclude=np.number).columns
            
            # Impute numeric columns with median
            if not numeric_cols.empty:
                imputer = SimpleImputer(strategy='median')
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
            # Impute categorical columns with most frequent value
            if not categorical_cols.empty:
                for col in categorical_cols:
                    if df[col].isna().any():
                        most_frequent = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                        df[col] = df[col].fillna(most_frequent)
        
        return df
    
    @staticmethod
    def encode_categorical_features(df, columns=None, method='auto'):
        """
        Encode categorical features for machine learning
        
        Args:
            df (pandas.DataFrame): The dataframe
            columns (list): List of columns to encode (None for auto-detection)
            method (str): Encoding method ('auto', 'onehot', 'label', 'ordinal')
            
        Returns:
            pandas.DataFrame: Dataframe with encoded features
        """
        if df is None or df.empty:
            return df, {}
        
        df = df.copy()
        encoders = {}
        
        # Auto-detect categorical columns if not specified
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Filter to only include columns that exist in the dataframe
        columns = [col for col in columns if col in df.columns]
        
        if not columns:
            return df, encoders
        
        # Auto method chooses encoding based on cardinality
        if method == 'auto':
            for col in columns:
                # If binary or low cardinality, use one-hot encoding
                if df[col].nunique() <= 10:
                    try:
                        # One-hot encoding
                        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                        encoded = encoder.fit_transform(df[[col]])
                        
                        # Create new column names
                        categories = encoder.categories_[0]
                        new_cols = [f"{col}_{cat}" for cat in categories]
                        
                        # Add encoded columns to dataframe
                        encoded_df = pd.DataFrame(encoded, columns=new_cols, index=df.index)
                        df = pd.concat([df, encoded_df], axis=1)
                        
                        # Store encoder for future use
                        encoders[col] = {
                            'encoder': encoder,
                            'method': 'onehot',
                            'new_columns': new_cols
                        }
                        
                        # Remove original column
                        df = df.drop(columns=[col])
                    except Exception as e:
                        # If one-hot encoding fails, leave the column as is
                        pass
                
                # For high cardinality, convert to category dtype
                # (sklearn can handle this automatically for decision trees)
                else:
                    df[col] = df[col].astype('category')
                    encoders[col] = {
                        'encoder': None,
                        'method': 'category',
                        'new_columns': [col]
                    }
        
        # One-hot encoding for all specified columns
        elif method == 'onehot':
            for col in columns:
                try:
                    # One-hot encoding
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(df[[col]])
                    
                    # Create new column names
                    categories = encoder.categories_[0]
                    new_cols = [f"{col}_{cat}" for cat in categories]
                    
                    # Add encoded columns to dataframe
                    encoded_df = pd.DataFrame(encoded, columns=new_cols, index=df.index)
                    df = pd.concat([df, encoded_df], axis=1)
                    
                    # Store encoder for future use
                    encoders[col] = {
                        'encoder': encoder,
                        'method': 'onehot',
                        'new_columns': new_cols
                    }
                    
                    # Remove original column
                    df = df.drop(columns=[col])
                except Exception as e:
                    # If one-hot encoding fails, leave the column as is
                    pass
        
        # For label and ordinal encoding, convert to category dtype
        elif method == 'label' or method == 'ordinal':
            for col in columns:
                df[col] = df[col].astype('category')
                encoders[col] = {
                    'encoder': None,
                    'method': 'category',
                    'new_columns': [col]
                }
        
        return df, encoders
    
    @staticmethod
    def scale_numeric_features(df, columns=None, method='standard'):
        """
        Scale numeric features for machine learning
        
        Args:
            df (pandas.DataFrame): The dataframe
            columns (list): List of columns to scale (None for all numeric)
            method (str): Scaling method ('standard', 'minmax')
            
        Returns:
            pandas.DataFrame: Dataframe with scaled features
        """
        if df is None or df.empty:
            return df, {}
        
        df = df.copy()
        scalers = {}
        
        # Auto-detect numeric columns if not specified
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()
        
        # Filter to only include columns that exist in the dataframe
        columns = [col for col in columns if col in df.columns]
        
        if not columns:
            return df, scalers
        
        # Standard scaling (mean=0, std=1)
        if method == 'standard':
            scaler = StandardScaler()
            df[columns] = scaler.fit_transform(df[columns])
            scalers['standard'] = {
                'scaler': scaler,
                'columns': columns
            }
        
        # Min-max scaling (0 to 1)
        elif method == 'minmax':
            scaler = MinMaxScaler()
            df[columns] = scaler.fit_transform(df[columns])
            scalers['minmax'] = {
                'scaler': scaler,
                'columns': columns
            }
        
        return df, scalers
    
    @staticmethod
    def detect_and_convert_types(df):
        """
        Automatically detect and convert column data types
        
        Args:
            df (pandas.DataFrame): The dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with appropriate data types
        """
        if df is None or df.empty:
            return df
        
        df = df.copy()
        
        # Try to convert object columns to more specific types
        for col in df.select_dtypes(include=['object']).columns:
            # Check if column can be converted to datetime
            try:
                date_series = pd.to_datetime(df[col], errors='raise')
                df[col] = date_series
                continue
            except:
                pass
            
            # Check if column can be converted to numeric
            try:
                numeric_series = pd.to_numeric(df[col], errors='raise')
                df[col] = numeric_series
                continue
            except:
                pass
            
            # Check if column is categorical (low cardinality)
            if df[col].nunique() < 20:
                df[col] = df[col].astype('category')
        
        return df
    
    @staticmethod
    def detect_outliers(df, column, method='iqr'):
        """
        Detect outliers in a numeric column
        
        Args:
            df (pandas.DataFrame): The dataframe
            column (str): Column name
            method (str): Detection method ('iqr', 'zscore')
            
        Returns:
            pandas.Series: Boolean mask of outliers
        """
        if column not in df.columns:
            return pd.Series(False, index=df.index)
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            return pd.Series(False, index=df.index)
        
        # IQR method (>1.5*IQR from Q1/Q3)
        if method == 'iqr':
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return (df[column] < lower_bound) | (df[column] > upper_bound)
        
        # Z-score method (>3 std devs from mean)
        elif method == 'zscore':
            z_scores = (df[column] - df[column].mean()) / df[column].std()
            return abs(z_scores) > 3
        
        # Default to no outliers
        return pd.Series(False, index=df.index)

    # Auto detect and fix data type
    @staticmethod
    def enhanced_auto_infer_data_types(df, date_formats=None, threshold=0.8, min_sample_size=100):
        """
        Enhanced automatic data type inference with support for:
        - Boolean detection
        - Custom date formats
        - Numeric validation
        - Categorical optimization
        """
        new_df = df.copy()
        changes = []
        
        # Default date formats to try
        default_date_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', 
            '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y',
            '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S'
        ]
        date_formats = date_formats or default_date_formats
        
        # Boolean mapping
        bool_values = {
            True: ['true', 'yes', '1', 't', 'y'],
            False: ['false', 'no', '0', 'f', 'n']
        }
        
        for col in new_df.columns:
            original_type = new_df[col].dtype
            sample = new_df[col].dropna().head(min_sample_size)
            
            if len(sample) == 0:
                continue
                
            # Try boolean conversion
            if original_type == object:
                bool_count = 0
                sample_lower = sample.astype(str).str.lower()
                for val in sample_lower:
                    if val in bool_values[True] or val in bool_values[False]:
                        bool_count += 1
                
                if bool_count / len(sample) >= threshold:
                    # Convert to boolean
                    bool_map = {v: k for k, lst in bool_values.items() for v in lst}
                    new_df[col] = new_df[col].astype(str).str.lower().map(bool_map)
                    changes.append({
                        'column': col,
                        'original_type': original_type,
                        'new_type': 'boolean',
                        'confidence': bool_count / len(sample)
                    })
                    continue
            
            # Try numeric conversion
            try:
                numeric_col = pd.to_numeric(sample, errors='raise')
                if isinstance(numeric_col, pd.Series):
                    # Determine if int or float
                    if numeric_col.dtype == np.int64 or numeric_col.apply(float.is_integer).all():
                        new_df[col] = pd.to_numeric(new_df[col], errors='coerce').astype('Int64')  # nullable integer
                    else:
                        new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                    changes.append({
                        'column': col,
                        'original_type': original_type,
                        'new_type': str(new_df[col].dtype),
                        'confidence': 0.9
                    })
                    continue
            except:
                pass
            
            # Try datetime conversion
            date_success = False
            for date_format in date_formats:
                try:
                    datetime_col = pd.to_datetime(sample, format=date_format, errors='raise')
                    if isinstance(datetime_col, pd.Series):
                        new_df[col] = pd.to_datetime(new_df[col], format=date_format, errors='coerce')
                        changes.append({
                            'column': col,
                            'original_type': original_type,
                            'new_type': 'datetime64[ns]',
                            'confidence': 0.85,
                            'format': date_format
                        })
                        date_success = True
                        break
                except:
                    continue
            
            if date_success:
                continue
                
            # Consider categorical for low-cardinality object columns
            if original_type == object:
                nunique = new_df[col].nunique()
                if nunique < min(20, len(new_df) * 0.05):  # Less than 5% unique values
                    new_df[col] = new_df[col].astype('category')
                    changes.append({
                        'column': col,
                        'original_type': original_type,
                        'new_type': 'category',
                        'confidence': 0.7,
                        'unique_values': nunique
                    })
        
        return new_df, changes

    @staticmethod
    def apply_inferred_types(df, changes, confirm=True):
        """
        Apply inferred type changes with optional confirmation
        """
        if confirm:
            st.write("### Suggested Data Type Changes")
            changes_df = pd.DataFrame(changes)
            changes_df['confidence'] = changes_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(changes_df)
            
            if st.checkbox("Apply suggested changes?", value=True):
                return df
        
        return df