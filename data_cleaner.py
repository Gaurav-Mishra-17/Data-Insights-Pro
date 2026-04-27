import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import re

def identify_cleaning_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify various data cleaning issues in the dataset
    
    Args:
        df: The pandas dataframe to analyze
        
    Returns:
        Dictionary of issues identified for each column
    """
    issues = {}
    
    # Analyze each column
    for col in df.columns:
        col_issues = {}
        
        # Missing values
        missing_count = df[col].isna().sum()
        missing_percent = missing_count / len(df) * 100
        if missing_count > 0:
            col_issues['missing_values'] = {
                'count': int(missing_count),
                'percentage': round(missing_percent, 2)
            }
        
        # Check data type specific issues
        dtype = df[col].dtype
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(dtype):
            # Outliers detection
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            
            if len(outliers) > 0:
                col_issues['outliers'] = {
                    'count': int(len(outliers)),
                    'percentage': round(len(outliers) / len(df) * 100, 2),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
            
            # Zero values (for ratios or variables that shouldn't be zero)
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                col_issues['zeros'] = {
                    'count': int(zero_count),
                    'percentage': round(zero_count / len(df) * 100, 2)
                }
            
            # Negative values (for variables that should be positive)
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                col_issues['negatives'] = {
                    'count': int(negative_count),
                    'percentage': round(negative_count / len(df) * 100, 2)
                }
        
        # Categorical/String columns
        elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            # Try to convert to numeric to check if it's mistyped
            try:
                numeric_conversion = pd.to_numeric(df[col], errors='coerce')
                conversion_success = numeric_conversion.notna().sum()
                # If most values can be converted, suggest type change
                if conversion_success > 0.8 * df[col].count():
                    col_issues['possible_numeric'] = {
                        'convertible_count': int(conversion_success),
                        'convertible_percentage': round(conversion_success / df[col].count() * 100, 2)
                    }
            except:
                pass
            
            # Check for inconsistent capitalization
            if pd.api.types.is_string_dtype(dtype):
                # Get non-null string values
                string_vals = df[col].dropna().astype(str)
                
                # Check for inconsistent case
                if not string_vals.empty:
                    lowercase = string_vals.str.lower()
                    unique_normal = string_vals.nunique()
                    unique_lower = lowercase.nunique()
                    
                    if unique_lower < unique_normal:
                        col_issues['inconsistent_capitalization'] = {
                            'original_unique': int(unique_normal),
                            'lowercase_unique': int(unique_lower),
                            'difference': int(unique_normal - unique_lower)
                        }
                    
                    # Check for leading/trailing whitespace
                    trimmed = string_vals.str.strip()
                    if (trimmed != string_vals).any():
                        col_issues['whitespace_issues'] = {
                            'count': int((trimmed != string_vals).sum()),
                            'percentage': round((trimmed != string_vals).sum() / len(df) * 100, 2)
                        }
                    
                    # Check for special characters that might be unwanted
                    special_chars = string_vals.str.contains(r'[^a-zA-Z0-9\s]', regex=True)
                    if special_chars.any():
                        col_issues['special_characters'] = {
                            'count': int(special_chars.sum()),
                            'percentage': round(special_chars.sum() / len(df) * 100, 2)
                        }
            
            # Check for high cardinality
            unique_count = df[col].nunique()
            if unique_count > 50 and unique_count > 0.5 * len(df):
                col_issues['high_cardinality'] = {
                    'unique_count': int(unique_count),
                    'percentage': round(unique_count / len(df) * 100, 2)
                }
        
        # Datetime columns or potentially datetime
        # Check if string column could be datetime
        if pd.api.types.is_string_dtype(dtype):
            try:
                datetime_conversion = pd.to_datetime(df[col], errors='coerce')
                conversion_success = datetime_conversion.notna().sum()
                
                # If most values can be converted, suggest type change
                if conversion_success > 0.8 * df[col].count():
                    col_issues['possible_datetime'] = {
                        'convertible_count': int(conversion_success),
                        'convertible_percentage': round(conversion_success / df[col].count() * 100, 2)
                    }
            except:
                pass
        
        # Add column issues if any were found
        if col_issues:
            issues[col] = col_issues
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues['global'] = {
            'duplicate_rows': {
                'count': int(duplicate_count),
                'percentage': round(duplicate_count / len(df) * 100, 2)
            }
        }
    
    return issues

def suggest_cleaning_actions(issues: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest cleaning actions based on the identified issues
    
    Args:
        issues: Dictionary of issues per column
        
    Returns:
        Dictionary of suggested actions per column
    """
    suggestions = {}
    
    for col, col_issues in issues.items():
        col_suggestions = []
        
        # Don't process the global issues here
        if col == 'global':
            suggestions[col] = {
                'duplicate_rows': 'Remove duplicate rows'
            }
            continue
        
        # Missing values
        if 'missing_values' in col_issues:
            missing_percent = col_issues['missing_values']['percentage']
            
            if missing_percent > 80:
                col_suggestions.append('Drop column due to excessive missing values')
            elif missing_percent > 30:
                col_suggestions.append('Consider dropping column or imputing with median/mode')
            else:
                col_suggestions.append('Impute missing values with mean/median/mode')
        
        # Outliers
        if 'outliers' in col_issues:
            outlier_percent = col_issues['outliers']['percentage']
            
            if outlier_percent > 10:
                col_suggestions.append('Investigate outliers - could indicate data quality issues')
            elif outlier_percent > 2:
                col_suggestions.append('Cap/floor outliers or transform data (log, sqrt)')
            else:
                col_suggestions.append('Remove or cap outliers')
        
        # Zero values
        if 'zeros' in col_issues:
            col_suggestions.append('Check if zeros are valid or should be treated as missing')
        
        # Negative values
        if 'negatives' in col_issues:
            col_suggestions.append('Verify if negative values are valid for this variable')
        
        # Type conversion - numeric
        if 'possible_numeric' in col_issues:
            col_suggestions.append('Convert to numeric data type')
        
        # Type conversion - datetime
        if 'possible_datetime' in col_issues:
            col_suggestions.append('Convert to datetime data type')
        
        # Inconsistent capitalization
        if 'inconsistent_capitalization' in col_issues:
            col_suggestions.append('Standardize text case (convert all to lowercase or title case)')
        
        # Whitespace issues
        if 'whitespace_issues' in col_issues:
            col_suggestions.append('Remove leading/trailing whitespace')
        
        # Special characters
        if 'special_characters' in col_issues:
            col_suggestions.append('Clean special characters if not required')
        
        # High cardinality
        if 'high_cardinality' in col_issues:
            col_suggestions.append('Consider encoding or grouping categories')
        
        suggestions[col] = col_suggestions
    
    return suggestions

def clean_data(df: pd.DataFrame, action: str, column: str = None, params: Dict = None) -> Tuple[pd.DataFrame, str]:
    """
    Perform a cleaning action on the dataframe
    
    Args:
        df: The pandas dataframe to clean
        action: The cleaning action to perform
        column: The column to apply the action to (if applicable)
        params: Additional parameters for the cleaning action
        
    Returns:
        Tuple containing the cleaned dataframe and a description of the action performed
    """
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    desc = ""
    
    if action == "remove_duplicates":
        # Remove duplicate rows
        before_count = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        after_count = len(cleaned_df)
        desc = f"Removed {before_count - after_count} duplicate rows"
    
    elif action == "drop_column":
        # Drop a column
        if column:
            cleaned_df = cleaned_df.drop(columns=[column])
            desc = f"Dropped column '{column}'"
    
    elif action == "impute_mean":
        # Impute missing values with mean
        if column and pd.api.types.is_numeric_dtype(cleaned_df[column].dtype):
            mean_value = cleaned_df[column].mean()
            cleaned_df[column] = cleaned_df[column].fillna(mean_value)
            desc = f"Imputed missing values in '{column}' with mean: {mean_value:.2f}"
    
    elif action == "impute_median":
        # Impute missing values with median
        if column and pd.api.types.is_numeric_dtype(cleaned_df[column].dtype):
            median_value = cleaned_df[column].median()
            cleaned_df[column] = cleaned_df[column].fillna(median_value)
            desc = f"Imputed missing values in '{column}' with median: {median_value:.2f}"
    
    elif action == "impute_mode":
        # Impute missing values with mode
        if column:
            mode_value = cleaned_df[column].mode()[0]
            cleaned_df[column] = cleaned_df[column].fillna(mode_value)
            desc = f"Imputed missing values in '{column}' with mode: {mode_value}"
    
    elif action == "impute_constant":
        # Impute missing values with a constant
        if column and params and 'value' in params:
            value = params['value']
            cleaned_df[column] = cleaned_df[column].fillna(value)
            desc = f"Imputed missing values in '{column}' with constant: {value}"
    
    elif action == "convert_numeric":
        # Convert column to numeric
        if column:
            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
            desc = f"Converted '{column}' to numeric type"
    
    elif action == "convert_datetime":
        # Convert column to datetime
        if column:
            format_str = params.get('format', None) if params else None
            if format_str:
                cleaned_df[column] = pd.to_datetime(cleaned_df[column], format=format_str, errors='coerce')
                desc = f"Converted '{column}' to datetime with format: {format_str}"
            else:
                cleaned_df[column] = pd.to_datetime(cleaned_df[column], errors='coerce')
                desc = f"Converted '{column}' to datetime"
    
    elif action == "to_lowercase":
        # Convert text to lowercase
        if column and pd.api.types.is_string_dtype(cleaned_df[column].dtype):
            cleaned_df[column] = cleaned_df[column].str.lower()
            desc = f"Converted '{column}' to lowercase"
    
    elif action == "to_uppercase":
        # Convert text to uppercase
        if column and pd.api.types.is_string_dtype(cleaned_df[column].dtype):
            cleaned_df[column] = cleaned_df[column].str.upper()
            desc = f"Converted '{column}' to uppercase"
    
    elif action == "to_titlecase":
        # Convert text to title case
        if column and pd.api.types.is_string_dtype(cleaned_df[column].dtype):
            cleaned_df[column] = cleaned_df[column].str.title()
            desc = f"Converted '{column}' to title case"
    
    elif action == "strip_whitespace":
        # Remove leading/trailing whitespace
        if column and pd.api.types.is_string_dtype(cleaned_df[column].dtype):
            cleaned_df[column] = cleaned_df[column].str.strip()
            desc = f"Stripped whitespace from '{column}'"
    
    elif action == "remove_special_chars":
        # Remove special characters
        if column and pd.api.types.is_string_dtype(cleaned_df[column].dtype):
            cleaned_df[column] = cleaned_df[column].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
            desc = f"Removed special characters from '{column}'"
    
    elif action == "cap_outliers":
        # Cap outliers to Q1-1.5*IQR and Q3+1.5*IQR
        if column and pd.api.types.is_numeric_dtype(cleaned_df[column].dtype):
            q1 = cleaned_df[column].quantile(0.25)
            q3 = cleaned_df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Count outliers before capping
            outliers_count = ((cleaned_df[column] < lower_bound) | (cleaned_df[column] > upper_bound)).sum()
            
            # Apply capping
            cleaned_df[column] = cleaned_df[column].clip(lower=lower_bound, upper=upper_bound)
            
            desc = f"Capped {outliers_count} outliers in '{column}' to range [{lower_bound:.2f}, {upper_bound:.2f}]"
    
    elif action == "remove_outliers":
        # Remove rows with outliers
        if column and pd.api.types.is_numeric_dtype(cleaned_df[column].dtype):
            q1 = cleaned_df[column].quantile(0.25)
            q3 = cleaned_df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Count rows before removal
            before_count = len(cleaned_df)
            
            # Remove outliers
            cleaned_df = cleaned_df[(cleaned_df[column] >= lower_bound) & (cleaned_df[column] <= upper_bound)]
            
            # Count rows after removal
            after_count = len(cleaned_df)
            
            desc = f"Removed {before_count - after_count} rows with outliers in '{column}'"
    
    elif action == "replace_values":
        # Replace specific values
        if column and params and 'old_value' in params and 'new_value' in params:
            old_value = params['old_value']
            new_value = params['new_value']
            
            # Count occurrences before replacement
            if pd.api.types.is_numeric_dtype(cleaned_df[column].dtype):
                occurrences = (cleaned_df[column] == old_value).sum()
            else:
                occurrences = (cleaned_df[column].astype(str) == str(old_value)).sum()
            
            # Replace values
            cleaned_df[column] = cleaned_df[column].replace(old_value, new_value)
            
            desc = f"Replaced {occurrences} occurrences of '{old_value}' with '{new_value}' in '{column}'"
    
    elif action == "log_transform":
        # Apply log transformation
        if column and pd.api.types.is_numeric_dtype(cleaned_df[column].dtype):
            # Add a small constant to handle zeros
            min_val = cleaned_df[column].min()
            offset = 0
            if min_val <= 0:
                offset = abs(min_val) + 1
            
            cleaned_df[f"{column}_log"] = np.log(cleaned_df[column] + offset)
            desc = f"Created log-transformed version of '{column}' as '{column}_log'"
    
    elif action == "sqrt_transform":
        # Apply square root transformation
        if column and pd.api.types.is_numeric_dtype(cleaned_df[column].dtype):
            # Add a small constant to handle zeros
            min_val = cleaned_df[column].min()
            offset = 0
            if min_val < 0:
                offset = abs(min_val)
            
            cleaned_df[f"{column}_sqrt"] = np.sqrt(cleaned_df[column] + offset)
            desc = f"Created square root-transformed version of '{column}' as '{column}_sqrt'"
    
    elif action == "bin_numeric":
        # Bin numeric data into categories
        if column and pd.api.types.is_numeric_dtype(cleaned_df[column].dtype) and params and 'bins' in params:
            bins = params['bins']
            labels = params.get('labels', None)
            
            if labels and len(labels) != bins:
                labels = None
            
            cleaned_df[f"{column}_binned"] = pd.cut(cleaned_df[column], bins=bins, labels=labels)
            desc = f"Created binned version of '{column}' as '{column}_binned' with {bins} bins"
    
    elif action == "one_hot_encode":
        # One-hot encode categorical variable
        if column and (pd.api.types.is_string_dtype(cleaned_df[column].dtype) or 
                      pd.api.types.is_categorical_dtype(cleaned_df[column].dtype)):
            
            # Get one-hot encoded columns
            dummies = pd.get_dummies(cleaned_df[column], prefix=column)
            
            # Add the new columns to the dataframe
            for dummy_col in dummies.columns:
                cleaned_df[dummy_col] = dummies[dummy_col]
            
            desc = f"One-hot encoded '{column}' creating {len(dummies.columns)} new columns"
    
    elif action == "normalize":
        # Normalize numeric column (min-max scaling)
        if column and pd.api.types.is_numeric_dtype(cleaned_df[column].dtype):
            min_val = cleaned_df[column].min()
            max_val = cleaned_df[column].max()
            
            if min_val != max_val:  # Avoid division by zero
                cleaned_df[f"{column}_norm"] = (cleaned_df[column] - min_val) / (max_val - min_val)
                desc = f"Created normalized version of '{column}' as '{column}_norm' using min-max scaling"
            else:
                desc = f"Could not normalize '{column}' - all values are identical"
    
    elif action == "standardize":
        # Standardize numeric column (z-score)
        if column and pd.api.types.is_numeric_dtype(cleaned_df[column].dtype):
            mean_val = cleaned_df[column].mean()
            std_val = cleaned_df[column].std()
            
            if std_val > 0:  # Avoid division by zero
                cleaned_df[f"{column}_std"] = (cleaned_df[column] - mean_val) / std_val
                desc = f"Created standardized version of '{column}' as '{column}_std' using z-score"
            else:
                desc = f"Could not standardize '{column}' - standard deviation is zero"
    
    return cleaned_df, desc

def get_cleaning_history(df_original: pd.DataFrame, df_current: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze differences between the original and current dataframe
    
    Args:
        df_original: The original dataframe
        df_current: The current dataframe after cleaning
        
    Returns:
        Dictionary summarizing the changes made
    """
    changes = {}
    
    # Check for row count changes
    original_rows = len(df_original)
    current_rows = len(df_current)
    if original_rows != current_rows:
        changes['rows'] = {
            'original': original_rows,
            'current': current_rows,
            'difference': current_rows - original_rows
        }
    
    # Check for column changes
    original_cols = set(df_original.columns)
    current_cols = set(df_current.columns)
    
    # New columns
    new_cols = current_cols - original_cols
    if new_cols:
        changes['new_columns'] = list(new_cols)
    
    # Removed columns
    removed_cols = original_cols - current_cols
    if removed_cols:
        changes['removed_columns'] = list(removed_cols)
    
    # Changed data types
    changed_dtypes = {}
    for col in original_cols.intersection(current_cols):
        if df_original[col].dtype != df_current[col].dtype:
            changed_dtypes[col] = {
                'original': str(df_original[col].dtype),
                'current': str(df_current[col].dtype)
            }
    
    if changed_dtypes:
        changes['changed_dtypes'] = changed_dtypes
    
    # Missing values changes
    missing_changes = {}
    for col in original_cols.intersection(current_cols):
        original_missing = df_original[col].isna().sum()
        current_missing = df_current[col].isna().sum()
        
        if original_missing != current_missing:
            missing_changes[col] = {
                'original': int(original_missing),
                'current': int(current_missing),
                'difference': int(current_missing - original_missing)
            }
    
    if missing_changes:
        changes['missing_values'] = missing_changes
    
    return changes
