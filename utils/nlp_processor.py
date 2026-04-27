import pandas as pd
import numpy as np
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


class NLPProcessor:
    """
    Class for processing natural language queries about data
    """
    
    def __init__(self):
        self.keywords = {
            "show": ["display", "list", "get", "find", "give", "show me", "present"],
            "top": ["highest", "most", "maximum", "max", "top", "largest", "biggest"],
            "bottom": ["lowest", "least", "minimum", "min", "bottom", "smallest"],
            "average": ["mean", "avg", "average"],
            "count": ["count", "how many", "number of", "total"],
            "sum": ["sum", "total", "add up"],
            "distribution": ["distribution", "histogram", "spread", "range"],
            "correlation": ["correlation", "relationship", "related", "associate", "connection", "compare"],
            "trend": ["trend", "pattern", "change", "over time", "time series"],
            "grouped": ["group by", "grouped by", "per", "by", "across", "segmented by"],
            "filter": ["where", "if", "when", "filter", "only", "just", "with"],
            "sort": ["sort", "order", "arrange", "rank"]
        }
        
        self.operations = {
            "summary": self._get_summary,
            "show_rows": self._show_rows,
            "filter_data": self._filter_data,
            "compute_statistic": self._compute_statistic,
            "correlation": self._compute_correlation,
            "distribution": self._show_distribution,
            "time_series": self._show_time_series,
            "group_by": self._group_data,
            "compare": self._compare_data
        }
    
    def process_query(self, df, query):
        """
        Process a natural language query and return relevant data insights
        
        Args:
            df (pandas.DataFrame): The dataframe to analyze
            query (str): Natural language query
            
        Returns:
            dict: Response with interpretation, data, visualization and explanation
        """
        if df is None or df.empty:
            return {
                "interpretation": "I couldn't process the query because no data is available.",
                "data_result": None,
                "visualization": None,
                "explanation": "Please upload a dataset first."
            }
        
        # Convert query to lowercase for easier matching
        query_lower = query.lower()
        
        # Get dataframe columns for reference
        columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Try to identify potential datetime columns that aren't already recognized as such
        potential_datetime_cols = []
        for col in categorical_cols:
            if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower():
                potential_datetime_cols.append(col)
        
        # Identify mentioned columns in the query
        mentioned_columns = []
        for col in columns:
            if col.lower() in query_lower or col.lower().replace('_', ' ') in query_lower:
                mentioned_columns.append(col)
        
        # Determine the operation type from the query
        operation = self._determine_operation(query_lower, mentioned_columns, numeric_cols, categorical_cols, datetime_cols + potential_datetime_cols, df)
        
        # Execute the operation
        try:
            if operation["type"] in self.operations:
                result = self.operations[operation["type"]](df, operation, query_lower)
                
                # Add interpretation based on the operation
                result["interpretation"] = self._generate_interpretation(operation, mentioned_columns)
                
                return result
            else:
                # If no specific operation is detected, provide a general summary
                return self._get_summary(df, operation, query_lower)
                
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "interpretation": "I'm sorry, I couldn't process that query correctly.",
                "data_result": None,
                "visualization": None,
                "explanation": str(e)
            }
    
    def _determine_operation(self, query, mentioned_columns, numeric_cols, categorical_cols, datetime_cols, df):
        """Determine what operation to perform based on the query"""
        operation = {"type": None, "params": {}}
        
        # Check for general summary request
        if any(keyword in query for keyword in ["summary", "overview", "describe", "summarize", "tell me about"]):
            operation["type"] = "summary"
            return operation
        
        # Check for show rows request
        if any(keyword in query for keyword in self.keywords["show"]):
            if any(keyword in query for keyword in ["first", "top", "head"]) and not any(keyword in query for keyword in self.keywords["top"]):
                operation["type"] = "show_rows"
                operation["params"]["position"] = "first"
                # Try to extract number of rows
                num_match = re.search(r'(?:top|first|head)\s+(\d+)', query)
                if num_match:
                    operation["params"]["n"] = int(num_match.group(1))
                else:
                    operation["params"]["n"] = 5
                return operation
            
            if any(keyword in query for keyword in ["last", "bottom", "tail"]) and not any(keyword in query for keyword in self.keywords["bottom"]):
                operation["type"] = "show_rows"
                operation["params"]["position"] = "last"
                # Try to extract number of rows
                num_match = re.search(r'(?:bottom|last|tail)\s+(\d+)', query)
                if num_match:
                    operation["params"]["n"] = int(num_match.group(1))
                else:
                    operation["params"]["n"] = 5
                return operation
        
        # Check for distribution analysis
        if any(keyword in query for keyword in self.keywords["distribution"]):
            operation["type"] = "distribution"
            
            # Try to identify which column to use
            for col in mentioned_columns:
                if col in numeric_cols:
                    operation["params"]["column"] = col
                    break
            
            # If no specific numeric column mentioned, use the first numeric column
            if "column" not in operation["params"] and numeric_cols:
                operation["params"]["column"] = numeric_cols[0]
                
            return operation
        
        # Check for correlation analysis
        if any(keyword in query for keyword in self.keywords["correlation"]):
            operation["type"] = "correlation"
            
            numeric_mentioned = [col for col in mentioned_columns if col in numeric_cols]
            
            # If specific columns mentioned, use them
            if len(numeric_mentioned) >= 2:
                operation["params"]["columns"] = numeric_mentioned[:2]
            elif len(numeric_mentioned) == 1 and len(numeric_cols) > 1:
                # If only one mentioned, pair it with the most correlated column
                col1 = numeric_mentioned[0]
                corr = df[numeric_cols].corr()[col1].abs().sort_values(ascending=False)
                col2 = corr.index[1]  # Index 0 would be the column itself
                operation["params"]["columns"] = [col1, col2]
            else:
                # If no specific columns, use all numeric columns for correlation matrix
                operation["params"]["columns"] = numeric_cols[:10]  # Limit to 10 columns
            
            return operation
        
        # Check for time series analysis
        if any(keyword in query for keyword in self.keywords["trend"]):
            operation["type"] = "time_series"
            
            # Find datetime column
            date_col = None
            for col in mentioned_columns:
                if col in datetime_cols:
                    date_col = col
                    break
            
            if not date_col and datetime_cols:
                date_col = datetime_cols[0]
            
            # Find value column
            value_col = None
            for col in mentioned_columns:
                if col in numeric_cols:
                    value_col = col
                    break
            
            if not value_col and numeric_cols:
                value_col = numeric_cols[0]
            
            if date_col and value_col:
                operation["params"]["date_column"] = date_col
                operation["params"]["value_column"] = value_col
                return operation
        
        # Check for group by analysis
        if any(keyword in query for keyword in self.keywords["grouped"]):
            operation["type"] = "group_by"
            
            # Find group column (categorical)
            group_col = None
            value_col = None
            
            for col in mentioned_columns:
                if col in categorical_cols and not group_col:
                    group_col = col
                elif col in numeric_cols and not value_col:
                    value_col = col
            
            if not group_col and categorical_cols:
                group_col = categorical_cols[0]
            
            if not value_col and numeric_cols:
                value_col = numeric_cols[0]
            
            if group_col:
                operation["params"]["group_column"] = group_col
                operation["params"]["value_column"] = value_col
                
                # Determine aggregation function
                if any(keyword in query for keyword in self.keywords["average"]):
                    operation["params"]["agg_func"] = "mean"
                elif any(keyword in query for keyword in self.keywords["sum"]):
                    operation["params"]["agg_func"] = "sum"
                elif any(keyword in query for keyword in self.keywords["count"]):
                    operation["params"]["agg_func"] = "count"
                else:
                    operation["params"]["agg_func"] = "mean"  # Default
                
                return operation
        
        # Check for statistic computation
        if any(keyword in query for keyword in self.keywords["average"] + self.keywords["sum"] + self.keywords["count"] + 
               self.keywords["top"] + self.keywords["bottom"]):
            operation["type"] = "compute_statistic"
            
            # Determine statistic type
            if any(keyword in query for keyword in self.keywords["average"]):
                operation["params"]["statistic"] = "mean"
            elif any(keyword in query for keyword in self.keywords["sum"]):
                operation["params"]["statistic"] = "sum"
            elif any(keyword in query for keyword in self.keywords["count"]):
                operation["params"]["statistic"] = "count"
            elif any(keyword in query for keyword in self.keywords["top"]):
                operation["params"]["statistic"] = "max"
            elif any(keyword in query for keyword in self.keywords["bottom"]):
                operation["params"]["statistic"] = "min"
            
            # Determine column
            for col in mentioned_columns:
                if col in numeric_cols:
                    operation["params"]["column"] = col
                    break
            
            if "column" not in operation["params"] and numeric_cols:
                operation["params"]["column"] = numeric_cols[0]
            
            return operation
        
        # Check for comparison between categories
        if any(keyword in query for keyword in ["compare", "comparison", "versus", "vs", "against"]):
            operation["type"] = "compare"
            
            # Find categorical and numeric columns
            cat_col = None
            value_col = None
            
            for col in mentioned_columns:
                if col in categorical_cols and not cat_col:
                    cat_col = col
                elif col in numeric_cols and not value_col:
                    value_col = col
            
            if not cat_col and categorical_cols:
                cat_col = categorical_cols[0]
            
            if not value_col and numeric_cols:
                value_col = numeric_cols[0]
            
            if cat_col and value_col:
                operation["params"]["category_column"] = cat_col
                operation["params"]["value_column"] = value_col
                return operation
        
        # Default to summary if no specific operation is detected
        if operation["type"] is None:
            operation["type"] = "summary"
            
        return operation
    
    def _generate_interpretation(self, operation, mentioned_columns):
        """Generate a natural language interpretation of the operation"""
        op_type = operation["type"]
        params = operation["params"]
        
        if op_type == "summary":
            return "Providing a summary of the dataset"
        
        elif op_type == "show_rows":
            position = params.get("position", "first")
            n = params.get("n", 5)
            return f"Showing the {position} {n} rows of the dataset"
        
        elif op_type == "distribution":
            column = params.get("column", "")
            return f"Analyzing the distribution of {column}"
        
        elif op_type == "correlation":
            columns = params.get("columns", [])
            if len(columns) == 2:
                return f"Examining the correlation between {columns[0]} and {columns[1]}"
            else:
                return "Analyzing correlations between multiple columns"
        
        elif op_type == "time_series":
            date_col = params.get("date_column", "")
            value_col = params.get("value_column", "")
            return f"Analyzing trends in {value_col} over {date_col}"
        
        elif op_type == "group_by":
            group_col = params.get("group_column", "")
            value_col = params.get("value_column", "")
            agg_func = params.get("agg_func", "mean")
            
            if agg_func == "mean":
                agg_desc = "average"
            else:
                agg_desc = agg_func
                
            return f"Calculating {agg_desc} {value_col} grouped by {group_col}"
        
        elif op_type == "compute_statistic":
            statistic = params.get("statistic", "")
            column = params.get("column", "")
            
            if statistic == "mean":
                stat_desc = "average"
            elif statistic == "max":
                stat_desc = "maximum"
            elif statistic == "min":
                stat_desc = "minimum"
            else:
                stat_desc = statistic
                
            return f"Computing the {stat_desc} of {column}"
        
        elif op_type == "compare":
            cat_col = params.get("category_column", "")
            value_col = params.get("value_column", "")
            return f"Comparing {value_col} across different {cat_col} categories"
        
        else:
            return "Analyzing the dataset"
    
    def _get_summary(self, df, operation, query):
        """Generate a summary of the dataframe"""
        # Get basic info
        num_rows, num_cols = df.shape
        missing = df.isna().sum().sum()
        missing_pct = (missing / (num_rows * num_cols)) * 100
        
        # Get column types
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Create summary text
        summary_text = f"""
        This dataset contains {num_rows:,} rows and {num_cols} columns.
        - {len(numeric_cols)} numeric columns: {', '.join(numeric_cols[:5])}{"..." if len(numeric_cols) > 5 else ""}
        - {len(categorical_cols)} categorical columns: {', '.join(categorical_cols[:5])}{"..." if len(categorical_cols) > 5 else ""}
        - {len(datetime_cols)} datetime columns: {', '.join(datetime_cols)}
        
        Missing values: {missing:,} ({missing_pct:.2f}% of total)
        """
        
        # Create a visualization for column types
        fig = go.Figure(data=[
            go.Bar(
                x=['Numeric', 'Categorical', 'Datetime'], 
                y=[len(numeric_cols), len(categorical_cols), len(datetime_cols)],
                marker_color=['#0078D7', '#50B8F0', '#83D0F5']
            )
        ])
        
        fig.update_layout(
            title="Column Types in Dataset",
            xaxis_title="Column Type",
            yaxis_title="Count",
            height=400
        )
        
        return {
            "interpretation": "Generating a summary of the dataset",
            "data_result": df.head(),
            "visualization": fig,
            "explanation": summary_text
        }
    
    def _show_rows(self, df, operation, query):
        """Show first or last rows of the dataframe"""
        params = operation["params"]
        position = params.get("position", "first")
        n = params.get("n", 5)
        
        if position == "first":
            result_df = df.head(n)
            explanation = f"Showing the first {n} rows of the dataset"
        else:
            result_df = df.tail(n)
            explanation = f"Showing the last {n} rows of the dataset"
        
        return {
            "interpretation": explanation,
            "data_result": result_df,
            "visualization": None,
            "explanation": explanation
        }
    
    def _filter_data(self, df, operation, query):
        """Filter dataframe based on conditions extracted from query"""
        # This is a simplified implementation
        # In a full implementation, we would extract filter conditions from the query
        # but that requires more complex NLP techniques
        return {
            "interpretation": "I couldn't determine specific filtering criteria from your query.",
            "data_result": df.head(),
            "visualization": None,
            "explanation": "To filter data, please specify the column and condition more clearly."
        }
    
    def _compute_statistic(self, df, operation, query):
        """Compute a statistic on a column"""
        params = operation["params"]
        statistic = params.get("statistic", "mean")
        column = params.get("column", None)
        
        if not column or column not in df.columns:
            return {
                "interpretation": f"Couldn't compute {statistic}, column not found.",
                "data_result": None,
                "visualization": None,
                "explanation": f"The column '{column}' was not found in the dataset."
            }
        
        # Compute the statistic
        if statistic == "mean":
            result = df[column].mean()
            text_stat = "mean (average)"
        elif statistic == "sum":
            result = df[column].sum()
            text_stat = "sum (total)"
        elif statistic == "count":
            result = df[column].count()
            text_stat = "count (number of non-missing values)"
        elif statistic == "max":
            result = df[column].max()
            text_stat = "maximum value"
        elif statistic == "min":
            result = df[column].min()
            text_stat = "minimum value"
        else:
            return {
                "interpretation": f"Unsupported statistic: {statistic}",
                "data_result": None,
                "visualization": None,
                "explanation": f"The statistic '{statistic}' is not supported."
            }
        
        # Create a visualization
        fig = go.Figure()
        
        if statistic in ["mean", "median"]:
            # Show distribution with the statistic
            fig = px.histogram(
                df, x=column,
                title=f"Distribution of {column} with {text_stat}",
                histnorm='probability density',
                marginal='box',
                color_discrete_sequence=['#0078D7']
            )
            
            fig.add_vline(x=result, line_dash="dash", line_color="red",
                          annotation_text=f"{text_stat.capitalize()}: {result:.4f}")
            
        else:
            # Create a simple metric visualization
            fig.add_trace(go.Indicator(
                mode = "number",
                value = result,
                title = {"text": f"{text_stat.capitalize()} of {column}"},
                domain = {'x': [0, 1], 'y': [0, 1]}
            ))
        
        explanation = f"The {text_stat} of '{column}' is {result:.4f}"
        
        return {
            "interpretation": f"Computing the {text_stat} of {column}",
            "data_result": result,
            "visualization": fig,
            "explanation": explanation
        }
    
    def _compute_correlation(self, df, operation, query):
        """Compute correlation between columns"""
        params = operation["params"]
        columns = params.get("columns", [])
        
        if not columns:
            return {
                "interpretation": "Couldn't compute correlation, no columns specified.",
                "data_result": None,
                "visualization": None,
                "explanation": "Please specify which columns to analyze for correlation."
            }
        
        # Check if all columns exist and are numeric
        missing_cols = [col for col in columns if col not in df.columns]
        non_numeric_cols = [col for col in columns if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
        
        if missing_cols:
            return {
                "interpretation": f"Columns not found: {', '.join(missing_cols)}",
                "data_result": None,
                "visualization": None,
                "explanation": f"The following columns were not found in the dataset: {', '.join(missing_cols)}"
            }
        
        if non_numeric_cols:
            return {
                "interpretation": f"Non-numeric columns: {', '.join(non_numeric_cols)}",
                "data_result": None,
                "visualization": None,
                "explanation": f"Correlation can only be computed for numeric columns. The following columns are not numeric: {', '.join(non_numeric_cols)}"
            }
        
        # If we have exactly two columns, create a scatter plot to show relationship
        if len(columns) == 2:
            # Compute correlation
            corr = df[columns].corr().iloc[0, 1]
            
            # Create scatter plot
            fig = px.scatter(
                df, x=columns[0], y=columns[1],
                trendline="ols",
                title=f"Correlation between {columns[0]} and {columns[1]}",
                color_discrete_sequence=['#0078D7']
            )
            
            fig.update_layout(
                annotations=[
                    dict(
                        x=0.5,
                        y=1.05,
                        xref="paper",
                        yref="paper",
                        text=f"Correlation coefficient: {corr:.4f}",
                        showarrow=False,
                        font=dict(size=14)
                    )
                ]
            )
            
            # Interpret correlation strength
            strength = ""
            if abs(corr) < 0.3:
                strength = "weak"
            elif abs(corr) < 0.7:
                strength = "moderate"
            else:
                strength = "strong"
            
            direction = "positive" if corr >= 0 else "negative"
            
            explanation = f"There is a {strength} {direction} correlation ({corr:.4f}) between {columns[0]} and {columns[1]}.\n\n"
            
            if corr > 0:
                explanation += f"This means that as {columns[0]} increases, {columns[1]} tends to increase as well."
            else:
                explanation += f"This means that as {columns[0]} increases, {columns[1]} tends to decrease."
            
            return {
                "interpretation": f"Analyzing correlation between {columns[0]} and {columns[1]}",
                "data_result": corr,
                "visualization": fig,
                "explanation": explanation
            }
        
        # If we have more than two columns, create a correlation matrix
        else:
            # Compute correlation matrix
            corr_matrix = df[columns].corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title="Correlation Matrix",
                zmin=-1, zmax=1
            )
            
            # Find top correlations
            corr_pairs = []
            for i in range(len(columns)):
                for j in range(i+1, len(columns)):
                    corr_pairs.append((columns[i], columns[j], corr_matrix.iloc[i, j]))
            
            # Sort by absolute correlation
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Create explanation
            explanation = "Top correlations:\n\n"
            for col1, col2, corr in corr_pairs[:5]:  # Show top 5
                strength = ""
                if abs(corr) < 0.3:
                    strength = "weak"
                elif abs(corr) < 0.7:
                    strength = "moderate"
                else:
                    strength = "strong"
                
                direction = "positive" if corr >= 0 else "negative"
                
                explanation += f"- {col1} and {col2}: {strength} {direction} correlation ({corr:.4f})\n"
            
            return {
                "interpretation": "Analyzing correlations between multiple columns",
                "data_result": corr_matrix,
                "visualization": fig,
                "explanation": explanation
            }
    
    def _show_distribution(self, df, operation, query):
        """Show distribution of a column"""
        params = operation["params"]
        column = params.get("column", None)
        
        if not column or column not in df.columns:
            return {
                "interpretation": "Couldn't analyze distribution, column not found.",
                "data_result": None,
                "visualization": None,
                "explanation": f"The column '{column}' was not found in the dataset."
            }
        
        # Check if the column is numeric or categorical
        if pd.api.types.is_numeric_dtype(df[column]):
            # Numeric column
            # Compute basic statistics
            stats = df[column].describe()
            
            # Create histogram
            fig = px.histogram(
                df, x=column,
                marginal="box",
                title=f"Distribution of {column}",
                color_discrete_sequence=['#0078D7'],
                histnorm='probability density'
            )
            
            # Add mean and median lines
            fig.add_vline(x=stats['mean'], line_dash="solid", line_color="red",
                         annotation_text=f"Mean: {stats['mean']:.4f}")
            fig.add_vline(x=stats['50%'], line_dash="dash", line_color="green",
                         annotation_text=f"Median: {stats['50%']:.4f}")
            
            # Create explanation
            explanation = f"""
            Distribution of {column}:
            - Mean: {stats['mean']:.4f}
            - Median: {stats['50%']:.4f}
            - Standard Deviation: {stats['std']:.4f}
            - Min: {stats['min']:.4f}
            - Max: {stats['max']:.4f}
            - 25th percentile: {stats['25%']:.4f}
            - 75th percentile: {stats['75%']:.4f}
            """
            
            # Add skewness analysis
            skew = df[column].skew()
            if abs(skew) < 0.5:
                explanation += "\nThe distribution is approximately symmetric."
            elif skew > 0:
                explanation += "\nThe distribution is positively skewed (right-tailed)."
            else:
                explanation += "\nThe distribution is negatively skewed (left-tailed)."
                
        else:
            # Categorical column
            # Get value counts
            value_counts = df[column].value_counts().reset_index()
            value_counts.columns = [column, 'count']
            value_counts['percentage'] = (value_counts['count'] / value_counts['count'].sum() * 100).round(2)
            
            # Create bar chart
            fig = px.bar(
                value_counts, x=column, y='count',
                title=f"Distribution of {column}",
                color=column,
                labels={'count': 'Count'},
                text='percentage'
            )
            
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            
            # Create explanation
            explanation = f"Distribution of {column}:\n\n"
            
            for i, row in value_counts.head(10).iterrows():
                explanation += f"- {row[column]}: {row['count']} ({row['percentage']:.2f}%)\n"
                
            if len(value_counts) > 10:
                explanation += f"\n(Showing top 10 out of {len(value_counts)} categories)"
        
        return {
            "interpretation": f"Analyzing the distribution of {column}",
            "data_result": None,
            "visualization": fig,
            "explanation": explanation
        }
    
    def _show_time_series(self, df, operation, query):
        """Show time series analysis"""
        params = operation["params"]
        date_column = params.get("date_column", None)
        value_column = params.get("value_column", None)
        
        # Check if columns exist
        if not date_column or date_column not in df.columns:
            return {
                "interpretation": "Couldn't perform time series analysis, date column not found.",
                "data_result": None,
                "visualization": None,
                "explanation": f"The date column '{date_column}' was not found in the dataset."
            }
            
        if not value_column or value_column not in df.columns:
            return {
                "interpretation": "Couldn't perform time series analysis, value column not found.",
                "data_result": None,
                "visualization": None,
                "explanation": f"The value column '{value_column}' was not found in the dataset."
            }
        
        # If the date column is not datetime, try to convert it
        if not pd.api.types.is_datetime64_dtype(df[date_column]):
            try:
                date_series = pd.to_datetime(df[date_column])
                # Create a copy to avoid warning
                df_copy = df.copy()
                df_copy[date_column] = date_series
                df = df_copy
            except:
                return {
                    "interpretation": "Couldn't convert column to datetime format.",
                    "data_result": None,
                    "visualization": None,
                    "explanation": f"The column '{date_column}' could not be converted to a datetime format."
                }
        
        # Check if value column is numeric
        if not pd.api.types.is_numeric_dtype(df[value_column]):
            return {
                "interpretation": "Value column must be numeric for time series analysis.",
                "data_result": None,
                "visualization": None,
                "explanation": f"The value column '{value_column}' is not numeric."
            }
        
        # Sort by date
        df_sorted = df.sort_values(by=date_column)
        
        # Create a line chart
        fig = px.line(
            df_sorted, x=date_column, y=value_column,
            title=f"{value_column} over time",
            labels={date_column: "Date", value_column: value_column},
            markers=True,
            color_discrete_sequence=['#0078D7']
        )
        
        # Add trend line
        try:
            fig_trend = px.scatter(
                df_sorted, x=date_column, y=value_column,
                trendline="ols"
            )
            trend_line = fig_trend.data[1]
            fig.add_trace(trend_line)
        except:
            pass  # Skip trend line if it fails
        
        # Analyze trend
        earliest_date = df_sorted[date_column].min()
        latest_date = df_sorted[date_column].max()
        start_value = df_sorted[value_column].iloc[0]
        end_value = df_sorted[value_column].iloc[-1]
        
        # Calculate change
        absolute_change = end_value - start_value
        percent_change = (absolute_change / start_value * 100) if start_value != 0 else float('inf')
        
        # Determine trend direction
        if percent_change > 5:
            trend = "increasing"
        elif percent_change < -5:
            trend = "decreasing"
        else:
            trend = "stable"
        
        # Create explanation
        explanation = f"""
        Time Series Analysis of {value_column} from {earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}:
        
        - Starting value: {start_value:.4f}
        - Ending value: {end_value:.4f}
        - Absolute change: {absolute_change:.4f}
        - Percent change: {percent_change:.2f}%
        
        The overall trend is {trend}.
        """
        
        # Check for seasonality (very basic check)
        if len(df_sorted) >= 12:
            explanation += "\n\nThis dataset may contain enough time points to analyze for seasonality or cyclical patterns."
        
        return {
            "interpretation": f"Analyzing trends in {value_column} over time",
            "data_result": df_sorted[[date_column, value_column]].head(10),
            "visualization": fig,
            "explanation": explanation
        }
    
    def _group_data(self, df, operation, query):
        """Group data by a column and apply an aggregation function"""
        params = operation["params"]
        group_column = params.get("group_column", None)
        value_column = params.get("value_column", None)
        agg_func = params.get("agg_func", "mean")
        
        # Check if columns exist
        if not group_column or group_column not in df.columns:
            return {
                "interpretation": "Couldn't perform grouping, group column not found.",
                "data_result": None,
                "visualization": None,
                "explanation": f"The group column '{group_column}' was not found in the dataset."
            }
            
        if not value_column or value_column not in df.columns:
            return {
                "interpretation": "Couldn't perform grouping, value column not found.",
                "data_result": None,
                "visualization": None,
                "explanation": f"The value column '{value_column}' was not found in the dataset."
            }
        
        # Check if value column is numeric
        if not pd.api.types.is_numeric_dtype(df[value_column]):
            return {
                "interpretation": "Value column must be numeric for aggregation.",
                "data_result": None,
                "visualization": None,
                "explanation": f"The value column '{value_column}' is not numeric."
            }
        
        # Perform grouping and aggregation
        grouped = df.groupby(group_column)[value_column].agg(agg_func).reset_index()
        grouped.columns = [group_column, f"{agg_func}_{value_column}"]
        
        # Sort by aggregated value
        grouped = grouped.sort_values(by=f"{agg_func}_{value_column}", ascending=False)
        
        # Create a bar chart
        fig = px.bar(
            grouped, x=group_column, y=f"{agg_func}_{value_column}",
            title=f"{agg_func.capitalize()} of {value_column} by {group_column}",
            color=group_column,
            labels={f"{agg_func}_{value_column}": f"{agg_func.capitalize()} of {value_column}"},
            text=f"{agg_func}_{value_column}"
        )
        
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        
        # Create explanation
        if agg_func == "mean":
            func_desc = "average"
        elif agg_func == "sum":
            func_desc = "sum"
        else:
            func_desc = agg_func
            
        explanation = f"Calculating the {func_desc} of {value_column} grouped by {group_column}:\n\n"
        
        for i, row in grouped.head(10).iterrows():
            explanation += f"- {row[group_column]}: {row[f'{agg_func}_{value_column}']:.4f}\n"
            
        if len(grouped) > 10:
            explanation += f"\n(Showing top 10 out of {len(grouped)} groups)"
        
        # Add insights
        if len(grouped) >= 2:
            top_group = grouped.iloc[0][group_column]
            top_value = grouped.iloc[0][f"{agg_func}_{value_column}"]
            bottom_group = grouped.iloc[-1][group_column]
            bottom_value = grouped.iloc[-1][f"{agg_func}_{value_column}"]
            
            # Calculate the difference between highest and lowest
            diff = top_value - bottom_value
            pct_diff = (diff / bottom_value * 100) if bottom_value != 0 else float('inf')
            
            explanation += f"\n\nInsights:\n"
            explanation += f"- The highest {func_desc} of {value_column} is in the '{top_group}' group ({top_value:.4f}).\n"
            explanation += f"- The lowest {func_desc} of {value_column} is in the '{bottom_group}' group ({bottom_value:.4f}).\n"
            
            if abs(pct_diff) != float('inf'):
                explanation += f"- The difference between highest and lowest is {diff:.4f} ({pct_diff:.2f}%).\n"
        
        return {
            "interpretation": f"Calculating {func_desc} of {value_column} grouped by {group_column}",
            "data_result": grouped,
            "visualization": fig,
            "explanation": explanation
        }
    
    def _compare_data(self, df, operation, query):
        """Compare a metric across different categories"""
        params = operation["params"]
        category_column = params.get("category_column", None)
        value_column = params.get("value_column", None)
        
        # Check if columns exist
        if not category_column or category_column not in df.columns:
            return {
                "interpretation": "Couldn't perform comparison, category column not found.",
                "data_result": None,
                "visualization": None,
                "explanation": f"The category column '{category_column}' was not found in the dataset."
            }
            
        if not value_column or value_column not in df.columns:
            return {
                "interpretation": "Couldn't perform comparison, value column not found.",
                "data_result": None,
                "visualization": None,
                "explanation": f"The value column '{value_column}' was not found in the dataset."
            }
        
        # Check if value column is numeric
        if not pd.api.types.is_numeric_dtype(df[value_column]):
            return {
                "interpretation": "Value column must be numeric for comparison.",
                "data_result": None,
                "visualization": None,
                "explanation": f"The value column '{value_column}' is not numeric."
            }
        
        # Calculate summary statistics by category
        stats = df.groupby(category_column)[value_column].agg(['mean', 'median', 'std', 'count']).reset_index()
        
        # Create a visualization
        if df[category_column].nunique() <= 10:
            # Use box plot for detailed comparison
            fig = px.box(
                df, x=category_column, y=value_column,
                title=f"Comparison of {value_column} across {category_column} categories",
                color=category_column,
                notched=True
            )
            
            # Add mean markers
            for i, row in stats.iterrows():
                fig.add_annotation(
                    x=row[category_column], y=row['mean'],
                    text="Mean", showarrow=True,
                    arrowhead=1, arrowcolor="red", arrowsize=1, arrowwidth=2,
                    ax=0, ay=-40
                )
        else:
            # Use bar chart with error bars for many categories
            fig = px.bar(
                stats, x=category_column, y='mean',
                title=f"Comparison of average {value_column} across {category_column} categories",
                color=category_column,
                error_y=stats['std'],
                labels={'mean': f'Mean {value_column}'},
                text='mean'
            )
            
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        
        # Create explanation
        explanation = f"Comparison of {value_column} across {category_column} categories:\n\n"
        
        for i, row in stats.iterrows():
            explanation += f"- {row[category_column]}:\n"
            explanation += f"  - Mean: {row['mean']:.4f}\n"
            explanation += f"  - Median: {row['median']:.4f}\n"
            explanation += f"  - Standard Deviation: {row['std']:.4f}\n"
            explanation += f"  - Count: {row['count']}\n\n"
        
        # Add ANOVA results for statistical significance if there are enough categories
        if df[category_column].nunique() > 1:
            try:
                from scipy import stats as spstats
                
                # Prepare data for ANOVA
                groups = [df[df[category_column] == cat][value_column].dropna().values 
                         for cat in df[category_column].unique()]
                
                # Run ANOVA
                f_val, p_val = spstats.f_oneway(*groups)
                
                explanation += "\nStatistical Analysis:\n"
                explanation += f"- F-value: {f_val:.4f}\n"
                explanation += f"- p-value: {p_val:.4f}\n"
                
                # Interpret results
                if p_val < 0.05:
                    explanation += "\nThere is a statistically significant difference in the means across categories."
                else:
                    explanation += "\nThere is no statistically significant difference in the means across categories."
            except:
                pass  # Skip ANOVA if it fails
        
        return {
            "interpretation": f"Comparing {value_column} across different {category_column} categories",
            "data_result": stats,
            "visualization": fig,
            "explanation": explanation
        }