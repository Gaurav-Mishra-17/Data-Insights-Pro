import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dotenv import load_dotenv
from pathlib import Path

# Load configuration only from the project root .env.
# A package-local fallback would make it easy to accidentally read or commit secrets.
ROOT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ROOT_ENV_PATH)

# Import Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Import OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None



# Initialize GeminiAI API client
Gemini_API_KEY = os.environ.get("Gemini_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if genai is not None and Gemini_API_KEY:
    genai.configure(api_key=Gemini_API_KEY)

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OpenAI is not None and OPENAI_API_KEY else None


def _get_provider() -> Optional[str]:
    """Select the AI provider based on available keys/packages."""
    if openai_client is not None:
        return "openai"
    if genai is not None and Gemini_API_KEY:
        return "gemini"
    return None


def _extract_response_text(raw_text: str) -> str:
    """Normalize model output to plain JSON text."""
    response_text = raw_text.strip()
    if not response_text:
        return ""

    if response_text.startswith("```"):
        response_text = response_text.lstrip("`")
        response_text = "\n".join(response_text.split("\n")[1:])
        if response_text.endswith("```"):
            response_text = response_text[:-3].strip()

    return response_text


def _generate_content(prompt: str) -> str:
    """Generate text from the configured provider."""
    provider = _get_provider()
    if provider is None:
        raise RuntimeError(
            "No AI provider configured. Add OPENAI_API_KEY or Gemini_API_KEY and install provider package(s)."
        )

    if provider == "openai":
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        response = openai_client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "system",
                    "content": "Return strictly valid JSON only. No markdown code fences.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.output_text or ""

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return getattr(response, "text", "") or ""

def check_api_key() -> bool:
    """
    Check if at least one AI API key is available
    
    Returns:
        bool: True if API key is available, False otherwise
    """
    return _get_provider() is not None

def generate_insights(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Generate AI-powered insights about the dataset
    
    Args:
        df (pd.DataFrame): The dataframe to analyze
        columns (List[str], optional): Specific columns to focus on
        
    Returns:
        Dict[str, Any]: Dictionary with insights
    """
    if not check_api_key():
        return {
            "error": "No AI API key is configured. Add OPENAI_API_KEY or Gemini_API_KEY to use AI-powered insights."
        }
    
    try:
        # Prepare data summary for the model
        df_sample = df.head(20).to_string()
        data_info = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_types": {col: str(df[col].dtype) for col in df.columns},
            "missing_values": df.isna().sum().to_dict(),
            "sample_data": df_sample
        }
        
        # Create a prompt for analysis
        prompt = f"""
        You are a data analysis expert. I have a dataset with the following characteristics:
        
        Rows: {data_info['rows']}
        Columns: {data_info['columns']}
        Column types: {data_info['column_types']}
        
        Here's a sample of the data:
        {df_sample}
        
        Please provide 5-7 meaningful insights about this data. Focus on:
        1. Key patterns or trends
        2. Potential issues or anomalies
        3. Interesting relationships between variables
        4. Suggestions for further analysis
        
        Format your response as JSON with the following structure:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation",
                    "type": "pattern|anomaly|relationship|suggestion",
                    "confidence": 0.0 to 1.0
                }}
            ]
        }}
        """
        
        response_text = _extract_response_text(_generate_content(prompt))

        if not response_text:
            return {"error": "AI provider returned an empty response."}

        try:
            result = json.loads(response_text)
        except Exception as e:
            return {
                "error": f"AI provider did not return valid JSON. Raw response: {response_text}"
            }
        return result
        # result = json.loads(response.text)
        # return result
    
    except Exception as e:
        return {
            "error": f"Error generating insights: {str(e)}"
        }

def analyze_patterns(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Analyze patterns in a specific column
    
    Args:
        df (pd.DataFrame): The dataframe to analyze
        column (str): The column to analyze
        
    Returns:
        Dict[str, Any]: Dictionary with pattern analysis
    """
    if not check_api_key():
        return {
            "error": "No AI API key is configured. Add OPENAI_API_KEY or Gemini_API_KEY to use AI-powered insights."
        }
    
    try:
        # Get column data
        col_data = df[column].dropna()
        col_type = str(df[column].dtype)
        col_sample = col_data.head(20).to_list()
        
        # Statistics for numeric data
        if pd.api.types.is_numeric_dtype(df[column]):
            stats = {
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "skew": float(col_data.skew()),
                "kurtosis": float(col_data.kurtosis())
            }
        else:
            stats = {
                "unique_values": col_data.nunique(),
                "top_values": col_data.value_counts().head(5).to_dict()
            }
        
        # Create a prompt for analysis
        prompt = f"""
        You are a data pattern recognition expert. I want you to analyze the following column in my dataset:
        
        Column name: {column}
        Data type: {col_type}
        
        Statistics: {stats}
        
        Sample values: {col_sample}
        
        Please analyze patterns in this data and provide insights on:
        1. Distribution characteristics
        2. Potential patterns or cycles
        3. Unusual features or outliers
        4. Suggestions for transformations or feature engineering
        
        Format your response as JSON with the following structure:
        {{
            "patterns": [
                {{
                    "title": "Pattern title",
                    "description": "Detailed explanation",
                    "confidence": 0.0 to 1.0,
                    "suggestion": "Optional suggestion for leveraging this pattern"
                }}
            ]
        }}
        """

        response_text = _extract_response_text(_generate_content(prompt))

        if not response_text:
            return {"error": "AI provider returned an empty response."}

        try:
            result = json.loads(response_text)
        except Exception as e:
            return {
                "error": f"AI provider did not return valid JSON. Raw response: {response_text}"
            }
        return result
        # result = json.loads(response.text)
        # return result
    
    except Exception as e:
        return {
            "error": f"Error analyzing patterns: {str(e)}"
        }

def explain_anomalies(df: pd.DataFrame, anomalies_df: pd.DataFrame, method: str, column: str) -> Dict[str, Any]:
    """
    Provide explanations for detected anomalies
    
    Args:
        df (pd.DataFrame): The original dataframe
        anomalies_df (pd.DataFrame): Dataframe with anomalies
        method (str): The method used for anomaly detection
        column (str): The column where anomalies were detected
        
    Returns:
        Dict[str, Any]: Dictionary with anomaly explanations
    """
    if not check_api_key():
        return {
            "error": "No AI API key is configured. Add OPENAI_API_KEY or Gemini_API_KEY to use AI-powered insights."
        }
    
    try:
        # Get context for anomalies
        num_anomalies = len(anomalies_df)
        anomaly_pct = (num_anomalies / len(df)) * 100
        
        # Sample of anomalies
        anomaly_sample = anomalies_df.head(10).to_string()
        
        # Column statistics
        col_stats = df[column].describe().to_dict()
        
        # Create a prompt for explanation
        prompt = f"""
        You are a data anomaly expert. I have detected {num_anomalies} anomalies ({anomaly_pct:.2f}% of data) 
        in the column '{column}' using the {method} method.
        
        Column statistics:
        {col_stats}
        
        Here's a sample of the anomalies:
        {anomaly_sample}
        
        Please provide expert explanations for:
        1. Possible causes of these anomalies
        2. How these anomalies might impact analysis
        3. Recommended actions for handling these anomalies
        4. Whether these are likely true anomalies or data quality issues
        
        Format your response as JSON with the following structure:
        {{
            "explanations": [
                {{
                    "title": "Explanation title",
                    "description": "Detailed explanation",
                    "impact": "Potential impact on analysis",
                    "recommendation": "Recommended action",
                    "confidence": 0.0 to 1.0
                }}
            ]
        }}
        """

        response_text = _extract_response_text(_generate_content(prompt))

        if not response_text:
            return {"error": "AI provider returned an empty response."}

        try:
            result = json.loads(response_text)
        except Exception as e:
            return {
                "error": f"AI provider did not return valid JSON. Raw response: {response_text}"
            }
        return result
        # result = json.loads(response.text)
        # return result
    
    except Exception as e:
        return {
            "error": f"Error explaining anomalies: {str(e)}"
        }

def advanced_query_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    """
    Analyze a natural language query and provide insights
    
    Args:
        df (pd.DataFrame): The dataframe to analyze
        query (str): Natural language query
        
    Returns:
        Dict[str, Any]: Dictionary with query analysis and insights
    """
    if not check_api_key():
        return {
            "error": "No AI API key is configured. Add OPENAI_API_KEY or Gemini_API_KEY to use AI-powered insights."
        }
    
    try:
        # Prepare data context
        columns_info = {
            col: {
                "type": str(df[col].dtype),
                "unique_values": df[col].nunique(),
                "sample_values": df[col].dropna().sample(min(5, df[col].nunique())).tolist()
            }
            for col in df.columns
        }
        
        # Create a prompt for query analysis
        prompt = f"""
        You are a data analysis expert. I have a dataset with these columns:
        {json.dumps(columns_info, indent=2)}
        
        User query: "{query}"

        IMPORTANT: The DataFrame is already loaded as 'df'. Do NOT use pd.read_csv or load any files. Use the provided 'df' variable for all analysis and plotting.
        
        Please analyze this query and provide:
        1. Interpretation of what the user is asking
        2. Relevant columns that should be examined
        3. Suggested analytical approaches to answer this query
        4. Additional insights the user might find valuable
        5. If possible, you MUST provide a Python code snippet (using pandas and plotly) to perform the analysis or create the requested chart. If not possible, explain why in the "reason_not_possible" field.

        Format your response as JSON with the following structure:
        {{
            "interpretation": "Interpretation of the query",
            "relevant_columns": ["column1", "column2"],
            "analytical_approach": "Detailed approach",
            "additional_insights": "Additional context or suggestions",
            "python_code": "Python code as a string, if possible",
            "reason_not_possible": "Reason if code cannot be generated",
            "confidence": 0.0 to 1.0
        }}
        """

        response_text = _extract_response_text(_generate_content(prompt))

        if not response_text:
            return {"error": "AI provider returned an empty response."}

        try:
            result = json.loads(response_text)
        except Exception as e:
            return {
                "error": f"AI provider did not return valid JSON. Raw response: {response_text}"
            }
        return result
        # result = json.loads(response.text)
        # return result
    
    except Exception as e:
        return {
            "error": f"Error analyzing query: {str(e)}"
        }