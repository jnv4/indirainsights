import os
import io
import json
import math
import base64
import gzip
import tempfile
import re
from io import StringIO, BytesIO
from datetime import datetime
from html import escape
import streamlit as st
import pandas as pd
import numpy as np
import requests
import duckdb
import openpyxl
from dotenv import load_dotenv
from supabase import create_client, Client
import google.generativeai as genai
import pyodbc
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    PageBreak
)
from markdown import markdown
from bs4 import BeautifulSoup

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = "marketing-cleaned"
METADATA_TABLE = "dataset_registry"

AZURE_SERVER = os.getenv("server")
AZURE_DATABASE = os.getenv("database")
AZURE_USERNAME = "ChatAgent"
AZURE_PASSWORD = os.getenv("password")
AZURE_DRIVER = os.getenv("driver")

import pymssql

def get_azure_connection():
    return pymssql.connect(
        server=AZURE_SERVER.replace("tcp:", "").replace(",1433", ""),
        user=AZURE_USERNAME,
        password=AZURE_PASSWORD,
        database=AZURE_DATABASE,
        port=1433,
        login_timeout=30,
        timeout=30
    )

def get_all_azure_tables():
    """Get list of ALL available tables from Azure SQL database"""
    try:
        conn = get_azure_connection()
        cursor = conn.cursor()
        
        # Query to get all user tables
        cursor.execute("""
            SELECT TABLE_NAME 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
        """)
        
        all_tables = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        return all_tables
    except Exception as e:
        raise Exception(f"Failed to fetch tables: {str(e)}")

def load_selected_azure_tables(selected_tables: list):
    """Load only the selected Azure SQL tables into session state"""
    datasets = {}
    
    try:
        for table_name in selected_tables:
            try:
                # Load table (sample first 10000 rows for performance)
                df = load_azure_table_to_dataframe(table_name, sample_size=10000)
                
                # Clean and optimize DataFrame
                df_clean = clean_dataframe(df)
                
                datasets[table_name] = {
                    'dataframe': df_clean,
                    'field': 'Azure SQL',
                    'source': 'azure_sql'
                }
                
            except Exception as e:
                st.warning(f"Could not load table {table_name}: {str(e)}")
                continue
        
        return datasets
    
    except Exception as e:
        st.error(f"Failed to load Azure tables: {str(e)}")
        return {}

def load_azure_table_to_dataframe(table_name: str, sample_size: int = None):
    """Load Azure SQL table into pandas DataFrame"""
    try:
        conn = get_azure_connection()
        
        query = f"SELECT * FROM {table_name}"
        if sample_size:
            query = f"SELECT TOP {sample_size} * FROM {table_name}"
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        return df
    except Exception as e:
        raise Exception(f"Failed to load table {table_name}: {str(e)}")

def get_azure_table_schema(table_name: str):
    """Get schema information for a specific Azure SQL table"""
    try:
        conn = get_azure_connection()
        cursor = conn.cursor()
        
        # Get column information
        cursor.execute(f"""
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                CHARACTER_MAXIMUM_LENGTH,
                IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{table_name}'
            ORDER BY ORDINAL_POSITION
        """)
        
        columns = []
        for row in cursor.fetchall():
            columns.append({
                "name": row[0],
                "type": row[1],
                "max_length": row[2],
                "nullable": row[3] == 'YES'
            })
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return {
            "columns": columns,
            "row_count": row_count
        }
    except Exception as e:
        raise Exception(f"Failed to get schema for {table_name}: {str(e)}")

@st.cache_data(show_spinner=False, ttl=3600)
def load_all_azure_tables():
    """Load all accessible Azure SQL tables into session state"""
    datasets = {}
    
    try:
        tables = get_azure_tables()
        
        for table_name in tables:
            try:
                # Load table (sample first 10000 rows for performance)
                df = load_azure_table_to_dataframe(table_name, sample_size=10000)
                
                # Clean and optimize DataFrame
                df_clean = clean_dataframe(df)
                
                datasets[table_name] = {
                    'dataframe': df_clean,
                    'field': 'Azure SQL',  # You can categorize differently if needed
                    'source': 'azure_sql'
                }
                
            except Exception as e:
                st.warning(f"Could not load table {table_name}: {str(e)}")
                continue
        
        return datasets
    
    except Exception as e:
        st.error(f"Failed to load Azure tables: {str(e)}")
        return {}

@st.cache_data(show_spinner=False)
def get_schema_info_from_azure():
    """Get enhanced schema info from Azure SQL tables"""
    schema_info = {}
    
    if st.session_state.active_datasets:
        for name, data in st.session_state.active_datasets.items():
            try:
                df = data['dataframe']
                
                # Get enhanced schema
                enhanced_schema = extract_schema(df)
                
                schema_info[name] = {
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "sample_values": {col: df[col].dropna().unique()[:5].tolist() for col in df.columns},
                    "row_count": len(df),
                    "field": data.get('field', 'Azure SQL'),
                    "dataset_role": enhanced_schema.get("dataset_role", "unknown"),
                    "business_entity": enhanced_schema.get("business_entity", "Other"),
                    "primary_keys": enhanced_schema.get("primary_keys", []),
                    "foreign_key_candidates": enhanced_schema.get("foreign_key_candidates", []),
                    "time_columns": enhanced_schema.get("time_columns", [])
                }
            except Exception as e:
                st.error(f"Error loading schema for {name}: {e}")
    
    return schema_info

def initialize_duckdb_from_azure_datasets():
    """Initialize DuckDB connection and load all Azure SQL datasets as tables"""
    import duckdb
    
    conn = duckdb.connect(database=':memory:')
    
    if not st.session_state.get("active_datasets"):
        raise Exception("No datasets available")
    
    for dataset_name, data in st.session_state.active_datasets.items():
        df = data['dataframe'] if isinstance(data, dict) else data
        
        # Sanitize table name for DuckDB
        table_name = dataset_name.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
        table_name = ''.join(c for c in table_name if c.isalnum() or c == '_')
        
        # Register DataFrame as DuckDB table
        conn.register(table_name, df)
    
    return conn

@st.cache_data(show_spinner=False)
def load_csv(url: str) -> pd.DataFrame:
    r = requests.get(url)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

def extract_schema(df):
    """Extract enhanced schema with dataset intelligence"""
    
    # Basic schema extraction
    raw_schema = {
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "row_count": int(len(df)),
        "sample_rows": df.head(5).to_dict(orient="records")
    }
    
    # Enhanced intelligence: infer dataset role
    columns_lower = [c.lower() for c in df.columns]
    
    # Detect fact tables (transaction/event data with metrics)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    has_metrics = len(numeric_cols) >= 2
    has_id_pattern = any('id' in c for c in columns_lower)
    has_timestamp = any(kw in ' '.join(columns_lower) for kw in ['date', 'time', 'created', 'updated', 'timestamp'])
    
    if has_metrics and (has_id_pattern or has_timestamp):
        dataset_role = "fact"
    elif len(df) < 500 and not has_metrics:
        dataset_role = "dimension"
    elif any(kw in ' '.join(columns_lower) for kw in ['lookup', 'reference', 'category', 'type']):
        dataset_role = "lookup"
    else:
        dataset_role = "fact"
    
    # Infer primary keys
    primary_keys = []
    for col in df.columns:
        col_lower = col.lower()
        # Check for ID patterns and uniqueness
        if 'id' in col_lower or col_lower.endswith('_key'):
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.95:
                primary_keys.append(col)
        # Check for natural keys
        elif col_lower in ['email', 'phone', 'phone_number', 'mobile']:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:
                primary_keys.append(col)
    
    # Infer time columns
    time_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['date', 'time', 'created', 'updated', 'timestamp', 'day', 'month', 'year']):
            time_columns.append(col)
        # Check dtype
        elif df[col].dtype == 'datetime64[ns]' or 'datetime' in str(df[col].dtype):
            time_columns.append(col)
    
    # Infer business entity
    text_content = ' '.join(columns_lower)
    
    if any(kw in text_content for kw in ['lead', 'prospect', 'contact', 'converted']):
        business_entity = "Lead"
    elif any(kw in text_content for kw in ['campaign', 'ad', 'advertisement', 'marketing']):
        business_entity = "Campaign"
    elif any(kw in text_content for kw in ['call', 'phone', 'duration', 'agent']):
        business_entity = "Call"
    elif any(kw in text_content for kw in ['competitor', 'competition', 'rival']):
        business_entity = "Competitor"
    elif any(kw in text_content for kw in ['region', 'location', 'city', 'state', 'country', 'geography']):
        business_entity = "Geography"
    elif any(kw in text_content for kw in ['website', 'web', 'session', 'visit', 'page', 'bounce']):
        business_entity = "WebAnalytics"
    else:
        business_entity = "Other"
    
    # Infer foreign key candidates (columns that could join to other tables)
    foreign_key_candidates = []
    for col in df.columns:
        col_lower = col.lower()
        # ID patterns that might reference other tables
        if ('_id' in col_lower or 'id_' in col_lower) and col not in primary_keys:
            foreign_key_candidates.append(col)
        # Common join columns
        elif col_lower in ['region', 'city', 'state', 'country', 'campaign', 'source', 'category', 'type', 'phone', 'phone_number', 'email']:
            foreign_key_candidates.append(col)
    
    raw_schema["dataset_role"] = dataset_role
    raw_schema["primary_keys"] = primary_keys
    raw_schema["foreign_key_candidates"] = foreign_key_candidates
    raw_schema["time_columns"] = time_columns
    raw_schema["business_entity"] = business_entity
    
    return make_json_safe(raw_schema)

@st.cache_data(show_spinner=False)
def get_schema_info():
    """Get enhanced schema info with dataset intelligence"""
    schema_info = {}
    
    # Use active datasets if available
    if st.session_state.active_datasets:
        for name, data in st.session_state.active_datasets.items():
            try:
                df = data['dataframe']
                
                # Get enhanced schema
                enhanced_schema = extract_schema(df)
                
                schema_info[name] = {
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "sample_values": {col: df[col].dropna().unique()[:5].tolist() for col in df.columns},
                    "row_count": len(df),
                    "field": data.get('field', 'Unknown'),
                    "dataset_role": enhanced_schema.get("dataset_role", "unknown"),
                    "business_entity": enhanced_schema.get("business_entity", "Other"),
                    "primary_keys": enhanced_schema.get("primary_keys", []),
                    "foreign_key_candidates": enhanced_schema.get("foreign_key_candidates", []),
                    "time_columns": enhanced_schema.get("time_columns", [])
                }
            except Exception as e:
                st.error(f"Error loading schema for {name}: {e}")
    
    return schema_info

def has_datasets() -> bool:
    return bool(st.session_state.get("active_datasets"))

def read_uploaded_file(file):
    file_ext = file.name.split('.')[-1].lower()

    try:
        # ---------------- CSV ----------------
        if file_ext == 'csv':
            df = pd.read_csv(file)
            return {file.name.replace('.csv', ''): df}

        # ---------------- EXCEL (ROBUST) ----------------
        elif file_ext in ['xlsx', 'xls']:
            try:
                # Try modern Excel first
                excel_file = pd.ExcelFile(file, engine="openpyxl")
            except Exception:
                # Fallback to old Excel
                file.seek(0)
                excel_file = pd.ExcelFile(file, engine="xlrd")

            sheets = {}
            for sheet_name in excel_file.sheet_names:
                sheets[sheet_name] = excel_file.parse(sheet_name)

            return sheets

        # ---------------- JSON ----------------
        elif file_ext == 'json':
            content = file.read()
            data = json.loads(content)

            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError("Invalid JSON structure")

            return {file.name.replace('.json', ''): df}

        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    except Exception as e:
        raise Exception(f"Error reading file {file.name}: {str(e)}")

def make_json_safe(obj):
    """Recursively convert objects to JSON-safe types"""
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj

def clean_dataframe(df):
    """Clean and normalize DataFrame with aggressive size optimization"""
    df_clean = df.copy()
    
    # Normalize column names
    df_clean.columns = [
        col.lower().strip().replace(' ', '_').replace('-', '_').replace('.', '_')
        for col in df_clean.columns
    ]
    
    # Remove special characters from column names
    df_clean.columns = [''.join(c if c.isalnum() or c == '_' else '_' for c in col) 
                        for col in df_clean.columns]
    
    # Drop fully empty columns
    df_clean = df_clean.dropna(axis=1, how='all')
    
    # Remove duplicate rows
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_dupes = initial_rows - len(df_clean)
    if removed_dupes > 0:
        st.info(f"Removed {removed_dupes:,} duplicate rows")
    
    # Trim string values and handle whitespace
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        # Replace empty strings with NaN
        df_clean[col] = df_clean[col].replace(['', 'nan', 'NaN', 'None', 'NULL'], pd.NA)
    
    # Aggressive data type optimization
    for col in df_clean.select_dtypes(include=['float64']).columns:
        if df_clean[col].notna().any():
            # Check if can be converted to int
            if (df_clean[col].dropna() % 1 == 0).all():
                # Check range for int type
                col_min = df_clean[col].min()
                col_max = df_clean[col].max()
                
                if col_min >= -32768 and col_max <= 32767:
                    df_clean[col] = df_clean[col].astype('Int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df_clean[col] = df_clean[col].astype('Int32')
                else:
                    df_clean[col] = df_clean[col].astype('Int64')
            else:
                df_clean[col] = df_clean[col].astype('float32')
    
    for col in df_clean.select_dtypes(include=['int64']).columns:
        col_min = df_clean[col].min()
        col_max = df_clean[col].max()
        
        if col_min >= -128 and col_max <= 127:
            df_clean[col] = df_clean[col].astype('int8')
        elif col_min >= -32768 and col_max <= 32767:
            df_clean[col] = df_clean[col].astype('int16')
        elif col_min >= -2147483648 and col_max <= 2147483647:
            df_clean[col] = df_clean[col].astype('int32')
    
    # Convert object columns with low cardinality to category
    for col in df_clean.select_dtypes(include=['object']).columns:
        num_unique = df_clean[col].nunique()
        if num_unique / len(df_clean) < 0.5:  # Less than 50% unique values
            df_clean[col] = df_clean[col].astype('category')
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    # Report size
    size_mb = df_clean.memory_usage(deep=True).sum() / (1024 * 1024)
    
    return df_clean

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (st.session_state["username"] == "indiraivf" and 
            st.session_state["password"] == "indira@"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("## 🔐 Login to Marketing Insights Dashboard")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        return False
    elif not st.session_state["password_correct"]:
        st.markdown("## 🔐 Login to Marketing Insights Dashboard")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.error("😕 User not known or password incorrect")
        return False
    else:
        return True

def identify_relevant_files(user_query: str, api_key: str) -> list:
    try:
        genai.configure(api_key=api_key)
        schema_info = get_schema_info_from_azure()  # Changed function name
        schema_description = "Available datasets:\n\n"
        for name, info in schema_info.items():
            schema_description += f"### {name}\n"
            schema_description += f"Columns: {', '.join(info['columns'])}\n"
            schema_description += f"Row count: {info['row_count']}\n"
            schema_description += f"Sample values:\n"
            for col, vals in info['sample_values'].items():
                schema_description += f"  - {col}: {vals}\n"
            schema_description += "\n"
        
        intent_model = genai.GenerativeModel('gemini-2.5-flash')
        
        intent_prompt = f"""You are a senior data analyst responsible for selecting datasets required to answer a user’s question.

INPUTS:
- Available datasets and schema:
{schema_description}

- User question:
{user_query}

TASK:
Identify which datasets are required to answer the user’s question.

RULES (STRICT):
1. You MUST return at least one dataset name.
2. You must NEVER return an empty array.
3. Be inclusive rather than exclusive:
   - If a dataset might be even slightly relevant, INCLUDE it.
4. If the question is vague, broad, exploratory, or requires overall business understanding,
   RETURN ALL AVAILABLE DATASETS.
5. For analytical or comparative questions, include all datasets that could reasonably
   contribute to metrics, filters, joins, or context.
6. Use dataset names EXACTLY as they appear in the schema.
7. Do NOT explain your reasoning.
8. Do NOT add any extra text.

OUTPUT FORMAT:
Return ONLY a valid JSON array of dataset names.

EXAMPLES:
- Specific question → ["CRM Leads", "Footfall"]
- Revenue trend by region → ["Revenue", "Region"]
- Vague or exploratory question → ALL datasets from the schema

REMEMBER:
❌ Never return []
✅ When in doubt, include more datasets
"""
                                
        response = intent_model.generate_content(intent_prompt)
        
        response_text = response.text.strip()
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        relevant_files = json.loads(response_text)
        
        return relevant_files
        
    except Exception as e:
        st.warning(f"Error identifying relevant files: {str(e)}")

def top_value_metrics(series: pd.Series):
    vc = series.astype(str).value_counts()
    total = vc.sum()

    top_val = vc.index[0]
    top_share = round((vc.iloc[0] / total) * 100, 2)
    low_val = vc.index[-1]
    return top_val, top_share, low_val, total

def validate_api_key(api_key: str) -> bool:
    """Validate Gemini API key by attempting to configure it"""
    if not api_key or len(api_key.strip()) == 0:
        return False
    try:
        genai.configure(api_key=api_key.strip())
        list(genai.list_models())
        return True
    except Exception as e:
        st.error(f"Invalid API key: {str(e)}")
    return False

def get_dataset(dataset_name: str) -> pd.DataFrame:
    if not dataset_name:
        raise ValueError("No dataset selected")

    if dataset_name in st.session_state.get("active_datasets", {}):
        data = st.session_state.active_datasets[dataset_name]
        if isinstance(data, dict):
            return data['dataframe']
        return data  # Backward compatibility

    raise ValueError(f"Dataset {dataset_name} not found")

def initialize_duckdb_from_datasets():
    """Initialize DuckDB connection and load all active datasets as tables"""
    import duckdb
    
    conn = duckdb.connect(database=':memory:')
    
    if not st.session_state.get("active_datasets"):
        raise Exception("No datasets available")
    
    for dataset_name, data in st.session_state.active_datasets.items():
        df = data['dataframe'] if isinstance(data, dict) else data
        
        # Sanitize table name
        table_name = dataset_name.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
        table_name = ''.join(c for c in table_name if c.isalnum() or c == '_')
        
        # Register DataFrame as DuckDB table
        conn.register(table_name, df)
    
    return conn

def get_duckdb_tables(conn):
    """Get list of all tables in DuckDB connection"""
    result = conn.execute("SHOW TABLES").fetchall()
    return [row[0] for row in result]

def extract_duckdb_schema(conn):
    """Extract comprehensive schema information from DuckDB tables"""
    schema_info = {}
    
    tables = get_duckdb_tables(conn)
    
    for table in tables:
        try:
            # Get column info using PRAGMA
            columns_info = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
            
            columns = []
            for col_info in columns_info:
                col_name = col_info[1]
                col_type = col_info[2]
                
                # Get sample values and statistics for better planning
                sample_values = []
                is_numeric = col_type.lower() in ['integer', 'bigint', 'double', 'float', 'decimal', 'numeric']
                is_date = 'date' in col_type.lower() or 'timestamp' in col_type.lower()
                
                try:
                    # Get distinct count
                    distinct_count = conn.execute(
                        f"SELECT COUNT(DISTINCT {col_name}) FROM {table}"
                    ).fetchone()[0]
                    
                    # Get sample values
                    samples = conn.execute(
                        f"SELECT DISTINCT {col_name} FROM {table} WHERE {col_name} IS NOT NULL LIMIT 5"
                    ).fetchall()
                    sample_values = [str(s[0]) for s in samples]
                    
                    # Get stats for numeric columns
                    stats = {}
                    if is_numeric:
                        stats_query = f"""
                            SELECT 
                                MIN({col_name}) as min_val,
                                MAX({col_name}) as max_val,
                                AVG({col_name}) as avg_val
                            FROM {table}
                        """
                        stats_result = conn.execute(stats_query).fetchone()
                        stats = {
                            'min': stats_result[0],
                            'max': stats_result[1],
                            'avg': stats_result[2]
                        }
                    
                    columns.append({
                        "name": col_name,
                        "type": col_type,
                        "distinct_count": distinct_count,
                        "sample_values": sample_values,
                        "is_numeric": is_numeric,
                        "is_date": is_date,
                        "stats": stats
                    })
                except:
                    # Fallback if stats collection fails
                    columns.append({
                        "name": col_name,
                        "type": col_type,
                        "distinct_count": None,
                        "sample_values": [],
                        "is_numeric": is_numeric,
                        "is_date": is_date,
                        "stats": {}
                    })
            
            # Get row count
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            
            schema_info[table] = {
                "columns": columns,
                "row_count": row_count
            }
            
        except Exception as e:
            st.warning(f"Could not extract schema for table {table}: {str(e)}")
            continue
    
    return schema_info

def create_multi_query_plan(question: str, schema_info: dict, api_key: str) -> dict:
    """Create multi-query execution plan with dependencies"""
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    
    # Build schema description
    schema_description = "AVAILABLE DATASETS:\n\n"
    for table_name, info in schema_info.items():
        schema_description += f"TABLE: {table_name}\n"
        schema_description += f"Rows: {info['row_count']}\n"
        schema_description += "COLUMNS:\n"
        for col in info['columns']:
            schema_description += f"  • {col['name']} ({col['type']})"
            if col.get('distinct_count'):
                schema_description += f" - {col['distinct_count']} distinct values"
            if col.get('sample_values'):
                schema_description += f" - Examples: {', '.join(str(v) for v in col['sample_values'][:3])}"
            schema_description += "\n"
        schema_description += "\n"
    
    planning_prompt = f"""You are a query planner. Analyze the user's question and create a multi-query execution plan.

    SCHEMA:
    {schema_description}

    USER QUESTION: {question}

    RULES:
    - Determine if the question requires 1 or multiple queries
    - Each query must serve a specific purpose
    - Queries can depend on others (specify dependencies)
    - Use ONLY tables and columns that exist in the schema
    - Do NOT invent columns, tables, or data

    Output MUST be valid JSON with this EXACT structure:
    {{
    "queries": [
        {{
        "id": "q1",
        "purpose": "describe what this query computes",
        "depends_on": [],
        "plan": {{
            "query_type": "aggregation|filter|join|simple_select|ranking",
            "primary_table": "table_name",
            "secondary_tables": [],
            "required_columns": [
            {{"table": "table_name", "column": "col_name", "purpose": "grouping|filtering|aggregation"}}
            ],
            "filters": [
            {{"column": "col_name", "operator": "=|>|<|LIKE|IN", "value": "value"}}
            ],
            "aggregations": [
            {{"function": "COUNT|SUM|AVG|MIN|MAX", "column": "col_name", "alias": "result_name"}}
            ],
            "grouping": ["col1"],
            "ordering": [
            {{"column": "col_name", "direction": "ASC|DESC"}}
            ],
            "limit": 50,
            "joins": [
            {{"left_table": "table1", "right_table": "table2", "left_key": "col1", "right_key": "col2", "type": "INNER|LEFT"}}
            ]
        }}
        }},
        "note" : ["any warning and note to avoid syntax errors for DuckDB queries]
    ]
    }}

    OUTPUT (valid JSON only, no markdown):"""
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    response = model.generate_content(
        planning_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            response_mime_type="application/json"
        )
    )
    plan_text = response.text.strip()
    if "```json" in plan_text:
        plan_text = plan_text.split("```json")[1].split("```")[0].strip()
    elif "```" in plan_text:
        plan_text = plan_text.split("```")[1].split("```")[0].strip()
    
    import json
    multi_query_plan = json.loads(plan_text)
    
    return multi_query_plan

def generate_strict_sql_from_plan(single_query_plan: dict, schema_info: dict, api_key: str) -> str:
    """Generate SQL that EXACTLY follows the single query plan"""
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    
    # Build focused schema
    relevant_tables = [single_query_plan['primary_table']] + single_query_plan.get('secondary_tables', [])
    
    focused_schema = "SCHEMA:\n\n"
    for table_name in relevant_tables:
        if table_name in schema_info:
            info = schema_info[table_name]
            focused_schema += f"Table: {table_name}\n"
            focused_schema += "Columns:\n"
            for col in info['columns']:
                focused_schema += f"  - {col['name']} {col['type']}\n"
            focused_schema += "\n"
    
    import json
    plan_str = json.dumps(single_query_plan, indent=2)
    
    sql_prompt = f"""Generate ONLY valid DUCKDB SQL query that EXACTLY implements this plan.
    **Be extra cautious of duckdb syntax and do not cause errors**
    {focused_schema}

    EXECUTION PLAN:
    {plan_str}
    Create as simple and straight-forward query as you can to avoid unexpected errors.
    Limit 50 for every query.
    STRICT RULES:
    1. Use ONLY the tables, columns, filters, aggregations, joins, grouping, and ordering specified in the plan
    2. Do NOT add any joins not in the plan
    3. Do NOT add any filters not in the plan
    4. Do NOT add any aggregations not in the plan
    5. Use exact column names from the schema
    6. Output ONLY valid DuckDB SQL - no markdown, no comments, no explanations

    Your top priority is **robust execution without runtime errors**.
    You are a **DuckDB SQL expert** generating queries on user data.

    **STRICT RULES (must always follow):**
    2. If a column may contain non-date strings (e.g. `""`, `"NA"`, `"null"`, empty strings), you MUST:
    1. **Never directly CAST or STRPTIME a column** unless it is guaranteed to be clean.
        * Use `TRY_STRPTIME()` or `TRY_CAST()`
        * OR clean values first using `CASE WHEN` / `NULLIF`
    3. **Always assume date/time columns are strings** unless schema explicitly says otherwise.
    4. Before parsing dates, **filter or normalize invalid values**:
        * `NULLIF(column, '')`
        * `NULLIF(column, '')`
        * `LOWER(column) NOT IN ('unassigned','na','null')`
    5. **Never allow a query to fail due to parsing errors.**
        * Invalid values must become `NULL`, not exceptions.
    6. If date parsing is required, use this pattern:
    TRY_STRPTIME(
    NULLIF(column_name, ''),
    '%m/%d/%Y %I:%M%p'
    )
    7. For numeric fields that may contain text:
    TRY_CAST(NULLIF(column_name, '') AS DOUBLE)
    8. If a column is used in `WHERE`, `GROUP BY`, or `ORDER BY`, ensure the cleaned/parsed version is used consistently.
    9. If multiple date formats are possible, prefer:
    COALESCE(
    TRY_STRPTIME(col, format1),
    TRY_STRPTIME(col, format2)
    )
    10. If safe parsing is not possible, **avoid using the column** and still return a valid query.

    **GOAL:**
    Generate DuckDB SQL that **never crashes**, even with dirty, mixed-type, or unexpected values.

    **Output ONLY valid DuckDB SQL. No explanations.**:"""
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    response = model.generate_content(
        sql_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0
        )
    )
    sql_query = response.text.strip()
    if "```sql" in sql_query:
        sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql_query:
        sql_query = sql_query.split("```")[1].split("```")[0].strip()
    
    return sql_query

def fix_sql_syntax_only(conn, sql_query: str, error_message: str, schema_info: dict, api_key: str) -> str:
    """Fix ONLY syntax errors - cannot change query logic or intent"""
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    
    schema_description = "Available tables and columns:\n\n"
    for table_name, info in schema_info.items():
        schema_description += f"Table: {table_name}\n"
        for col in info['columns']:
            schema_description += f"  - {col['name']} ({col['type']})\n"
        schema_description += "\n"
    
    fix_prompt = f"""Fix ONLY the syntax error in this DuckDB SQL query.

    SCHEMA:
    {schema_description}

    CURRENT QUERY:
    {sql_query}

    ERROR:
    {error_message}

    ALLOWED FIXES ONLY:
    - Fix column name typos (match exact schema names)
    - Fix table name typos
    - Add missing GROUP BY for aggregations
    - Fix DuckDB syntax errors
    - Fix quote issues

    FORBIDDEN:
    - Do NOT change query logic
    - Do NOT add/remove joins
    - Do NOT add/remove filters
    - Do NOT add/remove aggregations
    - Do NOT change the intent

    OUTPUT (corrected SQL only, no markdown):"""
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(
        fix_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1
        )
    )
    
    fixed_query = response.text.strip()
    
    if "```sql" in fixed_query:
        fixed_query = fixed_query.split("```sql")[1].split("```")[0].strip()
    elif "```" in fixed_query:
        fixed_query = fixed_query.split("```")[1].split("```")[0].strip()
    
    return fixed_query

def execute_single_query_with_retry(conn, sql_query: str, schema_info: dict, api_key: str, max_retries: int = 3):
    """Execute single SQL query with syntax-only retry"""
    import pandas as pd
    
    current_query = sql_query
    
    for attempt in range(max_retries):
        try:
            result = conn.execute(current_query).fetchdf()
            return result, None, current_query
        
        except Exception as e:
            error_message = str(e)
            
            if attempt == max_retries - 1:
                return pd.DataFrame(), f"Query failed after {max_retries} attempts: {error_message}", current_query
            
            # Try to fix syntax only
            current_query = fix_sql_syntax_only(conn, current_query, error_message, schema_info, api_key)
    
    return pd.DataFrame(), "Query execution failed", current_query

def execute_multi_query_plan(multi_plan: dict, conn, schema_info: dict, api_key: str):
    """Execute all queries in the multi-query plan - best effort execution"""
    import pandas as pd
    
    successful_results = {}
    failed_queries = {}
    executed_queries = {}
    
    queries = multi_plan['queries']
    executed_ids = set()
    
    while len(executed_ids) < len(queries):
        progress_made = False
        
        for query_obj in queries:
            query_id = query_obj['id']
            
            if query_id in executed_ids:
                continue
            
            depends_on = query_obj.get('depends_on', [])
            dependencies_met = all(dep_id in executed_ids for dep_id in depends_on)
            
            if dependencies_met:
                sql_query = generate_strict_sql_from_plan(query_obj['plan'], schema_info, api_key)
                
                result_df, error, final_query = execute_single_query_with_retry(
                    conn, sql_query, schema_info, api_key, max_retries=3
                )
                
                executed_queries[query_id] = final_query
                
                if error:
                    failed_queries[query_id] = {
                        "sql": final_query,
                        "error": error
                    }
                else:
                    successful_results[query_id] = result_df
                
                executed_ids.add(query_id)
                progress_made = True
        
        if not progress_made:
            remaining = [q['id'] for q in queries if q['id'] not in executed_ids]
            for query_id in remaining:
                failed_queries[query_id] = {
                    "sql": "Not executed",
                    "error": "Circular dependency or unmet dependency"
                }
                executed_ids.add(query_id)
            break
    #with open("output.txt", "w", encoding="utf-8") as f:
    #    f.write(str(successful_results))
    return {
        "success": successful_results,
        "failed": failed_queries
    }

def generate_final_explanation(question: str, query_results: dict, multi_plan: dict, api_key: str) -> str:
    """Generate explanation strictly based on actual query results"""
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    
    # Build results context
    results_text = ""
    for query_id, df in query_results.items():
        query_obj = next(q for q in multi_plan['queries'] if q['id'] == query_id)
        purpose = query_obj['purpose']
        
        results_text += f"\n\n{query_id} - {purpose}:\n"
        
        if df.empty:
            results_text += "No results\n"
        else:
            results_text += f"({len(df)} rows)\n"
            results_text += df.head(20).to_string(index=False)
            if len(df) > 20:
                results_text += f"\n... and {len(df) - 20} more rows"
    
    final_reasoning = multi_plan.get('final_reasoning', '')
    
    prompt = f"""

        **ROLE**
        You are a **senior data analyst**. Provide a **clear, concise, business-focused explanation** using **ONLY the actual query results provided**.

        ---

        ### 📌 USER QUESTION

        ```
        {question}
        ```

        ---

        ### 🧠 QUERY PLAN CONTEXT (FOR UNDERSTANDING ONLY)

        ```
        {final_reasoning}
        ```

        > This is intent context only. **Do NOT use it to infer, assume, or add information.**

        ---

        ### 📊 ACTUAL QUERY RESULTS (SOURCE OF TRUTH)

        ```
        {results_text}
        ```

        ---

        ## 🚫 STRICT RULES (MANDATORY)

        * Base your response **ONLY** on `ACTUAL QUERY RESULTS`
        * **Do NOT**:

        * Infer trends, causes, or correlations
        * Assume missing values or external context
        * Add domain knowledge not present in results
        * Hallucinate metrics, percentages, or comparisons
        * Every insight **MUST reference explicit values** from the results
        * Quantify insights **only if numbers are present**
        * If results are:
        * **Empty** → state that clearly, ask the user to retry
        * No recommendations unless directly supported by data

        ---

        ## 📐 ANALYSIS RULES

        * Use exact numbers, labels, and categories as shown
        * Comparisons are allowed **only if both values exist**
        * No speculation, no storytelling

        ---

        ## 🧾 OUTPUT FORMAT (SHORT & STYLED — MANDATORY)

        ###  Analysis Summary

        1–2 lines stating exactly what the data shows, using only visible values.

        ###  Key Findings

        * Bullet points with explicit numbers or categories
        * Format: Metric / Category → Value

        ###  Business Meaning

        * 1 short paragraph explaining what this data indicates
        * If interpretation is not possible:
        Business Strategies, Recommendations, Outcomes


        ---

        ### 📝 FINAL INSTRUCTION

        Respond **only** in the format above.
        Do not add commentary, assumptions, or extra sections.

        """
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2
        )
    )
    print(response.text.strip())
    return response.text.strip()

def df_to_gemini_payload(df, max_rows=200):
    df_sample = df.head(max_rows)
    return {
        "columns": df_sample.columns.tolist(),
        "rows": df_sample.values.tolist(),
        "row_count": len(df)
    }

def generate_ai_plot_from_result(
        df,
        user_question,
        api_key
    ):
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-3-pro-image-preview")

    payload = df_to_gemini_payload(df)

    prompt = f"""You are a senior data analyst with strong business intuition and visualization expertise.

    OBJECTIVE:
    Analyze the provided tabular data and produce a concise, high-impact visual dashboard that directly answers the user’s question.
    PRIORITIES: **ALWAYS ONLY REPLY WITH IMAGES, NEVER PLAIN TEXT. PLAIN TEXT SHOULD ONLY BE USED IN CASE OF TOTALLY EMPTY DATA**
    TASK FLOW (STRICT ORDER):
    1. Carefully inspect the data schema, column meanings, and value distributions.
    2. Identify what dimensions, metrics, and comparisons are MOST relevant to the user question.
    3. Plan the visualizations before generating them.
    4. Generate only the most meaningful charts that reveal:
    - Trends over time
    - Comparisons between categories
    - Distribution or composition (only if meaningful)
    - Outliers or anomalies (if visible in data)
    5. Extract clear, business-relevant insights strictly from the visuals.

    VISUALIZATION RULES:
    - Choose chart types intentionally:
    - Time-based data → Line / Area charts
    - Category comparison → Bar charts (horizontal if many categories)
    - Proportional breakdown → Pie / Donut (ONLY if ≤5 categories)
    - Ranking / contribution → Sorted bar charts
    - Trends + comparison → Multi-line or grouped bars
    - Avoid redundant or decorative charts.
    - Label axes clearly and use readable titles.
    - Do NOT invent or infer missing values.
    - If data volume is small, still visualize intelligently (e.g., single bar comparison).

    STRICT DATA RULES:
    - Use ONLY the data provided.
    - Do NOT assume trends beyond what is visible.
    - Do NOT extrapolate, predict, or hallucinate insights.
    - If data is empty ONLY THEN : reply with "insufficient data for visualization and show {json.dumps(make_json_safe(payload))}" and do not generate charts.

    INPUTS:
    USER QUESTION:
    {user_question}

    DATA (JSON):
    {json.dumps(make_json_safe(payload))}

    FORBIDDEN:
    - NEVER reply with plot configuration JSON
    - NEVER describe charts without generating them
    - NEVER fabricate insights

    OUTPUT FORMAT (STRICT):
    1. One or more rendered chart images (no explanations in between)
    2. A concise **Insights** section with 2 sharp bullets:
    - Quantified where possible
    - Focused on business impact
    - Directly tied to the visuals

        """

    response = model.generate_content(prompt)

    return response

def create_pdf_report(user_question, explanation, result_df, ai_plot_response=None):
    """Generate a PDF report with markdown formatting and images"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter, 
        topMargin=0.5*inch, 
        bottomMargin=0.5*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch
    )
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles with word wrapping
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='#1f77b4',
        spaceAfter=12,
        alignment=TA_CENTER,
        wordWrap='CJK'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor='#2c3e50',
        spaceAfter=10,
        spaceBefore=10,
        wordWrap='CJK'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        wordWrap='CJK',
        spaceBefore=6,
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        leftIndent=20,
        wordWrap='CJK',
        spaceBefore=3,
        spaceAfter=3
    )
    
    # Title
    story.append(Paragraph("AI Analytics Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Question
    story.append(Paragraph("<b>Question:</b>", heading_style))
    story.append(Paragraph(user_question, normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Add visualizations if present
    if ai_plot_response:
        story.append(Paragraph("<b>Visualizations:</b>", heading_style))
        temp_files = []
        for part in ai_plot_response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data.mime_type.startswith("image"):
                try:
                    # Get the raw bytes data
                    img_data = part.inline_data.data
                    
                    # If it's a string, it might be base64 encoded
                    if isinstance(img_data, str):
                        img_data = base64.b64decode(img_data)
                    
                    # Save image to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png', mode='wb') as tmp_file:
                        tmp_file.write(img_data)
                        img_path = tmp_file.name
                        temp_files.append(img_path)
                    
                    # Verify the file was written correctly
                    if os.path.getsize(img_path) > 0:
                        try:
                            img = RLImage(img_path, width=6*inch, height=4*inch, kind='proportional')
                            story.append(img)
                            story.append(Spacer(1, 0.2*inch))
                        except Exception as img_error:
                            story.append(Paragraph(f"[Image could not be rendered]", normal_style))
                            story.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    story.append(Paragraph(f"[Error processing image]", normal_style))
                    story.append(Spacer(1, 0.2*inch))
        
        story.append(Spacer(1, 0.1*inch))
    
    # Explanation (convert markdown to PDF-friendly format)
    story.append(Paragraph("<b>Analysis:</b>", heading_style))
    
    # Process the explanation text line by line to preserve formatting
    try:
        # Split explanation into lines
        lines = explanation.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.05*inch))
                continue
            
            # Check if it's a header (starts with ###, ##, or #)
            if line.startswith('###'):
                text = line.replace('###', '').strip()
                story.append(Paragraph(f"<b>{text}</b>", styles['Heading3']))
            elif line.startswith('##'):
                text = line.replace('##', '').strip()
                story.append(Paragraph(f"<b>{text}</b>", heading_style))
            elif line.startswith('#'):
                text = line.replace('#', '').strip()
                story.append(Paragraph(f"<b>{text}</b>", heading_style))
            # Check if it's a bullet point
            elif line.startswith('*') or line.startswith('-') or line.startswith('•'):
                text = line.lstrip('*-• ').strip()
                story.append(Paragraph(f"• {text}", bullet_style))
            # Check if it contains bold text
            elif '**' in line:
                # Replace markdown bold with HTML bold
                text = line.replace('**', '<b>', 1).replace('**', '</b>', 1)
                story.append(Paragraph(text, normal_style))
            else:
                story.append(Paragraph(line, normal_style))
    
    except Exception as e:
        # Fallback to raw text if processing fails
        story.append(Paragraph(explanation, normal_style))
    
    # Build PDF
    try:
        doc.build(story)
    except Exception as e:
        # If PDF build fails, clean up temp files and re-raise
        if ai_plot_response and 'temp_files' in locals():
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
        raise e
    
    # Clean up temporary image files
    if ai_plot_response and 'temp_files' in locals():
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
    
    buffer.seek(0)
    return buffer

def create_markdown_pdf_report(user_query, markdown_response):
    """
    Generate a robust PDF report from markdown response with proper handling of:
    - Markdown formatting (**, ###, ##, #)
    - HTML tables with color coding
    - Bullet points and lists
    - Error recovery and fallback mechanisms
    """
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch,
        leftMargin=0.5*inch,
        rightMargin=0.5*inch
    )
    story = []
    styles = getSampleStyleSheet()
    
    # Define custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        leading=20
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=10,
        spaceBefore=14,
        fontName='Helvetica-Bold',
        leading=17
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=8,
        spaceBefore=10,
        fontName='Helvetica-Bold',
        leading=15
    )
    
    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontSize=10,
        textColor=colors.HexColor('#465668'),
        spaceAfter=6,
        spaceBefore=8,
        fontName='Helvetica-Bold',
        leading=13
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        alignment=TA_LEFT,
        spaceBefore=4,
        spaceAfter=4
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        leftIndent=20,
        firstLineIndent=-10,
        spaceBefore=2,
        spaceAfter=2
    )
    
    cell_style = ParagraphStyle(
        'CellStyle',
        parent=styles['Normal'],
        fontSize=7,
        leading=9,
        alignment=TA_LEFT
    )
    
    def clean_markdown_text(text):
        """
        Clean and convert markdown to HTML tags for ReportLab
        Handles: **, *, italic, etc.
        """
        if not text:
            return ""
        
        text = text.strip()
        
        # Handle bold (**text**)
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        
        # Handle italic (*text* or _text_)
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
        text = re.sub(r'_(.+?)_', r'<i>\1</i>', text)
        
        # Handle inline code (`code`)
        text = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', text)
        
        return text
    
    def sanitize_for_paragraph(text):
        """
        Sanitize text for ReportLab Paragraph with proper escaping
        """
        if not text:
            return ""
        
        # First, clean markdown
        text = clean_markdown_text(text)
        
        # Extract HTML tags we want to preserve
        tags_to_preserve = []
        tag_pattern = r'(</?(?:b|i|u|font)[^>]*>)'
        parts = re.split(tag_pattern, text)
        
        result = []
        for part in parts:
            if re.match(tag_pattern, part):
                # This is a tag we want to preserve
                tags_to_preserve.append(part)
                result.append(f"__TAG_{len(tags_to_preserve)-1}__")
            else:
                # Escape this part
                result.append(escape(part))
        
        # Reconstruct with preserved tags
        text = ''.join(result)
        for i, tag in enumerate(tags_to_preserve):
            text = text.replace(f"__TAG_{i}__", tag)
        
        return text
    
    def safe_paragraph(text, style):
        """Create a paragraph with comprehensive error handling"""
        try:
            cleaned = sanitize_for_paragraph(text)
            return Paragraph(cleaned, style)
        except Exception as e:
            # Fallback: create plain text paragraph
            try:
                plain_text = re.sub(r'<[^>]*>', '', text)
                return Paragraph(escape(plain_text), style)
            except:
                # Ultimate fallback
                return Paragraph("(Content could not be rendered)", style)
    
    def parse_color(color_str):
        """Parse color from various formats to ReportLab color"""
        if not color_str:
            return None
        
        try:
            color_str = color_str.strip().lower()
            
            # Hex color
            if color_str.startswith('#'):
                return colors.HexColor(color_str)
            
            # RGB format: rgb(r, g, b)
            rgb_match = re.match(r'rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', color_str)
            if rgb_match:
                r, g, b = map(int, rgb_match.groups())
                return colors.Color(r/255.0, g/255.0, b/255.0)
            
            # Named colors
            color_map = {
                'red': colors.red,
                'green': colors.green,
                'blue': colors.blue,
                'yellow': colors.yellow,
                'orange': colors.orange,
                'gray': colors.gray,
                'grey': colors.grey,
                'white': colors.white,
                'black': colors.black,
            }
            if color_str in color_map:
                return color_map[color_str]
            
        except Exception:
            pass
        
        return None
    
    def extract_cell_color(cell):
        """Extract background color from cell style attribute"""
        style_attr = cell.get('style', '')
        
        # Look for background-color or background
        bg_patterns = [
            r'background-color\s*:\s*([^;]+)',
            r'background\s*:\s*([^;]+)',
        ]
        
        for pattern in bg_patterns:
            match = re.search(pattern, style_attr, re.IGNORECASE)
            if match:
                color_value = match.group(1).strip()
                color = parse_color(color_value)
                if color:
                    return color
        
        return None
    
    def process_table(table_html):
        """Process HTML table and convert to ReportLab Table with colors"""
        try:
            soup = BeautifulSoup(table_html, 'html.parser')
            table_elem = soup.find('table')
            
            if not table_elem:
                return None
            
            table_data = []
            cell_colors = []
            
            # Process all rows
            for tr in table_elem.find_all('tr'):
                row_data = []
                row_colors = []
                
                for cell in tr.find_all(['th', 'td']):
                    # Get cell text and clean it
                    cell_text = cell.get_text().strip()
                    cell_para = safe_paragraph(cell_text, cell_style)
                    row_data.append(cell_para)
                    
                    # Extract color
                    color = extract_cell_color(cell)
                    row_colors.append(color)
                
                if row_data:
                    table_data.append(row_data)
                    cell_colors.append(row_colors)
            
            if not table_data:
                return None
            
            # Calculate column widths
            num_cols = len(table_data[0])
            available_width = 7 * inch
            col_widths = [available_width / num_cols] * num_cols
            
            # Create table
            pdf_table = Table(table_data, colWidths=col_widths, repeatRows=1)
            
            # Base style commands
            style_commands = [
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ]
            
            # Apply cell-specific colors
            for row_idx, row_colors in enumerate(cell_colors):
                for col_idx, color in enumerate(row_colors):
                    if color:
                        style_commands.append(
                            ('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), color)
                        )
            
            # Default header background if no colors specified
            if all(c is None for c in cell_colors[0]):
                style_commands.append(
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f2f2f2'))
                )
            
            pdf_table.setStyle(TableStyle(style_commands))
            return pdf_table
            
        except Exception as e:
            print(f"Error processing table: {e}")
            return None
    
    def process_markdown_content(content):
        """Process markdown content and extract tables"""
        # Find all tables
        table_pattern = r'<table[^>]*>.*?</table>'
        tables = list(re.finditer(table_pattern, content, flags=re.DOTALL | re.IGNORECASE))
        
        segments = []
        current_pos = 0
        
        for match in tables:
            # Text before table
            text_before = content[current_pos:match.start()]
            if text_before.strip():
                segments.append(('text', text_before))
            
            # Table
            segments.append(('table', match.group(0)))
            current_pos = match.end()
        
        # Remaining text
        if current_pos < len(content):
            remaining = content[current_pos:]
            if remaining.strip():
                segments.append(('text', remaining))
        
        return segments
    
    def add_text_block(text):
        """Add text block with proper markdown parsing"""
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Heading level 1 (#)
            if line.startswith('# ') and not line.startswith('##'):
                text = line[2:].strip()
                story.append(safe_paragraph(text, heading1_style))
            
            # Heading level 2 (##)
            elif line.startswith('## ') and not line.startswith('###'):
                text = line[3:].strip()
                story.append(safe_paragraph(text, heading2_style))
            
            # Heading level 3 (###)
            elif line.startswith('### '):
                text = line[4:].strip()
                story.append(safe_paragraph(text, heading3_style))
            
            # Bullet points
            elif line.startswith(('* ', '- ', '• ')):
                text = line.lstrip('*-• ').strip()
                story.append(safe_paragraph(f"• {text}", bullet_style))
            
            # Numbered lists
            elif re.match(r'^\d+\.\s', line):
                text = re.sub(r'^\d+\.\s*', '', line)
                story.append(safe_paragraph(f"  {text}", bullet_style))
            
            # Regular paragraph
            else:
                story.append(safe_paragraph(line, normal_style))
    
    # Build the PDF
    try:
        # Title
        story.append(safe_paragraph("Sales & Market Intelligence Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Question section
        story.append(safe_paragraph("<b>Question:</b>", heading2_style))
        story.append(safe_paragraph(user_query, normal_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Analysis section
        story.append(safe_paragraph("<b>Analysis:</b>", heading2_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Process content
        segments = process_markdown_content(markdown_response)
        
        for seg_type, seg_content in segments:
            if seg_type == 'text':
                add_text_block(seg_content)
            elif seg_type == 'table':
                table = process_table(seg_content)
                if table:
                    story.append(Spacer(1, 0.1*inch))
                    story.append(table)
                    story.append(Spacer(1, 0.15*inch))
        
        # Build PDF
        doc.build(story)
        
    except Exception as e:
        # Fallback: create simple error report
        print(f"Error building PDF: {e}")
        story = [
            Paragraph("Sales & Market Intelligence Report", title_style),
            Spacer(1, 0.2*inch),
            Paragraph(f"<b>Question:</b> {escape(user_query)}", normal_style),
            Spacer(1, 0.2*inch),
            Paragraph("<b>Analysis:</b>", heading2_style),
            Spacer(1, 0.1*inch),
            Paragraph("An error occurred while generating the full report. Please try again or view the analysis in the web interface.", normal_style),
            Spacer(1, 0.1*inch),
            Paragraph(f"<i>Error details: {escape(str(e))}</i>", normal_style)
        ]
        doc.build(story)
    
    buffer.seek(0)
    return buffer