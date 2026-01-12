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
import pymssql
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

def get_dataset(dataset_name: str) -> pd.DataFrame:
    if not dataset_name:
        raise ValueError("No dataset selected")

    if dataset_name in st.session_state.get("active_datasets", {}):
        data = st.session_state.active_datasets[dataset_name]
        if isinstance(data, dict):
            return data['dataframe']
        return data  # Backward compatibility

    raise ValueError(f"Dataset {dataset_name} not found")
