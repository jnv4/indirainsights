import streamlit as st
import pandas as pd
import requests
from io import StringIO, BytesIO
from dotenv import load_dotenv
import os
import duckdb
import google.generativeai as genai
import json
from supabase import create_client, Client
from datetime import datetime
import openpyxl
import math
import numpy as np
# Supabase Configuration
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = "marketing-cleaned"
METADATA_TABLE = "dataset_registry"

supabase_client = None

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY missing")

supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

@st.cache_data(show_spinner=False)
def load_csv(url: str) -> pd.DataFrame:
    r = requests.get(url)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

@st.cache_data(show_spinner=False)
def get_schema_info():
    schema_info = {}
    
    # Use active datasets if available
    if st.session_state.active_datasets:
        for name, df in st.session_state.active_datasets.items():
            try:
                schema_info[name] = {
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "sample_values": {col: df[col].dropna().unique()[:5].tolist() for col in df.columns},
                    "row_count": len(df)
                }
            except Exception as e:
                st.error(f"Error loading schema for {name}: {e}")
    return schema_info

def has_datasets() -> bool:
    return bool(st.session_state.get("active_datasets"))

def read_uploaded_file(file):
    """Read uploaded file and return dict of {sheet_name: DataFrame}"""
    file_ext = file.name.split('.')[-1].lower()
    
    try:
        if file_ext == 'csv':
            df = pd.read_csv(file)
            return {file.name.replace('.csv', ''): df}
        
        elif file_ext in ['xlsx', 'xls']:
            excel_file = pd.ExcelFile(file)
            sheets = {}
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                sheets[sheet_name] = df
            return sheets
        
        elif file_ext == 'json':
            content = file.read()
            data = json.loads(content)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError("JSON must be list or dict")
            
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
    """Clean and normalize DataFrame"""
    df_clean = df.copy()
    
    # Normalize column names
    df_clean.columns = [
        col.lower().strip().replace(' ', '_').replace('-', '_')
        for col in df_clean.columns
    ]
    
    # Drop fully empty columns
    df_clean = df_clean.dropna(axis=1, how='all')
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # Trim string values
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def extract_schema(df):
    raw_schema = {
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "row_count": int(len(df)),
        "sample_rows": df.head(5).to_dict(orient="records")
    }
    return make_json_safe(raw_schema)

def upload_to_supabase(bucket, path, content):
    """Upload content to Supabase Storage"""
    if not supabase_client:
        raise Exception("Supabase client not initialized")
    
    try:
        # Delete existing file if present
        try:
            supabase_client.storage.from_(bucket).remove([path])
        except:
            pass
        
        # Upload new file
        result = supabase_client.storage.from_(bucket).upload(
            path=path,
            file=content,
            file_options={"content-type": "text/csv"}
        )
        
        return result
    
    except Exception as e:
        raise Exception(f"Error uploading to Supabase: {str(e)}")

def register_dataset(metadata):
    """Register dataset metadata in Supabase table"""
    if not supabase_client:
        raise Exception("Supabase client not initialized")
    
    try:
        # Check if dataset already exists
        existing = supabase_client.table(METADATA_TABLE).select("*").eq(
            "dataset_name", metadata["dataset_name"]
        ).execute()
        
        if existing.data:
            # Update existing record
            result = supabase_client.table(METADATA_TABLE).update({
                "file_name": metadata["file_name"],
                "bucket": metadata["bucket"],
                "schema": metadata["schema"],
                "row_count": metadata["row_count"],
                "uploaded_at": metadata["uploaded_at"]
            }).eq("dataset_name", metadata["dataset_name"]).execute()
        else:
            # Insert new record
            result = supabase_client.table(METADATA_TABLE).insert(metadata).execute()
        
        return result
    
    except Exception as e:
        raise Exception(f"Error registering dataset: {str(e)}")

def load_registered_datasets():
    if not supabase_client:
        return None
    
    try:
        # Fetch all registered datasets
        result = supabase_client.table(METADATA_TABLE).select("*").execute()
        
        if not result.data:
            return None
        
        datasets = {}
        
        for record in result.data:
            dataset_name = record["dataset_name"]
            file_name = record["file_name"]
            bucket = record["bucket"]
            
            try:
                # Download CSV from Supabase Storage
                file_data = supabase_client.storage.from_(bucket).download(file_name)
                
                # Load into DataFrame
                df = pd.read_csv(BytesIO(file_data))
                datasets[dataset_name] = df
                
            except Exception as e:
                st.warning(f"Could not load dataset {dataset_name}: {str(e)}")
                continue
        
        return datasets if datasets else None
    
    except Exception as e:
        st.warning(f"Error loading registered datasets: {str(e)}")
        return None

def delete_dataset(dataset_name: str):
    if not supabase_client:
        raise Exception("Supabase not configured")

    record = supabase_client.table(METADATA_TABLE)\
        .select("file_name, bucket")\
        .eq("dataset_name", dataset_name)\
        .execute()

    if not record.data:
        raise Exception("Dataset not found")

    file_name = record.data[0]["file_name"]
    bucket = record.data[0]["bucket"]

    # Delete from storage
    supabase_client.storage.from_(bucket).remove([file_name])

    # Delete metadata
    supabase_client.table(METADATA_TABLE)\
        .delete()\
        .eq("dataset_name", dataset_name)\
        .execute()

    # Delete from session
    st.session_state.active_datasets.pop(dataset_name, None)

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
        schema_info = get_schema_info()
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
        
        intent_prompt = f"""You are a data analyst. Based on the user's question, identify which datasets are needed to answer it.

                        {schema_description}

                        User question: {user_query}

                        Return ONLY a JSON array of dataset names that are needed. For example: ["CRM Leads", "Region", "Competition"]

                        Be selective - only include datasets that are directly relevant to answering the question."""
                                
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
        return st.session_state.active_datasets[dataset_name]

    raise ValueError(f"Dataset {dataset_name} not found")
