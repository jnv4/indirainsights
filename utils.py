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
# Define the 4 fields
AVAILABLE_FIELDS = ["Call Center", "Website", "CRM", "Competitors"]
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
                "uploaded_at": metadata["uploaded_at"],
                "field": metadata.get("field", "Unknown")
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
            field = record.get("field", "Unknown")
            
            try:
                # Download CSV from Supabase Storage
                file_data = supabase_client.storage.from_(bucket).download(file_name)
                
                # Load into DataFrame
                df = pd.read_csv(BytesIO(file_data))
                datasets[dataset_name] = {
                    'dataframe': df,
                    'field': field
                }
                
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
    if dataset_name in st.session_state.active_datasets:
        st.session_state.active_datasets.pop(dataset_name)

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

def create_query_plan(question: str, schema_info: dict, api_key: str) -> dict:
    """Create intelligent query execution plan using enhanced schema"""
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    
    # Build enhanced schema description with intelligence
    schema_description = "AVAILABLE DATASETS WITH INTELLIGENCE:\n\n"
    for table_name, info in schema_info.items():
        schema_description += f"TABLE: {table_name}\n"
        schema_description += f"Role: {info.get('dataset_role', 'unknown')} | Business Entity: {info.get('business_entity', 'Other')}\n"
        schema_description += f"Rows: {info['row_count']}\n"
        
        if info.get('primary_keys'):
            schema_description += f"Primary Keys: {', '.join(info['primary_keys'])}\n"
        if info.get('foreign_key_candidates'):
            schema_description += f"Foreign Key Candidates: {', '.join(info['foreign_key_candidates'])}\n"
        if info.get('time_columns'):
            schema_description += f"Time Columns: {', '.join(info['time_columns'])}\n"
        
        schema_description += "COLUMNS:\n"
        for col in info['columns']:
            schema_description += f"  • {col['name']} ({col['type']})"
            if col.get('distinct_count'):
                schema_description += f" - {col['distinct_count']} distinct values"
            if col.get('sample_values'):
                schema_description += f" - Examples: {', '.join(col['sample_values'][:3])}"
            if col.get('stats') and col['stats']:
                stats = col['stats']
                schema_description += f" - Range: [{stats.get('min')} to {stats.get('max')}]"
            schema_description += "\n"
        schema_description += "\n"
    
    # Check if file analysis with joins is available
    file_analysis = st.session_state.get('last_file_analysis', {})
    suggested_joins = file_analysis.get('joins', [])
    
    join_context = ""
    if suggested_joins:
        join_context = "\n\nSUGGESTED JOINS FROM ANALYSIS:\n"
        for join in suggested_joins:
            join_context += f"- {join['left']} JOIN {join['right']} ON {join['left_column']} = {join['right_column']} ({join['confidence']} confidence)\n"
            join_context += f"  Reasoning: {join['reasoning']}\n"
    
    planning_prompt = f"""You are a data analysis query planner with schema intelligence. Analyze the user's question and create a detailed execution plan.

    SCHEMA WITH INTELLIGENCE:
    {schema_description}
    {join_context}

    USER QUESTION: {question}

    PLANNING GUIDELINES BASED ON DATASET ROLES:
    - FACT tables: Use for aggregations (SUM, COUNT, AVG), metrics analysis
    - DIMENSION tables: Use for filtering, grouping, categorical breakdowns
    - LOOKUP tables: Use for reference data, category names

    Create a JSON execution plan with this EXACT structure:
    {{
        "query_type": "aggregation|filter|join|simple_select|ranking|time_series",
        "primary_table": "table_name",
        "primary_table_role": "fact|dimension|lookup",
        "secondary_tables": ["table2", "table3"],
        "required_columns": [
            {{"table": "table_name", "column": "col_name", "purpose": "grouping|filtering|aggregation|ordering|selection"}}
        ],
        "filters": [
            {{"column": "col_name", "operator": "=|>|<|LIKE|IN", "value": "filter_value", "reasoning": "why this filter"}}
        ],
        "aggregations": [
            {{"function": "COUNT|SUM|AVG|MIN|MAX", "column": "col_name", "alias": "result_name"}}
        ],
        "grouping": ["col1", "col2"],
        "ordering": [
            {{"column": "col_name", "direction": "ASC|DESC", "reasoning": "why this order"}}
        ],
        "limit": 50,
        "joins": [
            {{"left_table": "table1", "right_table": "table2", "left_key": "col1", "right_key": "col2", "type": "INNER|LEFT", "confidence": "high|medium|low"}}
        ],
        "reasoning": "explain the query strategy considering dataset roles and relationships",
        "potential_issues": ["list any data quality or ambiguity concerns"]
    }}

    CRITICAL RULES FOR JOINS:
    - If suggested joins are provided above, USE THEM EXACTLY as specified
    - Only include joins where columns actually exist in both tables
    - Verify column names match the schema exactly
    - Do NOT invent joins not supported by the schema
    - If no valid join exists, leave joins array empty

    OUTPUT (valid JSON only, no markdown):"""
    
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    response = model.generate_content(
        planning_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            response_mime_type="application/json"
        )
    )
    
    plan_text = response.text.strip()
    
    # Clean up markdown if present
    if "```json" in plan_text:
        plan_text = plan_text.split("```json")[1].split("```")[0].strip()
    elif "```" in plan_text:
        plan_text = plan_text.split("```")[1].split("```")[0].strip()
    
    import json
    query_plan = json.loads(plan_text)
    
    return query_plan

def generate_sql_from_plan(query_plan: dict, schema_info: dict, api_key: str) -> str:
    """Generate precise SQL query strictly following the execution plan"""
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    
    # Build focused schema with only relevant tables
    relevant_tables = [query_plan['primary_table']] + query_plan.get('secondary_tables', [])
    
    focused_schema = "RELEVANT SCHEMA:\n\n"
    for table_name in relevant_tables:
        if table_name in schema_info:
            info = schema_info[table_name]
            focused_schema += f"Table: {table_name} ({info['row_count']} rows)\n"
            focused_schema += f"Role: {info.get('dataset_role', 'unknown')}\n"
            focused_schema += "Columns:\n"
            for col in info['columns']:
                focused_schema += f"  - {col['name']} {col['type']}\n"
            focused_schema += "\n"
    
    import json
    plan_str = json.dumps(query_plan, indent=2)
    
    sql_generation_prompt = f"""You are a SQL code generator. Generate ONLY the SQL query following this execution plan EXACTLY.

    SCHEMA:
    {focused_schema}

    EXECUTION PLAN:
    {plan_str}

    CRITICAL RULES:
    1. Follow the execution plan EXACTLY - use specified tables, columns, aggregations, filters, joins, grouping, and ordering
    2. Output ONLY valid DuckDB SQL - no markdown, no comments, no explanations
    3. Use exact table and column names from the schema
    4. For JOINS:
    - Use ONLY the joins specified in the plan
    - Use EXACT column names from left_key and right_key
    - Do NOT infer or invent joins not in the plan
    - If no joins in plan, do NOT add any JOIN clauses
    5. If limit is specified in plan, use LIMIT clause
    6. For aggregations, use the specified function and alias
    7. Apply all filters from the plan using the specified operators
    8. Include GROUP BY if grouping is specified
    9. Include ORDER BY if ordering is specified
    10. Ensure query returns meaningful, summarized results (not raw dumps)

    EXAMPLE JOIN SYNTAX (only if specified in plan):
    SELECT ...
    FROM table1 t1
    INNER JOIN table2 t2 ON t1.column_name = t2.column_name

    OUTPUT (SQL query only):"""
    
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    response = model.generate_content(
        sql_generation_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0
        )
    )
    
    sql_query = response.text.strip()
    
    # Clean up markdown if present
    if "```sql" in sql_query:
        sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql_query:
        sql_query = sql_query.split("```")[1].split("```")[0].strip()
    
    return sql_query

def validate_and_fix_query(sql_query: str, query_plan: dict, schema_info: dict, api_key: str) -> str:
    """Validate query against plan and fix any issues"""
    import google.generativeai as genai
    import json
    
    genai.configure(api_key=api_key)
    
    plan_str = json.dumps(query_plan, indent=2)
    
    validation_prompt = f"""You are a SQL validator. Check if this query correctly implements the execution plan.

        EXECUTION PLAN:
        {plan_str}

        GENERATED QUERY:
        {sql_query}

        Check for:
        1. Are all required columns from the plan included?
        2. Are aggregations correct?
        3. Are filters applied properly?
        4. Is grouping correct?
        5. Is ordering correct?
        6. Are joins correct?

        If the query is correct, output: VALID
        If there are issues, output the CORRECTED query (SQL only, no explanation).

        OUTPUT:"""
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    response = model.generate_content(
        validation_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0
        )
    )
    
    result = response.text.strip()
    
    if result == "VALID" or "VALID" in result:
        return sql_query
    
    # Clean up corrected query
    corrected = result
    if "```sql" in corrected:
        corrected = corrected.split("```sql")[1].split("```")[0].strip()
    elif "```" in corrected:
        corrected = corrected.split("```")[1].split("```")[0].strip()
    
    return corrected

def execute_sql_with_retry(conn, sql_query: str, schema_info: dict, api_key: str, max_retries: int = 3):
    """Execute SQL query with retry mechanism using LLM to fix errors"""
    import pandas as pd
    import google.generativeai as genai
    
    current_query = sql_query
    error_history = []
    
    for attempt in range(max_retries):
        try:
            result = conn.execute(current_query).fetchdf()
            return result, None, current_query
        
        except Exception as e:
            error_message = str(e)
            error_history.append({"query": current_query, "error": error_message})
            
            if attempt == max_retries - 1:
                # Last attempt failed
                return pd.DataFrame(), f"Query failed after {max_retries} attempts. Last error: {error_message}", current_query
            
            # Try to fix the query using LLM with error history
            genai.configure(api_key=api_key)
            
            schema_description = "Available tables and columns:\n\n"
            for table_name, info in schema_info.items():
                schema_description += f"Table: {table_name}\n"
                for col in info['columns']:
                    schema_description += f"  - {col['name']} ({col['type']})\n"
                schema_description += "\n"
            
            error_context = "\n".join([
                f"Attempt {i+1}: {err['error']}" 
                for i, err in enumerate(error_history)
            ])
            
            fix_prompt = f"""Fix this DuckDB SQL query. Previous attempts have failed.

        SCHEMA:
        {schema_description}

        CURRENT QUERY:
        {current_query}

        ERROR HISTORY:
        {error_context}

        LATEST ERROR:
        {error_message}

        Common fixes:
        - Check column names match schema exactly
        - Ensure aggregations use GROUP BY
        - Verify table names are correct
        - Use proper DuckDB syntax

        OUTPUT (corrected SQL only, no markdown):"""
            
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(
                fix_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2
                )
            )
            
            fixed_query = response.text.strip()
            
            # Clean up markdown
            if "```sql" in fixed_query:
                fixed_query = fixed_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in fixed_query:
                fixed_query = fixed_query.split("```")[1].split("```")[0].strip()
            
            current_query = fixed_query
    
    return pd.DataFrame(), "Query execution failed", current_query

def execute_smart_query(question: str, conn, schema_info: dict, api_key: str):
    """Main function to execute smart query with planning"""
    try:
        # Step 1: Create execution plan
        st.info("🧠 Creating query execution plan...")
        query_plan = create_query_plan(question, schema_info, api_key)
        
        # Display plan to user
        with st.expander("📋 Query Execution Plan", expanded=False):
            st.json(query_plan)
        
        # Step 2: Generate SQL from plan
        st.info("⚙️ Generating SQL query...")
        sql_query = generate_sql_from_plan(query_plan, schema_info, api_key)
        
        # Step 3: Validate and fix if needed
        st.info("✅ Validating query...")
        validated_query = validate_and_fix_query(sql_query, query_plan, schema_info, api_key)
        
        # Display query
        st.code(validated_query, language="sql")
        
        # Step 4: Execute with retry
        st.info("🔄 Executing query...")
        result_df, error, final_query = execute_sql_with_retry(
            conn, validated_query, schema_info, api_key, max_retries=3
        )
        
        if error:
            st.error(f"Execution failed: {error}")
            st.code(final_query, language="sql")
            return None, None
        
        return result_df, query_plan
        
    except Exception as e:
        st.error(f"Smart query execution failed: {str(e)}")
        return None, None

def generate_result_explanation(question: str, result_df, query_plan: dict, api_key: str) -> str:
    """Generate business-friendly explanation with multi-file context"""
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    
    # Limit result preview to 20 rows
    result_preview = result_df.head(20)
    
    if result_df.empty:
        result_text = "No results found."
    else:
        result_text = f"Results ({len(result_df)} total rows):\n\n"
        result_text += result_preview.to_string(index=False)
        
        if len(result_df) > 20:
            result_text += f"\n\n... and {len(result_df) - 20} more rows"
    
    # Extract multi-dataset context from query plan
    multi_dataset_context = ""
    if query_plan:
        primary_table = query_plan.get('primary_table', 'unknown')
        primary_role = query_plan.get('primary_table_role', 'unknown')
        secondary_tables = query_plan.get('secondary_tables', [])
        joins = query_plan.get('joins', [])
        
        if secondary_tables or joins:
            multi_dataset_context = f"\n\nMULTI-DATASET CONTEXT:\n"
            multi_dataset_context += f"Primary table: {primary_table} (role: {primary_role})\n"
            if secondary_tables:
                multi_dataset_context += f"Secondary tables: {', '.join(secondary_tables)}\n"
            if joins:
                multi_dataset_context += "Relationships used:\n"
                for join in joins:
                    multi_dataset_context += f"  - {join['left_table']}.{join['left_key']} ↔ {join['right_table']}.{join['right_key']}\n"
    
    prompt = f"""**ROLE:**
    You are a senior business analyst with strong data interpretation skills. Your task is to translate raw query results into clear, actionable business insights for non-technical stakeholders.

    **USER QUESTION:**
    {question}

    **QUERY RESULTS:**
    {result_text}
    {multi_dataset_context}

    **ANALYSIS INSTRUCTIONS:**

    * Start by briefly restating what the query was intended to answer (in plain business terms).
    * If multiple datasets were used, explain how they were combined and what relationships were leveraged.
    * Clearly explain what the data shows, including key metrics, totals, trends, or comparisons visible in the results.
    * Identify notable patterns, outliers, or relationships *only if they are directly supported by the data*.
    * If the results include multiple categories, segments, or time periods, compare them explicitly.
    * Quantify insights wherever possible (percentages, differences, rankings), using only the provided data.
    * If the data is incomplete, sparse, or inconclusive, explicitly state this instead of speculating.
    * Avoid technical SQL or database terminology unless necessary.

    **STRICT CONSTRAINTS:**

    * Do NOT make assumptions beyond the provided results.
    * Do NOT infer causes, forecasts, or recommendations unless they are explicitly supported by the data.
    * Do NOT use external knowledge or industry benchmarks.
    * Do NOT invent missing values or trends.

    **OUTPUT FORMAT:**

    * **Summary:** 1–2 sentences explaining the overall outcome
    * **Key Insights:** Bullet points highlighting the most important observations
    * **Data Relationships:** (if multi-dataset) Briefly explain how datasets were combined and what this reveals
    * **Business Interpretation:** What these results mean for the business, strictly based on the data
    * **Recommendations:** Business insights with recommendations for improvements and growth

    **FINAL EXPLANATION:**"""
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3
        )
    )
    
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

    prompt = f"""
    You are a senior data analyst.
    Respond with a dashboard of plots image
    TASK:
    1. Analyze the tabular data provided
    2. Generate the MOST meaningful visualizations (bar, line, pie, trend, comparison)
    3. Add short, sharp business insights (2–4 bullets max)
    
    RULES:
    - Choose chart types intelligently
    - Prefer clarity over decoration
    - Focus on trends, anomalies, and business impact
    - DO NOT hallucinate missing data
    - If data is small, still visualize intelligently

    USER QUESTION:
    {user_question}

    DATA (JSON):
    {json.dumps(payload)}

    OUTPUT FORMAT:
    - One or more charts
    - Followed by a concise insight message
"""

    response = model.generate_content(prompt)

    return response
