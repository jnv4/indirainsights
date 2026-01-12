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
    print(sql_query)
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
