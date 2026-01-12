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
from database import get_schema_info_from_azure
from markdown import markdown
from bs4 import BeautifulSoup
from database import make_json_safe
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
