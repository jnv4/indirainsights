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
from utils import (
    create_markdown_pdf_report,
    create_multi_query_plan,
    generate_strict_sql_from_plan,
    fix_sql_syntax_only,
    execute_multi_query_plan,
    execute_single_query_with_retry,
    generate_final_explanation,
    has_datasets,
    create_pdf_report,
    check_password,
    extract_duckdb_schema,
    read_uploaded_file,
    make_json_safe,
    clean_dataframe,
    extract_schema,
    identify_relevant_files,
    top_value_metrics,
    validate_api_key,
    get_dataset,
    get_schema_info_from_azure,
    load_csv,
    initialize_duckdb_from_azure_datasets,
    get_duckdb_tables,
    load_all_azure_tables
)
load_dotenv()

st.set_page_config(
    page_title="Marketing Insights Dashboard",
    layout="wide"
)
def load_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

if not check_password():
    st.stop()

if 'datasets_loaded' not in st.session_state:
    st.session_state.datasets_loaded = False
    st.session_state.active_datasets = {}

if not st.session_state.datasets_loaded:
    with st.spinner("Loading Azure SQL tables..."):
        azure_datasets = load_all_azure_tables()
        
        if azure_datasets:
            st.session_state.active_datasets = azure_datasets
            st.success(f"✅ Loaded {len(azure_datasets)} tables from Azure SQL")
        else:
            st.warning("⚠️ No Azure SQL tables loaded. Check your connection settings.")
        
        st.session_state.datasets_loaded = True

if 'gemini_api_key' not in st.session_state:
    env_api_key = os.getenv('GEMINI_API_KEY')
    st.session_state.gemini_api_key = env_api_key if env_api_key else None
    
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = bool(st.session_state.gemini_api_key)

st.title("📊 Marketing Performance Insights")

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

tab1, tab2, tab3= st.tabs(["📈 Dashboard", "📈AI Report", "🔍 AI Analytics"])
with tab1:
    st.sidebar.title("📊 Azure SQL Tables")

    registered_datasets = st.session_state.active_datasets

    if not registered_datasets:
        st.sidebar.info("🔭 No datasets available")
        st.info("No Azure SQL tables loaded. Check your database connection.")
    else:
        # List all tables in sidebar
        st.sidebar.markdown("**Available Tables:**")
        table_names = list(registered_datasets.keys())
        
        selected_table = st.sidebar.selectbox(
            "Select a table to view",
            table_names,
            key="selected_table"
        )
        
        if selected_table:
            df = get_dataset(selected_table)
            
            st.markdown(f"## 📊 {selected_table}")
            
            # Basic metrics
            total_records = len(df)
            num_columns = len(df.columns)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Records", f"{total_records:,}")
            c2.metric("Columns", num_columns)
            c3.metric("Memory Size", f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
            
            st.divider()
            
            # Show data preview
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(100), use_container_width=True)
            
            # Column info
            with st.expander("📝 Column Information"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count().values,
                    'Null': df.isnull().sum().values,
                    'Unique': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)

with tab2:
    st.markdown("## 🤖 Sales & Market Intelligence Report Bot")

    if 'tab2_response' not in st.session_state:
        st.session_state.tab2_response = None
    if 'tab2_query' not in st.session_state:
        st.session_state.tab2_query = None

    if not has_datasets():
        st.warning("🔭 No datasets available")
        st.info("Upload at least one dataset to enable the report.")
    else:
        if st.session_state.api_key_configured:
            a=1
        else:
            st.warning("⚠️ GEMINI_API_KEY not found.")

        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.api_key_configured:
            from prompts import SYSTEM_PROMPT
            
            user_query = st.text_area("Your Question", height=120, label_visibility="visible", key="tab2_query_input")
            analyze_btn = st.button("🔍 Analyze", type="primary", key="tab2_analyze_btn")
            
            if analyze_btn and user_query:
                st.session_state.active_tab = 1
                st.session_state.tab2_query = user_query
                st.session_state.tab2_response = None
                try:
                    # Get all available tables
                    all_tables = list(st.session_state.active_datasets.keys())

                    # Identify relevant tables using AI
                    relevant_files = identify_relevant_files(user_query, st.session_state.gemini_api_key)

                    # Filter to only existing tables
                    relevant_files = [t for t in relevant_files if t in all_tables]

                    if not relevant_files:
                        st.warning("No relevant tables identified. Using all available tables.")
                        relevant_files = all_tables

                    relevant_data = {}
                    for file_name in relevant_files:
                        with st.spinner(f"Loading {file_name}..."):
                            try:
                                relevant_data[file_name] = get_dataset(file_name)
                            except Exception as e:
                                st.warning(f"Could not load {file_name}: {str(e)}")
                        
                    data_context = ""

                    for name, data in relevant_data.items():
                        df = data['dataframe'] if isinstance(data, dict) else data
                        table_name = name.lower().replace(" ", "_").replace("-", "_")

                        data_context += f"\n\n### Dataset: {name}\n"
                        data_context += f"Table name: {table_name}\n"
                        data_context += f"Total rows: {len(df)}\n"
                        data_context += f"Columns ({len(df.columns)}): {', '.join(df.columns)}\n"

                            # ---------- Numeric summary ----------
                        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                        if numeric_cols:
                            numeric_summary = (
                                df[numeric_cols]
                                .agg(["min", "max", "mean", "median", "std"])
                                .round(2)
                                .to_dict()
                            )

                            data_context += "\nNumeric column summary (distribution-level):\n"
                            for col, stats in numeric_summary.items():
                                data_context += (
                                        f"- {col}: min={stats['min']}, "
                                        f"max={stats['max']}, "
                                        f"mean={stats['mean']}, "
                                        f"median={stats['median']}, "
                                        f"std={stats['std']}\n"
                                    )

                            # ---------- Categorical dominance ----------
                        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
                        if categorical_cols:
                            data_context += "\nCategorical dominance (top 3 values):\n"
                            for col in categorical_cols[:4]:  # cap to avoid noise
                                vc = df[col].value_counts(normalize=True).head(3)
                                dominance = {k: round(v * 100, 2) for k, v in vc.items()}
                                data_context += f"- {col}: {dominance}\n"

                            # ---------- Skew / imbalance signal ----------
                        if numeric_cols:
                            skew_info = {
                                col: round(df[col].skew(), 2)
                                for col in numeric_cols
                                if df[col].nunique() > 5
                            }
                            if skew_info:
                                data_context += "\nDistribution skew indicators:\n"
                                for col, skew in skew_info.items():
                                    data_context += f"- {col}: skew={skew}\n"
                            # ---------- Missing data ---------
                        missing = (
                            df.isna().mean()
                            .mul(100)
                            .round(2)
                            .to_dict()
                        )
                        significant_missing = {k: v for k, v in missing.items() if v > 5}
                        if significant_missing:
                            data_context += "\nMissing data (>5%):\n"
                            for col, pct in significant_missing.items():
                                data_context += f"- {col}: {pct}% missing\n"
                
                    with st.spinner("🤔 Generating insights..."):
                        full_prompt = f"""{SYSTEM_PROMPT}
                            **Your answer should be VERY detailed**
                            Always consider cultural, regional, demographic, socio-economic, and platform-specific factors when analyzing IVF-related data, insights, or strategies. Additionally, factor in all relevant marketing dimensions, including audience segmentation, messaging sensitivity, channel effectiveness, regional ad behavior, conversion funnels, trust signals, compliance constraints, and competitive positioning.

                            Deliver a detailed, region-wise comparative analysis covering both cultural context and marketing performance, with clear rationale, trade-offs, and actionable implications. No shortcuts, no surface-level summaries—depth, rigor, and completeness are mandatory.
                                    DATA AVAILABLE:
                                    {data_context}
                                    USER QUESTION: {user_query}
                            Always colour code **numerical** values full cells in tables.
                            When returning tables:
                            - Use HTML tables only (DO NOT use markdown tables (STRICTLY HTML))
                            - Apply color coding using HEX colors ONLY
                            - Use ONLY the following palette:
                            Positive / Outperforming: #34d399
                            Competitive / Needs optimization: #fbbf24
                            Underperforming: #f87171
                            Benchmark / Reference: #93c5fd
                            - Do not invent new colors
                            - Do not explain colors outside the table


                            """
                        genai.configure(api_key=st.session_state.gemini_api_key)
                            
                        model = genai.GenerativeModel('gemini-2.5-pro')
                        
                        response = model.generate_content(full_prompt)

                        st.session_state.tab2_response = response.text
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("If this is an API key error, please reconfigure your API key using the 'Change API Key' button above")
            
            if st.session_state.tab2_response:
                st.markdown(st.session_state.tab2_response, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # PDF Download Option
                st.markdown("---")
                try:
                    with st.spinner("📄 Preparing PDF report..."):
                        pdf_buffer = create_markdown_pdf_report(
                            st.session_state.tab2_query,
                            st.session_state.tab2_response
                        )
                        
                        if 'report_pdf_data' not in st.session_state:
                            st.session_state.report_pdf_data = {}
                        
                        st.session_state.report_pdf_data['buffer'] = pdf_buffer.getvalue()
                        st.session_state.report_pdf_data['filename'] = f"market_intelligence_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=st.session_state.report_pdf_data['buffer'],
                        file_name=st.session_state.report_pdf_data['filename'],
                        mime="application/pdf",
                        type="secondary",
                        use_container_width=True
                    )
                except Exception as pdf_error:
                    st.warning(f"⚠️ Could not generate PDF: {str(pdf_error)}")

with tab3:
    st.markdown("## 🔍 AI Analytics")
    st.markdown("Ask natural language questions and get answers based on your data using intelligent query planning.")
    
    if 'tab4_explanation' not in st.session_state:
        st.session_state.tab4_explanation = None
    if 'tab4_question' not in st.session_state:
        st.session_state.tab4_question = None
    if 'tab4_plot_response' not in st.session_state:
        st.session_state.tab4_plot_response = None
    if 'tab4_primary_result' not in st.session_state:
        st.session_state.tab4_primary_result = None
    
    if not has_datasets():
        st.warning("🔭 No datasets available")
        st.info("Upload at least one dataset to use the SQL Query Generator.")
    else:
        if not st.session_state.api_key_configured:
            st.warning("⚠️ GEMINI_API_KEY not found. Please configure API key.")
        else:
            # Initialize DuckDB connection
            if 'duckdb_conn' not in st.session_state:
                st.session_state.duckdb_conn = None
            
            # Load data into DuckDB
            with st.spinner("📊 Initializing DuckDB tables from Azure SQL..."):
                try:
                    conn = initialize_duckdb_from_azure_datasets()  # Changed function name
                    st.session_state.duckdb_conn = conn
                        
                except Exception as e:
                    st.error(f"Error initializing DuckDB: {str(e)}")
                    st.stop()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # User input
            # User input
            user_question = st.text_area(
                "Ask a question about your data",
                height=100,
                key="tab4_question_input")
            show_visuals = st.checkbox(
                "📊 Generate visualizations (charts + insights)",
                value=False,
                key="tab4_show_visuals"
            )

            generate_btn = st.button("🔍 Generate Answer", type="primary", key="tab4_generate_btn")
            
            if generate_btn and user_question:
                st.session_state.active_tab = 3
                st.session_state.tab4_question = user_question
                st.session_state.tab4_explanation = None
                st.session_state.tab4_plot_response = None
                st.session_state.tab4_primary_result = None
                with st.spinner("🧠 Analyzing your question..."):
                    try:
                        schema_info = extract_duckdb_schema(st.session_state.duckdb_conn)
                        
                        multi_plan = create_multi_query_plan(
                            user_question,
                            schema_info,
                            st.session_state.gemini_api_key
                        )
                        
                        execution_result = execute_multi_query_plan(
                            multi_plan,
                            st.session_state.duckdb_conn,
                            schema_info,
                            st.session_state.gemini_api_key
                        )
                        
                        successful_results = execution_result["success"]
                        failed_queries = execution_result["failed"]
                        
                        non_empty_results = {
                            qid: df for qid, df in successful_results.items() 
                            if not df.empty
                        }
                        
                        if not non_empty_results:
                            st.warning("No relevant data found. Try rephrasing your question.")
                            
                            if failed_queries:
                                with st.expander("🔍 Debug Info (Optional)", expanded=False):
                                    st.json({"failed_queries": failed_queries})
                            st.stop()
                        
                        explanation = generate_final_explanation(
                            user_question,
                            non_empty_results,
                            multi_plan,
                            st.session_state.gemini_api_key
                        )
                        
                        st.session_state.tab4_explanation = explanation
                        st.session_state.tab4_primary_result = list(non_empty_results.values())[0]
                        
                        if show_visuals:
                            from utils import generate_ai_plot_from_result
                            
                            st.info("📊 Generating visualizations...")
                            ai_plot_response = generate_ai_plot_from_result(
                                st.session_state.tab4_primary_result,
                                user_question,
                                st.session_state.gemini_api_key
                            )
                            
                            st.session_state.tab4_plot_response = ai_plot_response
                        
                        if failed_queries:
                            with st.expander("⚠️ Some queries failed (optional debug)", expanded=False):
                                for qid, fail_info in failed_queries.items():
                                    st.code(f"Query: {qid}\nSQL: {fail_info['sql']}\nError: {fail_info['error']}")
                    
                    except Exception as e:
                        st.error(f"❌ Unexpected Error: {str(e)}")
                        st.info("Something went wrong. Please try again or rephrase your question.")
                        import traceback
                        with st.expander("🔍 Debug Info"):
                            st.code(traceback.format_exc())
            
            if st.session_state.tab4_explanation:
                if st.session_state.tab4_plot_response:
                    for part in st.session_state.tab4_plot_response.candidates[0].content.parts:
                        if hasattr(part, "inline_data") and part.inline_data.mime_type.startswith("image"):
                            st.image(part.inline_data.data)
                        elif hasattr(part, "text"):
                            st.markdown(part.text)
                
                st.markdown(st.session_state.tab4_explanation)
                
                if 'pdf_data' not in st.session_state:
                    st.session_state.pdf_data = {}
                
                st.markdown("---")
                
                try:
                    with st.spinner("📄 Preparing PDF report..."):
                        pdf_buffer = create_pdf_report(
                            st.session_state.tab4_question,
                            st.session_state.tab4_explanation,
                            st.session_state.tab4_primary_result,
                            st.session_state.tab4_plot_response
                        )
                        
                        st.session_state.pdf_data['buffer'] = pdf_buffer.getvalue()
                        st.session_state.pdf_data['filename'] = f"ai_analytics_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=st.session_state.pdf_data['buffer'],
                        file_name=st.session_state.pdf_data['filename'],
                        mime="application/pdf",
                        type="secondary",
                        use_container_width=True
                    )
                except Exception as pdf_error:
                    st.warning(f"⚠️ Could not generate PDF: {str(pdf_error)}")