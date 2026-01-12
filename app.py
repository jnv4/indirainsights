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
from database import (
    has_datasets,
    check_password,
    make_json_safe,
    clean_dataframe,
    extract_schema,
    get_dataset,
    get_schema_info_from_azure,
    initialize_duckdb_from_azure_datasets,
    get_all_azure_tables,
    load_selected_azure_tables
)
from ai_utils import identify_relevant_files, generate_final_explanation, df_to_gemini_payload, generate_ai_plot_from_result
from  query import get_duckdb_tables, extract_duckdb_schema, create_multi_query_plan, generate_strict_sql_from_plan, fix_sql_syntax_only, execute_multi_query_plan, execute_single_query_with_retry
from utils import create_markdown_pdf_report, create_pdf_report, perform_web_search
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

# Default tables to select (case-insensitive matching)
DEFAULT_TABLES = [
    "IIVF_Master_Clinics",
    "IIVF_Fact_Payments",
    "IIVF_Master_Employees",
    "IIVF_Master_SourceSubCategories",
    "IIVF_Master_SourceCategories",
    "IIVF_Fact_Transactions",
    "IIVF_Master_VisitCategories"
]

# Initialize session state variables
if 'available_tables' not in st.session_state:
    st.session_state.available_tables = []
if 'selected_tables' not in st.session_state:
    st.session_state.selected_tables = []
if 'active_datasets' not in st.session_state:
    st.session_state.active_datasets = {}
if 'tables_fetched' not in st.session_state:
    st.session_state.tables_fetched = False

# Fetch available tables only once
if not st.session_state.tables_fetched:
    with st.spinner("Fetching available tables from Azure SQL..."):
        from database import get_all_azure_tables
        try:
            st.session_state.available_tables = get_all_azure_tables()
            
            # Set default selected tables with case-insensitive matching
            available_lower = {table.lower(): table for table in st.session_state.available_tables}
            st.session_state.selected_tables = [
                available_lower[table.lower()] 
                for table in DEFAULT_TABLES 
                if table.lower() in available_lower
            ]
            
            st.session_state.tables_fetched = True
        except Exception as e:
            st.error(f"Failed to fetch tables: {str(e)}")

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
    
    if not st.session_state.available_tables:
        st.sidebar.warning("⚠️ No tables available from Azure SQL")
        st.info("Could not fetch tables from Azure SQL. Please check your connection.")
    else:
        st.sidebar.markdown(f"**Available Tables:** {len(st.session_state.available_tables)}")
        
        # Multi-select for tables (limit to 20)
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Select Tables to Load** (Max 20)")
        
        selected = st.sidebar.multiselect(
            "Choose tables:",
            options=st.session_state.available_tables,
            default=st.session_state.selected_tables,
            max_selections=20,
            key="table_selector"
        )
        
        st.session_state.selected_tables = selected
        
        # Action buttons
        col1, col2 = st.sidebar.columns(2)
        
        load_btn = col1.button(
            "🔄 Load Data",
            disabled=len(selected) == 0,
            width='stretch'
        )
        
        clear_btn = col2.button(
            "❌ Clear All",
            width='stretch'
        )
        
        if clear_btn:
            st.session_state.selected_tables = []
            st.session_state.active_datasets = {}
            st.rerun()
        
        if load_btn and selected:
            with st.spinner(f"Loading {len(selected)} tables..."):
                loaded_datasets = load_selected_azure_tables(selected)
                st.session_state.active_datasets = loaded_datasets
                st.sidebar.success(f"✅ Loaded {len(loaded_datasets)} tables")
                st.rerun()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Currently Loaded:** {len(st.session_state.active_datasets)} tables")
    
    # Main content area
    if not st.session_state.active_datasets:
        st.info("👈 Please select and load tables from the sidebar to view data")
    else:
        # List loaded tables
        st.markdown("## 📊 Loaded Tables")
        
        table_names = list(st.session_state.active_datasets.keys())
        
        selected_table = st.selectbox(
            "Select a table to view",
            table_names,
            key="selected_table_view"
        )
        
        if selected_table:
            df = get_dataset(selected_table)
            
            st.markdown(f"### 📋 {selected_table}")
            
            # Basic metrics
            total_records = len(df)
            num_columns = len(df.columns)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Records", f"{total_records:,}")
            c2.metric("Columns", num_columns)
            c3.metric("Memory Size", f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
            
            st.divider()
            
            # Show data preview
            st.markdown("#### 📋 Data Preview")
            st.dataframe(df, width='stretch')
            
            # Column info
            with st.expander("🔍 Column Information"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count().values,
                    'Null': df.isnull().sum().values,
                    'Unique': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, width='stretch')

with tab2:
    st.markdown("## 🤖 Sales & Market Intelligence Report Bot")

    # Initialize session state variables
    if 'tab2_response' not in st.session_state:
        st.session_state.tab2_response = None
    if 'tab2_query' not in st.session_state:
        st.session_state.tab2_query = None
    if 'tab2_citations' not in st.session_state:
        st.session_state.tab2_citations = []
    if 'tab2_data_source' not in st.session_state:
        st.session_state.tab2_data_source = "internal"

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
            
            # Question input
            user_query = st.text_area(
                "Your Question", 
                height=120, 
                label_visibility="visible", 
                key="tab2_query_input",
                placeholder="E.g., What are the latest trends in IVF marketing? Or analyze our Q4 performance."
            )
            
            # Data source radio buttons
            st.markdown("### 📊 Select Data Source")
            data_source = st.radio(
                "Choose where to get information from:",
                options=["internal", "web_search", "both"],
                format_func=lambda x: {
                    "internal": "📊 Internal Data Only - Use only our Azure SQL data",
                    "web_search": "🌐 Web Search Only - Search the internet for current information", 
                    "both": "🔄 Both - Combine internal data with web research"
                }[x],
                key="tab2_data_source_radio",
                horizontal=False
            )
            
            # Action buttons
            col1, col2 = st.columns([3, 1])
            with col1:
                analyze_btn = st.button("🔍 Analyze", type="primary", key="tab2_analyze_btn", use_container_width=True)
    
            if analyze_btn and user_query:
                st.session_state.active_tab = 1
                st.session_state.tab2_query = user_query
                st.session_state.tab2_response = None
                st.session_state.tab2_citations = []
                st.session_state.tab2_data_source = data_source
                
                try:
                    data_context = ""
                    web_search_results = None
                    
                    # ========== HANDLE INTERNAL DATA ==========
                    if data_source in ["internal", "both"]:
                        with st.spinner("📊 Loading internal data..."):
                            # Get all available tables
                            all_tables = list(st.session_state.active_datasets.keys())
                            
                            # Identify relevant tables using AI
                            relevant_files = identify_relevant_files(user_query, st.session_state.gemini_api_key)
                            
                            # Filter to only existing tables
                            relevant_files = [t for t in relevant_files if t in all_tables]
                            if not relevant_files:
                                relevant_files = all_tables
                            
                            st.info(f"📂 Using {len(relevant_files)} relevant tables: {', '.join(relevant_files)}")
                            
                            relevant_data = {}
                            for file_name in relevant_files:
                                try:
                                    relevant_data[file_name] = get_dataset(file_name)
                                except Exception as e:
                                    st.warning(f"Could not load {file_name}: {str(e)}")
                            
                            # Build data context
                            for name, data in relevant_data.items():
                                df = data['dataframe'] if isinstance(data, dict) else data
                                table_name = name.lower().replace(" ", "_").replace("-", "_")

                                data_context += f"\n\n### Dataset: {name}\n"
                                data_context += f"Table name: {table_name}\n"
                                data_context += f"Total rows: {len(df)}\n"
                                data_context += f"Columns ({len(df.columns)}): {', '.join(df.columns)}\n"

                                # Numeric summary
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

                                # Categorical dominance
                                categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
                                if categorical_cols:
                                    data_context += "\nCategorical dominance (top 3 values):\n"
                                    for col in categorical_cols[:4]:
                                        vc = df[col].value_counts(normalize=True).head(3)
                                        dominance = {k: round(v * 100, 2) for k, v in vc.items()}
                                        data_context += f"- {col}: {dominance}\n"

                                # Skew info
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
                                
                                # Missing data
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
                    
                    # ========== HANDLE WEB SEARCH ==========
                    if data_source in ["web_search", "both"]:
                        with st.spinner("🌐 Searching the web..."):
                            from utils import perform_web_search
                            web_search_results = perform_web_search(user_query, st.session_state.gemini_api_key)
                            
                            if web_search_results["success"]:
                                st.session_state.tab2_citations = web_search_results["citations"]
                                st.success(f"✅ Found {len(web_search_results['citations'])} web sources")
                            else:
                                st.error(f"❌ Web search failed: {web_search_results['error']}")
                                with st.expander("🔍 Debug Info"):
                                    st.code(web_search_results['error'])
                    
                    # ========== GENERATE FINAL RESPONSE ==========
                    with st.spinner("🤔 Generating comprehensive insights..."):
                        # Build the prompt
                        full_prompt = f"""{SYSTEM_PROMPT}

                        **Your answer should be VERY detailed**

                        Always consider cultural, regional, demographic, socio-economic, and platform-specific factors when analyzing IVF-related data, insights, or strategies. Additionally, factor in all relevant marketing dimensions, including audience segmentation, messaging sensitivity, channel effectiveness, regional ad behavior, conversion funnels, trust signals, compliance constraints, and competitive positioning.

                        Deliver a detailed, region-wise comparative analysis covering both cultural context and marketing performance, with clear rationale, trade-offs, and actionable implications. No shortcuts, no surface-level summaries—depth, rigor, and completeness are mandatory.
                        """
                        
                        # Add internal data context
                        if data_context:
                            full_prompt += f"\n\n## INTERNAL DATA AVAILABLE:\n{data_context}"
                        
                        # Add web search results
                        if web_search_results and web_search_results["success"]:
                            full_prompt += f"\n\n## WEB SEARCH RESULTS:\n{web_search_results['text']}\n"
                            
                            if web_search_results['citations']:
                                full_prompt += "\n### SOURCES:\n"
                                for idx, citation in enumerate(web_search_results['citations'], 1):
                                    full_prompt += f"[{idx}] {citation['title']}\n"
                                    full_prompt += f"    URL: {citation['uri']}\n"
                                    if citation.get('snippet'):
                                        full_prompt += f"    Context: {citation['snippet']}\n"
                                
                                full_prompt += "\n**IMPORTANT**: Reference these sources in your response using [1], [2], etc. when citing web information.\n"
                        
                        # Add the user question
                        full_prompt += f"\n\n## USER QUESTION:\n{user_query}\n"
                        
                        # Add formatting instructions
                        full_prompt += """

                        ## FORMATTING REQUIREMENTS:

                        Always colour code **numerical** values full cells in tables.

                        When returning tables:
                        - Use HTML tables only (DO NOT use markdown tables - STRICTLY HTML)
                        - Apply color coding using HEX colors ONLY
                        - Use ONLY the following palette:
                        * Positive / Outperforming: #34d399
                        * Competitive / Needs optimization: #fbbf24
                        * Underperforming: #f87171
                        * Benchmark / Reference: #93c5fd
                        - Do not invent new colors
                        - Do not explain colors outside the table

                        When using web search results:
                        - Clearly indicate which insights come from web sources vs internal data
                        - Use citation numbers [1], [2], etc. for web sources
                        - Provide a balanced analysis combining both sources when available
                        """
                        
                        # Generate response
                        genai.configure(api_key=st.session_state.gemini_api_key)
                        model = genai.GenerativeModel('gemini-3-pro-preview')
                        response = model.generate_content(full_prompt)
                        
                        st.session_state.tab2_response = response.text
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("🔍 Full Error Details"):
                        st.code(traceback.format_exc())
            
            # ========== DISPLAY RESULTS ==========
            if st.session_state.tab2_response:
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Show data source indicator
                source_icon = {
                    "internal": "📊",
                    "web_search": "🌐",
                    "both": "🔄"
                }
                st.info(f"{source_icon.get(st.session_state.tab2_data_source, '📊')} Data Source: **{st.session_state.tab2_data_source.replace('_', ' ').title()}**")
                
                # Display the main response
                st.markdown(st.session_state.tab2_response, unsafe_allow_html=True)
                
                # ========== DISPLAY CITATIONS ==========
                if st.session_state.tab2_data_source in ["web_search", "both"] and st.session_state.tab2_citations:
                    st.markdown("---")
                    st.markdown("### 📚 Sources & Citations")
                    
                    st.markdown("""
                    <style>
                    .citation-box {
                        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                        border-left: 4px solid #1f77b4;
                        padding: 16px;
                        margin: 12px 0;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        transition: transform 0.2s;
                    }
                    .citation-box:hover {
                        transform: translateX(5px);
                        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                    }
                    .citation-number {
                        display: inline-block;
                        background: #1f77b4;
                        color: white;
                        width: 28px;
                        height: 28px;
                        border-radius: 50%;
                        text-align: center;
                        line-height: 28px;
                        font-weight: bold;
                        margin-right: 10px;
                    }
                    .citation-title {
                        font-weight: bold;
                        color: #2c3e50;
                        font-size: 1.05em;
                        margin-bottom: 8px;
                    }
                    .citation-url {
                        color: #1f77b4;
                        font-size: 0.9em;
                        word-break: break-all;
                        text-decoration: none;
                    }
                    .citation-url:hover {
                        text-decoration: underline;
                    }
                    .citation-snippet {
                        color: #666;
                        font-size: 0.9em;
                        margin-top: 8px;
                        font-style: italic;
                        line-height: 1.4;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    for idx, citation in enumerate(st.session_state.tab2_citations, 1):
                        snippet_html = ""
                        if citation.get('snippet'):
                            snippet_html = f'<div class="citation-snippet">"{citation["snippet"]}"</div>'
                        
                        st.markdown(f"""
                        <div class="citation-box">
                            <div>
                                <span class="citation-number">{idx}</span>
                                <span class="citation-title">{citation['title']}</span>
                            </div>
                            <div style="margin-left: 38px;">
                                <a href="{citation['uri']}" target="_blank" class="citation-url">🔗 {citation['uri']}</a>
                                {snippet_html}
                            
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # ========== PDF DOWNLOAD ==========
                st.markdown("---")
                st.markdown("### 📥 Export Options")
                
                try:
                    with st.spinner("📄 Preparing PDF report..."):
                        pdf_buffer = create_markdown_pdf_report(
                            st.session_state.tab2_query,
                            st.session_state.tab2_response,
                            st.session_state.tab2_citations
                        )
                        
                        if 'report_pdf_data' not in st.session_state:
                            st.session_state.report_pdf_data = {}
                        
                        st.session_state.report_pdf_data['buffer'] = pdf_buffer.getvalue()
                        st.session_state.report_pdf_data['filename'] = f"market_intelligence_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=st.session_state.report_pdf_data['buffer'],
                            file_name=st.session_state.report_pdf_data['filename'],
                            mime="application/pdf",
                            type="secondary",
                            use_container_width=True
                        )
                    with col2:
                        if st.button("🔄 New Analysis", use_container_width=True):
                            st.session_state.tab2_response = None
                            st.session_state.tab2_citations = []
                            st.rerun()
                            
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
                        print(execution_result)
                        successful_results = execution_result["success"]
                        failed_queries = execution_result["failed"]
                        
                        non_empty_results = {
                            qid: df for qid, df in successful_results.items() 
                            if not df.empty
                        }
                        
                        if not non_empty_results:
                            st.error("No relevant data found. Try rephrasing your question.")
                            
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
                        type="secondary",width='stretch'
                    )
                except Exception as pdf_error:
                    st.warning(f"⚠️ Could not generate PDF: {str(pdf_error)}")
