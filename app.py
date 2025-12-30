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
from utils import has_datasets, check_password,generate_sql_from_plan, validate_and_fix_query, execute_sql_with_retry, extract_duckdb_schema, load_registered_datasets, upload_to_supabase, read_uploaded_file, make_json_safe, clean_dataframe, extract_schema, register_dataset, load_registered_datasets, delete_dataset, identify_relevant_files, top_value_metrics, validate_api_key, get_dataset, SUPABASE_BUCKET, get_schema_info, load_csv, AVAILABLE_FIELDS, initialize_duckdb_from_datasets, get_duckdb_tables, create_query_plan, execute_smart_query, generate_result_explanation
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

# ━━━━━━━━━━━━━━━━━━━━ LOAD DATASETS ON LOGIN ━━━━━━━━━━━━━━━━━━━━

if 'datasets_loaded' not in st.session_state:
    st.session_state.datasets_loaded = False
    st.session_state.active_datasets = {}

if not st.session_state.datasets_loaded:
    with st.spinner("Loading registered datasets..."):
        registered_datasets = load_registered_datasets()
        
        if registered_datasets:
            st.session_state.active_datasets = registered_datasets
            st.success(f"✅ Loaded {len(registered_datasets)} datasets from Supabase")

        
        st.session_state.datasets_loaded = True

# ━━━━━━━━━━━━━━━━━━━━ INITIALIZE SESSION STATE ━━━━━━━━━━━━━━━━━━━━

if 'gemini_api_key' not in st.session_state:
    env_api_key = os.getenv('GEMINI_API_KEY')
    st.session_state.gemini_api_key = env_api_key if env_api_key else None
    
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = bool(st.session_state.gemini_api_key)

st.title("📊 Marketing Performance Insights")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "📈AI Report", "📤 Upload Data", "🔍 AI Analytics"])
with tab1:
    st.sidebar.title("📂 Insights By Field")

    registered_datasets = st.session_state.active_datasets

    if not registered_datasets:
        st.sidebar.info("🔭 No datasets available")
        st.info("No datasets found. Please upload data from the **Upload Data** tab.")
    else:
        # Group datasets by field
        datasets_by_field = {}
        for dataset_name, data in registered_datasets.items():
            field = data.get('field', 'Unknown')
            if field not in datasets_by_field:
                datasets_by_field[field] = []
            datasets_by_field[field].append(dataset_name)

        # Sidebar: Select field
        selected_field = st.sidebar.radio(
            "Select a field",
            AVAILABLE_FIELDS,
            key="selected_field"
        )

        # Get all datasets for this field
        field_datasets = datasets_by_field.get(selected_field, [])
        
        if not field_datasets:
            st.info(f"No datasets uploaded for **{selected_field}** yet.")
        else:
            st.markdown(f"## 📊 {selected_field} - Consolidated Insights")
            
            # Aggregate metrics across all datasets in this field
            total_records = 0
            all_primary_values = []

            for dataset_name in field_datasets:
                df = get_dataset(dataset_name)
                total_records += len(df)
                
                # Collect primary column values (excluding NaN, None, empty strings)
                primary_col = df.columns[0]
                valid_values = df[primary_col].dropna().astype(str)
                valid_values = valid_values[valid_values.str.strip() != '']
                valid_values = valid_values[valid_values.str.lower() != 'nan']
                valid_values = valid_values[valid_values.str.lower() != 'none']
                all_primary_values.extend(valid_values.tolist())

            # Calculate aggregated metrics
            if len(all_primary_values) > 0:
                value_counts = pd.Series(all_primary_values).value_counts()
                top_val = value_counts.index[0]
                top_share = round((value_counts.iloc[0] / len(all_primary_values)) * 100, 2)
                low_val = value_counts.index[-1]
            else:
                top_val = "N/A"
                top_share = 0
                low_val = "N/A"
            c1, c2, c3, c4 = st.columns(4)
            
            c1.metric("Total Records", f"{total_records:,}")
            c2.metric("Top Value", top_val)
            c3.metric("Top Share %", f"{top_share}%")
            c4.metric("Lowest Value", low_val)

            # CRM-specific conversion analysis
            if selected_field == "CRM":
                st.divider()
                st.markdown("### 🎯 Conversion Insights (Across All CRM Datasets)")
                
                total_leads = 0
                total_converted = 0
                
                for dataset_name in field_datasets:
                    df = get_dataset(dataset_name)
                    if "Converted" in df.columns or "converted" in df.columns:
                        conv_col = "Converted" if "Converted" in df.columns else "converted"
                        total_leads += len(df)
                        total_converted += len(df[df[conv_col].astype(str).str.lower() == "true"])
                
                if total_leads > 0:
                    conversion_rate = round((total_converted / total_leads) * 100, 2)
                    
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Total Leads", f"{total_leads:,}")
                    cc2.metric("Converted Leads", f"{total_converted:,}")
                    cc3.metric("Conversion Rate", f"{conversion_rate}%")

            # Show individual datasets under this field
            st.divider()
            st.markdown(f"### 📁 Datasets in {selected_field}")
            
            for dataset_name in field_datasets:
                df = get_dataset(dataset_name)
                with st.expander(f"🔍 {dataset_name} ({len(df):,} rows)"):
                    st.dataframe(df.head(100), width='stretch')

with tab2:
    st.markdown("## 🤖 Sales & Market Intelligence Report Bot")

    if not has_datasets():
        st.warning("📭 No datasets available")
        st.info("Upload at least one dataset to enable the report.")
    else:
        if st.session_state.api_key_configured:
            print('<div class="success-box">✅ API Key configured successfully!</div>')
        else:
            st.warning("⚠️ GEMINI_API_KEY not found.")

        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.api_key_configured:
            from prompts import SYSTEM_PROMPT
            
            user_query = st.text_area("Your Question", height=120, label_visibility="visible")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            analyze_btn = st.button("🔍 Analyze", type="primary")
            
            if analyze_btn and user_query:
                with st.spinner("🧠 Identifying relevant data..."):
                    try:
                        relevant_files = identify_relevant_files(user_query, st.session_state.gemini_api_key)

                        if not relevant_files:
                            st.warning("No relevant datasets found for this question.")
                        else:
                            st.info(f"📁 Analyzing: {', '.join(relevant_files)}")

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

                            # ---------- Missing data ----------
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
                        print(data_context)
                        with st.spinner("🤔 Generating insights..."):
                            full_prompt = f"""{SYSTEM_PROMPT}
                            **Your answer should be VERY detailed**
                            Always consider cultural, regional, demographic, socio-economic, and platform-specific factors when analyzing IVF-related data, insights, or strategies. Additionally, factor in all relevant marketing dimensions, including audience segmentation, messaging sensitivity, channel effectiveness, regional ad behavior, conversion funnels, trust signals, compliance constraints, and competitive positioning.

                            Deliver a detailed, region-wise comparative analysis covering both cultural context and marketing performance, with clear rationale, trade-offs, and actionable implications. No shortcuts, no surface-level summaries—depth, rigor, and completeness are mandatory.
                                    DATA AVAILABLE:
                                    {data_context}
                                    USER QUESTION: {user_query}
                            Always colour code **numerical** values in tables.
                            When returning tables:
                            - Use HTML tables only
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
                            
                            st.markdown(response.text, unsafe_allow_html=True)

                            st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("If this is an API key error, please reconfigure your API key using the 'Change API Key' button above")

with tab3:
    st.markdown("## 📤 Upload Marketing Data")
    st.markdown("Upload CSV, Excel, or JSON files to extend your analytics")
    try:
        from utils import supabase_client
    except Exception:
        st.error("⚠️ Supabase is not configured. Please set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env file.")
        st.stop()

    else:
        st.markdown("<br>", unsafe_allow_html=True)
            
        # Field selection
        selected_upload_field = st.selectbox(
            "📌 Select Field for Upload",
            AVAILABLE_FIELDS,
            key="upload_field_selector"
        )

        st.markdown(f"**Uploading to:** `{selected_upload_field}`")
        st.markdown("<br>", unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['csv', 'xlsx', 'xls', 'json'],
            accept_multiple_files=True
        )
            
        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file(s) selected**")
                
            if st.button("🚀 Process & Upload", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                    
                total_files = len(uploaded_files)
                processed_count = 0
                success_count = 0
                    
                for idx, file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Processing {file.name}...")
                            
                            # Read file
                        sheets = read_uploaded_file(file)
                            
                        for sheet_name, df in sheets.items():
                                # Clean data
                            df_clean = clean_dataframe(df)
                                
                                # Extract schema
                            schema = extract_schema(df_clean)
                                
                                # Convert to CSV
                            csv_buffer = BytesIO()
                            df_clean.to_csv(csv_buffer, index=False)
                            csv_buffer.seek(0)
                                
                                # Generate unique file name
                            dataset_name = f"{file.name.split('.')[0]}_{sheet_name}" if len(sheets) > 1 else file.name.split('.')[0]
                            file_name = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            
                                # Upload to Supabase
                            try:
                                upload_to_supabase(SUPABASE_BUCKET, file_name, csv_buffer.getvalue())

                                metadata = {
                                    "dataset_name": dataset_name,
                                    "file_name": file_name,
                                    "bucket": SUPABASE_BUCKET,
                                    "schema": extract_schema(df_clean),
                                    "row_count": int(len(df_clean)),
                                    "uploaded_at": datetime.utcnow().isoformat(),
                                    "field": selected_upload_field
                                }

                                register_dataset(metadata)

                                st.session_state.active_datasets[dataset_name] = {
                                    'dataframe': df_clean,
                                    'field': selected_upload_field
                                }
                                success_count += 1

                            except Exception as e:
                                # 🔥 ROLLBACK storage if metadata fails
                                try:
                                    supabase_client.storage.from_(SUPABASE_BUCKET).remove([file_name])
                                except:
                                    pass

                                raise Exception(f"Upload failed and rolled back: {str(e)}")

                            
                        processed_count += 1
                        progress_bar.progress(processed_count / total_files)
                            
                    except Exception as e:
                        st.error(f"❌ Error processing {file.name}: {str(e)}")
                    
                status_text.text("✅ Upload complete!")
                st.success(f"Successfully processed {success_count} dataset(s)")
                    
                if success_count > 0:
                    st.info("🔄 Reload the page to see new datasets in the dashboard")
            
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### 📊 Registered Datasets")

        if st.session_state.active_datasets:

            # Make a COPY of keys to avoid mutation issues
            dataset_names = list(st.session_state.active_datasets.keys())

            for dataset_name in dataset_names:
                data = st.session_state.active_datasets[dataset_name]
                df = data['dataframe'] if isinstance(data, dict) else data
                field = data.get('field', 'Unknown') if isinstance(data, dict) else 'Unknown'

                with st.expander(f"🔍 {dataset_name} [{field}]", expanded=False):

                    # --- HEADER ROW ---
                    h1, h2, h3 = st.columns([4, 2, 1])

                    h1.markdown(f"**{dataset_name}**")
                    h2.markdown(f"**Rows:** {len(df):,}")

                    # 🔥 DELETE BUTTON (THIS WILL SHOW)
                    if h3.button("🗑 Delete", key=f"delete_{dataset_name}"):
                        try:
                            delete_dataset(dataset_name)
                            st.success(f"Deleted `{dataset_name}`")
                            st.session_state.pop("selected_dataset", None)
                            st.rerun()

                            if not st.session_state.active_datasets:
                                st.info("All datasets deleted. Upload new data to continue.")

                        except Exception as e:
                            st.error(str(e))

                    st.divider()

                    # --- DETAILS ---
                    st.markdown("**Columns:**")
                    st.code(", ".join(df.columns.tolist()))

                    st.markdown("**Preview (Top 10 rows)**")
                    st.dataframe(df.head(10), width='stretch')

        else:
            st.info("No custom datasets uploaded yet. Upload files above to get started.")

with tab4:
    st.markdown("## 🔍 AI Analytics")
    st.markdown("Ask natural language questions and get answers based on your data using intelligent query planning.")
    
    if not has_datasets():
        st.warning("📭 No datasets available")
        st.info("Upload at least one dataset to use the SQL Query Generator.")
    else:
        if not st.session_state.api_key_configured:
            st.warning("⚠️ GEMINI_API_KEY not found. Please configure API key.")
        else:
            # Initialize DuckDB connection
            if 'duckdb_conn' not in st.session_state:
                st.session_state.duckdb_conn = None
            
            # Load data into DuckDB
            with st.spinner("📊 Initializing DuckDB tables..."):
                try:
                    conn = initialize_duckdb_from_datasets()
                    st.session_state.duckdb_conn = conn
                        
                except Exception as e:
                    st.error(f"Error initializing DuckDB: {str(e)}")
                    st.stop()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # User input
            user_question = st.text_area(
                "Ask a question about your data",
                height=100 )
            show_visuals = st.checkbox(
                "📊 Generate visualizations (charts + insights)",
                value=True
            )

            generate_btn = st.button("🔍 Generate Answer", type="primary")
            
            if generate_btn and user_question:
                with st.spinner("🧠 Analyzing your question..."):
                    try:
                        # Extract enhanced schema
                        schema_info = extract_duckdb_schema(st.session_state.duckdb_conn)
                        
                        query_plan = create_query_plan(
                            user_question, 
                            schema_info,
                            st.session_state.gemini_api_key
                        )
                        sql_query = generate_sql_from_plan(
                            query_plan,
                            schema_info,
                            st.session_state.gemini_api_key
                        )
                        validated_query = validate_and_fix_query(
                            sql_query,
                            query_plan,
                            schema_info,
                            st.session_state.gemini_api_key
                        )
                        result_df, execution_error, final_query = execute_sql_with_retry(
                            st.session_state.duckdb_conn,
                            validated_query,
                            schema_info,
                            st.session_state.gemini_api_key,
                            max_retries=3
                        )
                        
                        if execution_error:
                            st.error(f"❌ Query Execution Failed: {execution_error}")
                            with st.expander("🔍 Failed SQL Query", expanded=True):
                                st.code(final_query, language="sql")
                            st.info("💡 Try rephrasing your question or check if the column/table names are correct.")
                            st.stop()
                        
                        # Step 6: Generate AI explanation
                        with st.spinner("💡 Generating intelligent explanation..."):
                            explanation = generate_result_explanation(
                                user_question,
                                result_df,
                                query_plan,
                                st.session_state.gemini_api_key
                            )
                        if show_visuals==True:
                            from utils import generate_ai_plot_from_result
                            # Generate AI plots + insights
                            with st.spinner("📊 Generating AI visualizations & insights..."):
                                ai_plot_response = generate_ai_plot_from_result(
                                    result_df,
                                    user_question,
                                    st.session_state.gemini_api_key
                                )
                                # Gemini image responses render automatically
                                for part in ai_plot_response.candidates[0].content.parts:
                                    if hasattr(part, "inline_data") and part.inline_data.mime_type.startswith("image"):
                                        st.image(part.inline_data.data)
                                    elif hasattr(part, "text"):
                                        st.markdown(part.text)
                        st.markdown(explanation)
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Error parsing query plan: {str(e)}")
                        st.info("The AI had trouble creating a structured plan. Try rephrasing your question.")
                        with st.expander("🔍 Debug Info"):
                            st.code(str(e))
                    
                    except Exception as e:
                        st.error(f"❌ Unexpected Error: {str(e)}")
                        st.info("Something went wrong. Please try again or rephrase your question.")
                        import traceback
                        with st.expander("🔍 Debug Info"):
                            st.code(traceback.format_exc())