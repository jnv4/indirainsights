import streamlit as st
import pandas as pd
import requests
from io import StringIO
from dotenv import load_dotenv
import os
import duckdb
import google.generativeai as genai
import json

# Load environment variables
load_dotenv()

CSV_FILES = {
    "CRM Leads": "https://azxyptwlpqmvwfskitck.supabase.co/storage/v1/object/sign/indiraivf/crm_leads.csv?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV8yZjZlMTVhNC1iMzkxLTRjMTQtODU2MC0zNGExMTc3M2IzYzUiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJpbmRpcmFpdmYvY3JtX2xlYWRzLmNzdiIsImlhdCI6MTc2NjU5MjI0NiwiZXhwIjoxNzk4MTI4MjQ2fQ.amRpJ4_ri4wYOlyKCOPA5na3wbawY2lAmuvgEO7WeUY",
    "Region": "https://azxyptwlpqmvwfskitck.supabase.co/storage/v1/object/sign/indiraivf/Region.csv?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV8yZjZlMTVhNC1iMzkxLTRjMTQtODU2MC0zNGExMTc3M2IzYzUiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJpbmRpcmFpdmYvUmVnaW9uLmNzdiIsImlhdCI6MTc2NjU5MjI2NCwiZXhwIjoxNzk4MTI4MjY0fQ.4CczvVRMD4TdCEYf39XXCpiXMy8Kf6O10yh4Ldtw5gM",
    "City": "https://azxyptwlpqmvwfskitck.supabase.co/storage/v1/object/sign/indiraivf/city.csv?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV8yZjZlMTVhNC1iMzkxLTRjMTQtODU2MC0zNGExMTc3M2IzYzUiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJpbmRpcmFpdmYvY2l0eS5jc3YiLCJpYXQiOjE3NjY1OTIyODQsImV4cCI6MTc5ODEyODI4NH0.wFqqzb-ofqzKK_GwcIjGxOOgxwozx89X8HxbM_bnevI",
    "Channel": "https://azxyptwlpqmvwfskitck.supabase.co/storage/v1/object/sign/indiraivf/Channel.csv?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV8yZjZlMTVhNC1iMzkxLTRjMTQtODU2MC0zNGExMTc3M2IzYzUiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJpbmRpcmFpdmYvQ2hhbm5lbC5jc3YiLCJpYXQiOjE3NjY1OTIyOTgsImV4cCI6MTc5ODEyODI5OH0.bJNxsNcffMwmhPFeFYLWp--fxC3i28IoLOFRQcfeytI",
    "Competition": "https://azxyptwlpqmvwfskitck.supabase.co/storage/v1/object/sign/indiraivf/competition.csv?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV8yZjZlMTVhNC1iMzkxLTRjMTQtODU2MC0zNGExMTc3M2IzYzUiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJpbmRpcmFpdmYvY29tcGV0aXRpb24uY3N2IiwiaWF0IjoxNzY2NTkyMzE1LCJleHAiOjE3OTgxMjgzMTV9.2UnaQ0rscfywUBSOA-wkK32psgTDpyhLGJ42GyDLAnA",
    "Landing Page": "https://azxyptwlpqmvwfskitck.supabase.co/storage/v1/object/sign/indiraivf/landing%20page.csv?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV8yZjZlMTVhNC1iMzkxLTRjMTQtODU2MC0zNGExMTc3M2IzYzUiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJpbmRpcmFpdmYvbGFuZGluZyBwYWdlLmNzdiIsImlhdCI6MTc2NjU5MjMyNiwiZXhwIjoxNzk4MTI4MzI2fQ.VUFofCb6QDAurZquwtmh-5vAPiGVSQjfu9aVcf6El08",
    "Date": "https://azxyptwlpqmvwfskitck.supabase.co/storage/v1/object/sign/indiraivf/Date.csv?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV8yZjZlMTVhNC1iMzkxLTRjMTQtODU2MC0zNGExMTc3M2IzYzUiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJpbmRpcmFpdmYvRGF0ZS5jc3YiLCJpYXQiOjE3NjY1OTIzMzcsImV4cCI6MTc5ODEyODMzN30.Ldq8VVLtIKz5iWMLXK3yNVqUQJFTAyoyvhfAyeABrMs",
    "Hour": "https://azxyptwlpqmvwfskitck.supabase.co/storage/v1/object/sign/indiraivf/Hour.csv?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV8yZjZlMTVhNC1iMzkxLTRjMTQtODU2MC0zNGExMTc3M2IzYzUiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJpbmRpcmFpdmYvSG91ci5jc3YiLCJpYXQiOjE3NjY1OTIzNDUsImV4cCI6MTc5ODEyODM0NX0.W6S2Ce-hwL73NDCGZ2meMjWiVh3SHOPERU8NonolLAs",
    "Gender": "https://azxyptwlpqmvwfskitck.supabase.co/storage/v1/object/sign/indiraivf/Gender.csv?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV8yZjZlMTVhNC1iMzkxLTRjMTQtODU2MC0zNGExMTc3M2IzYzUiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJpbmRpcmFpdmYvR2VuZGVyLmNzdiIsImlhdCI6MTc2NjU5MjM1NCwiZXhwIjoxNzk4MTI4MzU0fQ.NUoT23Jv8-ceRX03Jw1iQ4f490j7ULPDG4kThQ7zUbo",
    "Device & Input": "https://azxyptwlpqmvwfskitck.supabase.co/storage/v1/object/sign/indiraivf/Device%20and%20Inpur.csv?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV8yZjZlMTVhNC1iMzkxLTRjMTQtODU2MC0zNGExMTc3M2IzYzUiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJpbmRpcmFpdmYvRGV2aWNlIGFuZCBJbnB1ci5jc3YiLCJpYXQiOjE3NjY1OTIzNjQsImV4cCI6MTc5ODEyODM2NH0.e-0JxFPuZRWOUIHxeY3DUIjyc95pXjdiYlC9asaS8eI",
    "Source Medium": "https://azxyptwlpqmvwfskitck.supabase.co/storage/v1/object/sign/indiraivf/Source%20Medium.csv?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV8yZjZlMTVhNC1iMzkxLTRjMTQtODU2MC0zNGExMTc3M2IzYzUiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJpbmRpcmFpdmYvU291cmNlIE1lZGl1bS5jc3YiLCJpYXQiOjE3NjY1OTI0MDQsImV4cCI6MTc5ODEyODQwNH0.D-kvk6VKII2QC_R4Hx1lhpliJg1It7ZKTnL923gBa8U",
    "Meta": "https://zknzcataoufenwdyhoyg.supabase.co/storage/v1/object/sign/Insights/meta.csv?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV9jYjBmYjA2Mi0zNTExLTQwZWMtYTExZi01NzEzNGViZWNlMDYiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJJbnNpZ2h0cy9tZXRhLmNzdiIsImlhdCI6MTc2NjY0ODg5OSwiZXhwIjoxNzk4MTg0ODk5fQ.6d987ItADVZkCzLJh70nNNYlErGY8HuooIl6U7-NfXI"
}

st.set_page_config(
    page_title="Marketing Insights Dashboard",
    layout="wide"
)

# Initialize session state with API key from .env
if 'gemini_api_key' not in st.session_state:
    env_api_key = os.getenv('GEMINI_API_KEY')
    st.session_state.gemini_api_key = env_api_key if env_api_key else None
    
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = bool(st.session_state.gemini_api_key)

st.markdown("""
            <style>

            /* Card base */
            div[data-testid="metric-container"] {
                background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
                border-radius: 18px;
                padding: 22px;
                border: 1px solid rgba(148, 163, 184, 0.18);

                /* REAL DEPTH */
                box-shadow:
                    0 12px 24px rgba(0, 0, 0, 0.55),
                    0 2px 6px rgba(0, 0, 0, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.08);

                transform: perspective(1000px) translateZ(0);
                transition: all 0.25s ease;
            }

            /* Hover lift */
            div[data-testid="metric-container"]:hover {
                transform: perspective(1000px) translateY(-6px) scale(1.02);
                box-shadow:
                    0 20px 36px rgba(0, 0, 0, 0.75),
                    0 6px 14px rgba(0, 0, 0, 0.6),
                    inset 0 1px 0 rgba(255, 255, 255, 0.12);
            }

            /* Metric label */
            div[data-testid="metric-container"] label {
                font-size: 13px;
                color: #c7d2fe;
                letter-spacing: 0.5px;
                text-transform: uppercase;
            }

            /* Metric value */
            div[data-testid="metric-container"] div {
                font-size: 30px;
                font-weight: 700;
                color: #ffffff;
            }

            /* Section spacing */
            section[data-testid="stHorizontalBlock"] {
                gap: 1.4rem;
            }

            /* Chatbot tab styling */
            .stTextArea textarea {
                background: linear-gradient(160deg, #1e293b 0%, #0f172a 100%) !important;
                border: 1px solid rgba(148, 163, 184, 0.3) !important;
                border-radius: 12px !important;
                color: #e2e8f0 !important;
                padding: 16px !important;
                font-size: 15px !important;
            }

            .stTextArea textarea::placeholder {
                color: #94a3b8 !important;
            }

            .stButton button {
                background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
                color: white !important;
                border: none !important;
                border-radius: 10px !important;
                padding: 12px 24px !important;
                font-weight: 600 !important;
                font-size: 16px !important;
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
                transition: all 0.3s ease !important;
            }

            .stButton button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 16px rgba(59, 130, 246, 0.6) !important;
            }

            /* Response styling */
            .response-container {
                background: linear-gradient(160deg, #1e293b 0%, #0f172a 100%);
                border-radius: 16px;
                padding: 32px;
                border: 1px solid rgba(148, 163, 184, 0.2);
                box-shadow: 
                    0 8px 20px rgba(0, 0, 0, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
                margin-top: 24px;
                color: #e2e8f0;
            }

            .response-container h1,
            .response-container h2,
            .response-container h3 {
                color: #f1f5f9;
            }

            .response-container table {
                color: #e2e8f0;
            }

            /* API Key Config Box */
            .api-config-box {
                background: linear-gradient(160deg, #1e293b 0%, #0f172a 100%);
                border-radius: 16px;
                padding: 24px;
                border: 1px solid rgba(148, 163, 184, 0.2);
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
                margin-bottom: 24px;
            }

            .success-box {
                background: linear-gradient(160deg, #064e3b 0%, #022c22 100%);
                border: 1px solid rgba(16, 185, 129, 0.3);
                border-radius: 12px;
                padding: 16px;
                color: #6ee7b7;
            }

            </style>
            """, unsafe_allow_html=True)

st.title("📊 Marketing Performance Insights")
st.caption("Aesthetic, insight-first dashboard for marketing decision makers")

@st.cache_data(show_spinner=False)
def load_csv(url: str) -> pd.DataFrame:
    r = requests.get(url)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

@st.cache_data(show_spinner=False)
def get_schema_info():
    schema_info = {}
    for name, url in CSV_FILES.items():
        try:
            # Load only first few rows to get schema
            df = load_csv(url)
            schema_info[name] = {
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "sample_values": {col: df[col].dropna().unique()[:5].tolist() for col in df.columns},
                "row_count": len(df)
            }
        except Exception as e:
            st.error(f"Error loading schema for {name}: {e}")
    return schema_info

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
        
        # Use fast model for intent classification
        intent_model = genai.GenerativeModel('gemini-2.5-flash')
        
        intent_prompt = f"""You are a data analyst. Based on the user's question, identify which datasets are needed to answer it.

                        {schema_description}

                        User question: {user_query}

                        Return ONLY a JSON array of dataset names that are needed. For example: ["CRM Leads", "Region", "Competition"]

                        Be selective - only include datasets that are directly relevant to answering the question."""
                                
        response = intent_model.generate_content(intent_prompt)
        
        # Parse the response to get the list of files
        response_text = response.text.strip()
        
        # Extract JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        relevant_files = json.loads(response_text)
        
        return relevant_files
        
    except Exception as e:
        st.warning(f"Error identifying relevant files: {str(e)}. Using all files as fallback.")
        return list(CSV_FILES.keys())

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
        # Try to list models as a validation check
        list(genai.list_models())
        return True
    except Exception as e:
        st.error(f"Invalid API key: {str(e)}")
        return False

tab1, tab2 = st.tabs(["📈 Dashboard", "🤖 Chatbot"])

with tab1:
    st.sidebar.title("📂 Insights By Field")
    selected_dataset = st.sidebar.radio(
        "Select a field",
        list(CSV_FILES.keys())
    )

    df = load_csv(CSV_FILES[selected_dataset])
    primary_col = df.columns[0]

    top_val, top_share, low_val, total_records = top_value_metrics(df[primary_col])

    st.markdown(f"## 🔍 {selected_dataset}")
    st.caption(f"Primary Dimension: `{primary_col}`")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{total_records:,}")
    c2.metric("Top Value", top_val)
    c3.metric("Top Share %", f"{top_share}%")
    c4.metric("Lowest Value", low_val)

    if selected_dataset == "CRM Leads" and "Converted" in df.columns:
        st.divider()
        st.markdown("### 🎯 Conversion Insights")

        total_leads = len(df)
        converted = df[df["Converted"].astype(str).str.lower() == "true"]
        conversion_rate = round((len(converted) / total_leads) * 100, 2)

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Total Leads", f"{total_leads:,}")
        cc2.metric("Converted Leads", f"{len(converted):,}")
        cc3.metric("Conversion Rate", f"{conversion_rate}%")

    with st.expander("🔎 Data Preview"):
        st.dataframe(
            df.head(100),width='stretch'
        )

with tab2:
    st.markdown("## 🤖 Sales & Market Intelligence Bot")
    st.markdown("Ask strategic questions about your marketing data")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="api-config-box">', unsafe_allow_html=True)
    st.markdown("### 🔑 API Configuration")
    
    if st.session_state.api_key_configured:
        st.markdown('<div class="success-box">✅ API Key configured successfully!</div>', unsafe_allow_html=True)
        
        # Show option to change API key
        if st.button("🔄 Change API Key"):
            st.session_state.api_key_configured = False
            st.session_state.gemini_api_key = None
            st.rerun()
    else:
        if st.session_state.gemini_api_key:
            # Auto-configure from .env
            st.session_state.api_key_configured = True
            st.rerun()
        else:
            api_key_input = st.text_input(
                "Enter your Gemini API Key",
                type="password"
            )
            if st.button("✅ Configure API Key", type="primary"):
                if validate_api_key(api_key_input):
                    st.session_state.gemini_api_key = api_key_input.strip()
                    st.session_state.api_key_configured = True
                    st.success("API Key configured successfully!")
                    st.rerun()
            else:
                st.warning("⚠️ GEMINI_API_KEY not found in .env file. Please enter manually.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.api_key_configured:
        # System prompt
        SYSTEM_PROMPT = f"""You are the Indira IVF Sales & Market Intelligence Bot—a world-class strategy oracle. 
      Your mission: Integrate internal performance data with external market intelligence to drive strategic growth.

      MARKET INTEL SPECIFICS:
      - MAHARASHTRA DOMINANCE: Always include [Progenesis IVF] as a primary competitor in Maharashtra.
      - NORTH INDIA: Focus on [Medicover], [ART Fertility], and [Nova IVF].
      - GUJARAT: Benchmark against [Sneh IVF] (Low cost/High volume) and [Wings IVF].
      - SOUTH INDIA: Analyze [GarbhaGudi] (Holistic) and [Milann].

      STRATEGIC FOCUS AREAS:
      1. TOP-END GROWTH: Increasing lead volume and brand dominance in key territories.
      2. BOTTOM-END GROWTH: Improving conversion velocity and operational sales efficiency.
      3. CUSTOMER EXPERIENCE: Leveraging USPs like Androlife (Oasis) or Premium Hospitality (Cloudnine) to improve patient satisfaction.

      ANALYSIS PROTOCOL:
      - Always query the [leads] and [competition] tables to find "Problems" (e.g., Conversion gaps).
      - Use Markdown Tables for all metric comparisons.
      - Every strategic suggestion MUST be preceded by a quantitative data "Insight".

      RESPONSE FORMAT (STRICT):
      1. SALES PERFORMANCE OVERVIEW: (Markdown table of internal metrics)
      2. REGIONAL GROWTH & CHALLENGES: (Deep dive into city/region problem areas)
      3. COMPETITOR MOVES: (Direct comparison with players like Progenesis, Nova, or Oasis)
      4. ECONOMIC & REGULATORY FACTORS: (Macro impacts)
      5. STRATEGIC RECOMMENDATIONS: (Actionable steps for Growth and Experience)
            """
            
        # User input
        user_query = st.text_area("Your Question", height=120, label_visibility="visible")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        analyze_btn = st.button("🔍 Analyze", type="primary", width='stretch')
        
        # Process query
        if analyze_btn and user_query:
            with st.spinner("🧠 Identifying relevant data..."):
                try:
                    # Step 1: Identify relevant files
                    relevant_files = identify_relevant_files(user_query, st.session_state.gemini_api_key)
                    
                    st.info(f"📁 Analyzing: {', '.join(relevant_files)}")
                    
                    # Step 2: Load only the relevant files
                    relevant_data = {}
                    for file_name in relevant_files:
                        if file_name in CSV_FILES:
                            with st.spinner(f"Loading {file_name}..."):
                                relevant_data[file_name] = load_csv(CSV_FILES[file_name])
                    
                    # Step 3: Prepare data context for only relevant files
                    data_context = ""
                    
                    for name, df in relevant_data.items():
                        table_name = name.lower().replace(" ", "_").replace("-", "_")
                        
                        # Get basic statistics
                        data_context += f"\n\n### Dataset: {name}\n"
                        data_context += f"Table name: {table_name}\n"
                        data_context += f"Total records: {len(df)}\n"
                        data_context += f"Columns: {', '.join(df.columns.tolist())}\n"
                        
                        # Add sample data (first 50 rows)
                        data_context += f"\nSample data (first 50 rows):\n"
                        data_context += df.head(50).to_markdown(index=False)
                        data_context += "\n"
                        
                        # Add column statistics for numeric columns
                        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                        if numeric_cols:
                            data_context += f"\nNumeric column statistics:\n"
                            data_context += df[numeric_cols].describe().to_markdown()
                            data_context += "\n"
                        
                        # Add value counts for categorical columns
                        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                        for col in categorical_cols[:5]:
                            value_counts = df[col].value_counts().head(10)
                            if len(value_counts) > 0:
                                data_context += f"\nTop values in '{col}':\n"
                                data_context += value_counts.to_markdown()
                                data_context += "\n"
                    
                    # Step 4: Create full prompt and generate response
                    with st.spinner("🤔 Generating insights..."):
                        full_prompt = f"""{SYSTEM_PROMPT}

DATA AVAILABLE:
{data_context}

USER QUESTION: {user_query}
"""
                        
                        # Configure Gemini with stored API key
                        genai.configure(api_key=st.session_state.gemini_api_key)
                        
                        # Initialize Gemini model (unchanged as per requirement)
                        model = genai.GenerativeModel('gemini-3-pro-preview')
                        
                        # Generate response
                        response = model.generate_content(full_prompt)
                        
                        # Display response
                        st.markdown('<div class="response-container">', unsafe_allow_html=True)
                        st.markdown(response.text)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("If this is an API key error, please reconfigure your API key using the 'Change API Key' button above")