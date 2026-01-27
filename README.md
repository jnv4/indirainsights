# Marketing Performance Insights Dashboard

A comprehensive AI-powered analytics platform designed for Indira IVF to analyze marketing performance data, generate intelligent reports, and provide actionable business insights through natural language queries.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Key Components](#key-components)
- [API Integration](#api-integration)
- [Data Management](#data-management)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

The Marketing Performance Insights Dashboard is a Streamlit-based web application that enables marketing teams to:

- **Upload and manage** marketing data from multiple sources (Call Center, Website, CRM, Competitors)
- **Visualize** consolidated insights by field with aggregated metrics
- **Generate AI-powered reports** combining internal data with web research
- **Query data naturally** using natural language to get SQL-generated insights
- **Export professional PDF reports** with visualizations and analysis

The platform leverages Google Gemini AI for intelligent query planning, SQL generation, and comprehensive report creation, making complex data analysis accessible through conversational interfaces.

## ✨ Features

### 1. **Dashboard Tab** 📈
- Field-based data organization (Call Center, Website, CRM, Competitors)
- Consolidated metrics across all datasets in each field
- Real-time aggregation of:
  - Total records
  - Top value and market share
  - Lowest performing values
  - CRM-specific conversion analysis
- Interactive dataset exploration with expandable previews

### 2. **AI Report Tab** 🤖
- Natural language question answering
- **Three data source modes**:
  - **Internal Data Only**: Analyze uploaded datasets
  - **Web Search Only**: Research current market trends
  - **Both**: Combine internal data with web intelligence
- Intelligent dataset relevance detection
- Comprehensive markdown reports with:
  - Sales performance overview
  - Regional growth analysis
  - Competitor comparisons
  - Economic and regulatory factors
  - Strategic recommendations
- Professional PDF export with citations
- Color-coded HTML tables for metric visualization

### 3. **Upload Data Tab** 📤
- Multi-format support: CSV, Excel (XLSX/XLS), JSON
- Automatic data cleaning and optimization:
  - Column name normalization
  - Duplicate removal
  - Data type optimization
  - Memory efficiency improvements
- Field-based categorization
- Supabase cloud storage integration
- Automatic compression for large files (>40MB)
- Dataset registration and metadata management
- Dataset deletion with rollback protection

### 4. **AI Analytics Tab** 🔍
- Natural language to SQL conversion
- **Intelligent Query Planning**:
  - Multi-query execution with dependencies
  - Automatic schema understanding
  - Query optimization and retry logic
- **DuckDB Integration**:
  - In-memory SQL database
  - Fast analytical queries
  - Automatic table registration from datasets
- **Optional Visualizations**:
  - AI-generated charts and graphs
  - Business-focused insights
  - Trend analysis
- **PDF Report Generation**:
  - Professional formatting
  - Embedded visualizations
  - Structured analysis sections

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Dashboard │  │AI Report │  │Upload    │  │Analytics │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer (app.py)                │
│  • Session state management                                 │
│  • UI orchestration                                         │
│  • User interaction handling                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Utility Layer (utils.py)                    │
│  • Data processing & cleaning                                │
│  • Supabase operations                                      │
│  • DuckDB management                                        │
│  • SQL generation & execution                               │
│  • PDF report generation                                    │
│  • Web search integration                                   │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Supabase    │  │  DuckDB      │  │  Gemini AI   │
│  Storage     │  │  In-Memory   │  │  API         │
│  & Metadata  │  │  SQL Engine  │  │  Processing  │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Data Flow

1. **Upload Flow**:
   ```
   User Upload → File Reading → Data Cleaning → 
   Supabase Storage → Metadata Registration → Session State Update
   ```

2. **Query Flow**:
   ```
   Natural Language Question → Schema Analysis → 
   Query Plan Generation → SQL Generation → 
   DuckDB Execution → Result Processing → 
   AI Explanation → Visualization (optional) → PDF Export
   ```

3. **Report Flow**:
   ```
   User Question → Dataset Relevance Detection → 
   Internal Data Loading → Web Search (optional) → 
   AI Analysis → Markdown Report → PDF Generation
   ```

## 🛠️ Technology Stack

### Core Framework
- **Streamlit**: Web application framework
- **Python 3.11+**: Programming language

### Data Processing
- **Pandas**: Data manipulation and analysis
- **DuckDB**: In-memory analytical SQL database
- **NumPy**: Numerical computing

### AI & Machine Learning
- **Google Generative AI (Gemini)**: 
  - `gemini-3-pro-preview`: Advanced report generation
  - `gemini-2.5-flash`: Fast query planning and SQL generation
  - `gemini-3-pro-image-preview`: Visualization generation
  - `gemini-2.0-flash-exp`: Web search with grounding

### Storage & Backend
- **Supabase**: 
  - Cloud storage for datasets
  - PostgreSQL metadata registry
  - Authentication and access control

### File Processing
- **openpyxl**: Excel file reading
- **json**: JSON parsing

### Report Generation
- **ReportLab**: PDF creation
- **BeautifulSoup**: HTML parsing for tables
- **Markdown**: Text formatting

### Utilities
- **python-dotenv**: Environment variable management
- **requests**: HTTP requests for web search

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Supabase account and project
- Google Gemini API key

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd indirainsights
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration

Create a `.env` file in the root directory:

```env
# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_role_key
```

### Step 5: Supabase Setup

1. **Create Storage Bucket**:
   - Go to Supabase Dashboard → Storage
   - Create a bucket named `marketing-cleaned`
   - Set bucket to public or configure appropriate policies

2. **Create Metadata Table**:
   ```sql
   CREATE TABLE dataset_registry (
       id SERIAL PRIMARY KEY,
       dataset_name TEXT UNIQUE NOT NULL,
       file_name TEXT NOT NULL,
       bucket TEXT NOT NULL,
       schema JSONB,
       row_count INTEGER,
       uploaded_at TIMESTAMP,
       field TEXT
   );
   ```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## ⚙️ Configuration

### Available Fields

The application supports four marketing fields defined in `utils.py`:

```python
AVAILABLE_FIELDS = ["Call Center", "Website", "CRM", "Competitors"]
```

To add new fields, modify this list in `utils.py`.

### Supabase Configuration

- **Bucket Name**: `marketing-cleaned` (configurable in `utils.py`)
- **Metadata Table**: `dataset_registry` (configurable in `utils.py`)
- **File Size Limits**: 
  - API limit: 50MB
  - Compression threshold: 40MB
  - Automatic gzip compression for large files

### AI Model Configuration

Models used (configurable in code):

- **Report Generation**: `gemini-3-pro-preview`
- **Query Planning**: `gemini-2.5-flash`
- **SQL Generation**: `gemini-2.5-flash`
- **Visualization**: `gemini-3-pro-image-preview`
- **Web Search**: `gemini-2.0-flash-exp`

## 📖 Usage

### Uploading Data

1. Navigate to the **Upload Data** tab
2. Select the target field (Call Center, Website, CRM, Competitors)
3. Click "Choose files" and select your data files (CSV, Excel, or JSON)
4. Click "🚀 Process & Upload"
5. Wait for processing and confirmation

**Supported Formats**:
- CSV files
- Excel files (.xlsx, .xls) - all sheets are processed
- JSON files (arrays or objects)

### Viewing Dashboard

1. Go to the **Dashboard** tab
2. Select a field from the sidebar
3. View consolidated metrics and individual datasets
4. Expand datasets to preview data

### Generating AI Reports

1. Navigate to the **AI Report** tab
2. Enter your question in the text area
3. Select data source:
   - **Internal Data Only**: Uses uploaded datasets
   - **Web Search Only**: Searches the internet
   - **Both**: Combines internal and web data
4. Click "🔍 Analyze"
5. Review the comprehensive report
6. Download PDF if needed

**Example Questions**:
- "What are the latest trends in IVF marketing?"
- "Analyze our Q4 performance compared to competitors"
- "What are the conversion rates by region?"

### Using AI Analytics

1. Go to the **AI Analytics** tab
2. Enter a natural language question about your data
3. (Optional) Check "Generate visualizations" for charts
4. Click "🔍 Generate Answer"
5. Review the analysis and results
6. Download PDF report if needed

**Example Questions**:
- "What is the total number of leads by source?"
- "Show me conversion rates by region"
- "Which campaigns have the highest ROI?"

## 📁 Project Structure

```
indirainsights/
│
├── app.py                 # Main Streamlit application
├── utils.py              # Utility functions and core logic
├── prompts.py            # AI system prompts
├── styles.css            # Custom CSS styling
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (not in git)
├── .gitattributes        # Git configuration
│
├── .devcontainer/        # Development container config
│   └── devcontainer.json
│
└── README.md            # This file
```

### Key Files Description

- **app.py**: Main application entry point with Streamlit UI, four main tabs, and user interaction handling
- **utils.py**: Core functionality including:
  - Data processing and cleaning
  - Supabase operations
  - DuckDB management
  - SQL generation and execution
  - PDF report generation
  - Web search integration
- **prompts.py**: System prompts for AI analysis, customized for Indira IVF marketing intelligence
- **styles.css**: Custom styling for enhanced UI/UX

## 🔄 Workflow

### Data Upload Workflow

```
1. User selects field and uploads file(s)
   ↓
2. File is read (CSV/Excel/JSON)
   ↓
3. Data is cleaned and optimized
   ↓
4. Schema is extracted and analyzed
   ↓
5. File is uploaded to Supabase storage
   ↓
6. Metadata is registered in database
   ↓
7. Dataset is loaded into session state
   ↓
8. User can now use dataset in other tabs
```

### AI Analytics Workflow

```
1. User enters natural language question
   ↓
2. DuckDB schema is extracted from loaded datasets
   ↓
3. AI creates multi-query execution plan
   ↓
4. For each query in plan:
   a. AI generates DuckDB SQL
   b. SQL is executed with retry logic
   c. Results are collected
   ↓
5. AI generates explanation from results
   ↓
6. (Optional) AI generates visualizations
   ↓
7. Results and explanation are displayed
   ↓
8. PDF report is generated on demand
```

### AI Report Workflow

```
1. User enters question and selects data source
   ↓
2. If internal data:
   a. AI identifies relevant datasets
   b. Data context is built with statistics
   ↓
3. If web search:
   a. Gemini performs web search with grounding
   b. Citations are extracted
   ↓
4. Combined prompt is sent to Gemini
   ↓
5. Comprehensive markdown report is generated
   ↓
6. Report is displayed with citations
   ↓
7. PDF is generated with formatted content
```

## 🔧 Key Components

### Data Processing (`utils.py`)

#### `clean_dataframe(df)`
- Normalizes column names
- Removes duplicates
- Optimizes data types for memory efficiency
- Handles missing values
- Converts low-cardinality columns to categories

#### `extract_schema(df)`
- Extracts column information
- Infers dataset role (fact/dimension/lookup)
- Identifies primary keys
- Detects time columns
- Infers business entities (Lead, Campaign, Call, etc.)
- Finds foreign key candidates

#### `upload_to_supabase(bucket, path, content)`
- Handles file compression for large files
- Uploads to Supabase storage
- Manages file size limits
- Provides error handling and rollback

### SQL Generation (`utils.py`)

#### `create_multi_query_plan(question, schema_info, api_key)`
- Analyzes user question
- Creates execution plan with dependencies
- Identifies required tables and columns
- Plans aggregations, filters, and joins

#### `generate_strict_sql_from_plan(plan, schema_info, api_key)`
- Generates DuckDB-compliant SQL
- Uses safe date parsing (TRY_STRPTIME)
- Handles dirty data gracefully
- Follows plan exactly without modifications

#### `execute_multi_query_plan(plan, conn, schema_info, api_key)`
- Executes queries respecting dependencies
- Handles failures gracefully
- Returns successful and failed queries
- Supports retry logic

### Report Generation (`utils.py`)

#### `create_markdown_pdf_report(query, response, citations)`
- Converts markdown to PDF
- Processes HTML tables with color coding
- Formats citations professionally
- Handles images and formatting

#### `create_pdf_report(question, explanation, result_df, plot_response)`
- Generates analytics PDF reports
- Embeds AI-generated visualizations
- Formats data tables
- Structures analysis sections

### Web Search (`utils.py`)

#### `perform_web_search(query, api_key)`
- Uses Gemini's search grounding
- Extracts top 5 authoritative sources
- Returns citations with metadata
- Handles rate limiting and retries
- Provides quality scoring

## 🔌 API Integration

### Google Gemini API

The application uses multiple Gemini models for different tasks:

1. **Query Planning**: `gemini-2.5-flash` (fast, cost-effective)
2. **SQL Generation**: `gemini-2.5-flash` (precise, syntax-focused)
3. **Report Generation**: `gemini-3-pro-preview` (comprehensive, detailed)
4. **Visualization**: `gemini-3-pro-image-preview` (image generation)
5. **Web Search**: `gemini-2.0-flash-exp` (search grounding)

**API Key Configuration**:
- Set `GEMINI_API_KEY` in `.env` file
- Or configure in Streamlit secrets for production

### Supabase API

**Storage Operations**:
- Upload files to `marketing-cleaned` bucket
- Download files for loading datasets
- Support for compressed files (.csv.gz)

**Database Operations**:
- Insert/update dataset metadata
- Query registered datasets
- Delete datasets with cascade

**Authentication**:
- Uses service role key for backend operations
- Configured via `SUPABASE_SERVICE_KEY`

## 💾 Data Management

### Dataset Lifecycle

1. **Registration**: Metadata stored in `dataset_registry` table
2. **Storage**: Files stored in Supabase storage bucket
3. **Loading**: Datasets loaded into session state on app start
4. **Querying**: Data registered as DuckDB tables for SQL queries
5. **Deletion**: Removes both storage file and metadata

### Data Optimization

- **Memory Efficiency**: 
  - Optimized data types (int8, int16, float32)
  - Category types for low-cardinality columns
  - Duplicate removal

- **Storage Efficiency**:
  - Automatic gzip compression for files >40MB
  - CSV format for compatibility
  - Metadata stored separately

### Schema Intelligence

The system automatically infers:
- **Dataset Role**: Fact table, dimension table, or lookup table
- **Primary Keys**: Based on uniqueness and naming patterns
- **Time Columns**: Date/timestamp detection
- **Business Entities**: Lead, Campaign, Call, Competitor, etc.
- **Foreign Keys**: Potential join columns

## 🐛 Troubleshooting

### Common Issues

#### 1. **Supabase Connection Error**
```
Error: SUPABASE_URL or SUPABASE_SERVICE_KEY missing
```
**Solution**: Ensure `.env` file contains valid Supabase credentials.

#### 2. **Gemini API Key Error**
```
Error: Invalid API key
```
**Solution**: Verify `GEMINI_API_KEY` in `.env` file and check API quota.

#### 3. **File Upload Fails**
```
Error: File too large
```
**Solution**: 
- Files are automatically compressed if >40MB
- Split large datasets into smaller files (<100K rows)
- Maximum size: 50MB after compression

#### 4. **SQL Query Errors**
```
Error: Query failed after 3 attempts
```
**Solution**:
- Check dataset schema matches query expectations
- Verify column names and data types
- Review error details in debug expander

#### 5. **Dataset Not Loading**
```
Warning: Could not load dataset
```
**Solution**:
- Check Supabase storage bucket permissions
- Verify file exists in storage
- Check network connectivity

### Debug Mode

Enable detailed error information:
- Check expandable debug sections in error messages
- Review browser console for client-side errors
- Check Streamlit logs for server-side errors

### Performance Optimization

1. **Large Datasets**:
   - Use data sampling for previews
   - Enable compression for uploads
   - Consider data partitioning

2. **Query Performance**:
   - DuckDB is optimized for analytical queries
   - Complex joins may be slow on very large datasets
   - Consider indexing strategies

3. **Memory Usage**:
   - Data types are optimized automatically
   - Large datasets are loaded on-demand
   - Session state is cleared on refresh

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style

- Follow PEP 8 Python style guide
- Use type hints where possible
- Add docstrings to functions
- Keep functions focused and modular

### Testing

Test the following scenarios:
- Data upload (all formats)
- Query generation and execution
- Report generation
- PDF export
- Error handling

## 👥 Authors

Janhavi Yadav - janhavi2004yadav@gmail.com

## 🙏 Acknowledgments

- Streamlit team for the excellent framework
- Google for Gemini AI capabilities
- Supabase for backend infrastructure
- DuckDB for fast analytical queries

---

**Note**: This application is designed specifically for Indira IVF's marketing analytics needs. Customize prompts and field configurations in `prompts.py` and `utils.py` for your organization.
