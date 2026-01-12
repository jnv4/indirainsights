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

def create_pdf_report(user_question, explanation, result_df, ai_plot_response=None):
    """Generate a PDF report with markdown formatting and images"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter, 
        topMargin=0.5*inch, 
        bottomMargin=0.5*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch
    )
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles with word wrapping
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='#1f77b4',
        spaceAfter=12,
        alignment=TA_CENTER,
        wordWrap='CJK'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor='#2c3e50',
        spaceAfter=10,
        spaceBefore=10,
        wordWrap='CJK'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        wordWrap='CJK',
        spaceBefore=6,
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        leftIndent=20,
        wordWrap='CJK',
        spaceBefore=3,
        spaceAfter=3
    )
    
    # Title
    story.append(Paragraph("AI Analytics Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Question
    story.append(Paragraph("<b>Question:</b>", heading_style))
    story.append(Paragraph(user_question, normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Add visualizations if present
    if ai_plot_response:
        story.append(Paragraph("<b>Visualizations:</b>", heading_style))
        temp_files = []
        for part in ai_plot_response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data.mime_type.startswith("image"):
                try:
                    # Get the raw bytes data
                    img_data = part.inline_data.data
                    
                    # If it's a string, it might be base64 encoded
                    if isinstance(img_data, str):
                        img_data = base64.b64decode(img_data)
                    
                    # Save image to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png', mode='wb') as tmp_file:
                        tmp_file.write(img_data)
                        img_path = tmp_file.name
                        temp_files.append(img_path)
                    
                    # Verify the file was written correctly
                    if os.path.getsize(img_path) > 0:
                        try:
                            img = RLImage(img_path, width=6*inch, height=4*inch, kind='proportional')
                            story.append(img)
                            story.append(Spacer(1, 0.2*inch))
                        except Exception as img_error:
                            story.append(Paragraph(f"[Image could not be rendered]", normal_style))
                            story.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    story.append(Paragraph(f"[Error processing image]", normal_style))
                    story.append(Spacer(1, 0.2*inch))
        
        story.append(Spacer(1, 0.1*inch))
    
    # Explanation (convert markdown to PDF-friendly format)
    story.append(Paragraph("<b>Analysis:</b>", heading_style))
    
    # Process the explanation text line by line to preserve formatting
    try:
        # Split explanation into lines
        lines = explanation.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.05*inch))
                continue
            
            # Check if it's a header (starts with ###, ##, or #)
            if line.startswith('###'):
                text = line.replace('###', '').strip()
                story.append(Paragraph(f"<b>{text}</b>", styles['Heading3']))
            elif line.startswith('##'):
                text = line.replace('##', '').strip()
                story.append(Paragraph(f"<b>{text}</b>", heading_style))
            elif line.startswith('#'):
                text = line.replace('#', '').strip()
                story.append(Paragraph(f"<b>{text}</b>", heading_style))
            # Check if it's a bullet point
            elif line.startswith('*') or line.startswith('-') or line.startswith('•'):
                text = line.lstrip('*-• ').strip()
                story.append(Paragraph(f"• {text}", bullet_style))
            # Check if it contains bold text
            elif '**' in line:
                # Replace markdown bold with HTML bold
                text = line.replace('**', '<b>', 1).replace('**', '</b>', 1)
                story.append(Paragraph(text, normal_style))
            else:
                story.append(Paragraph(line, normal_style))
    
    except Exception as e:
        # Fallback to raw text if processing fails
        story.append(Paragraph(explanation, normal_style))
    
    # Build PDF
    try:
        doc.build(story)
    except Exception as e:
        # If PDF build fails, clean up temp files and re-raise
        if ai_plot_response and 'temp_files' in locals():
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
        raise e
    
    # Clean up temporary image files
    if ai_plot_response and 'temp_files' in locals():
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
    
    buffer.seek(0)
    return buffer

def create_markdown_pdf_report(user_query, markdown_response, citations=None):
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch,
        leftMargin=0.5*inch,
        rightMargin=0.5*inch
    )
    story = []
    styles = getSampleStyleSheet()
    
    # Define custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        leading=20
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=10,
        spaceBefore=14,
        fontName='Helvetica-Bold',
        leading=17
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=8,
        spaceBefore=10,
        fontName='Helvetica-Bold',
        leading=15
    )
    
    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontSize=10,
        textColor=colors.HexColor('#465668'),
        spaceAfter=6,
        spaceBefore=8,
        fontName='Helvetica-Bold',
        leading=13
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        alignment=TA_LEFT,
        spaceBefore=4,
        spaceAfter=4
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        leftIndent=20,
        firstLineIndent=-10,
        spaceBefore=2,
        spaceAfter=2
    )
    
    cell_style = ParagraphStyle(
        'CellStyle',
        parent=styles['Normal'],
        fontSize=7,
        leading=9,
        alignment=TA_LEFT
    )
    
    def clean_markdown_text(text):
        """
        Clean and convert markdown to HTML tags for ReportLab
        Handles: **, *, italic, etc.
        """
        if not text:
            return ""
        
        text = text.strip()
        
        # Handle bold (**text**)
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        
        # Handle italic (*text* or _text_)
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
        text = re.sub(r'_(.+?)_', r'<i>\1</i>', text)
        
        # Handle inline code (`code`)
        text = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', text)
        
        return text
    
    def sanitize_for_paragraph(text):
        """
        Sanitize text for ReportLab Paragraph with proper escaping
        """
        if not text:
            return ""
        
        # First, clean markdown
        text = clean_markdown_text(text)
        
        # Extract HTML tags we want to preserve
        tags_to_preserve = []
        tag_pattern = r'(</?(?:b|i|u|font)[^>]*>)'
        parts = re.split(tag_pattern, text)
        
        result = []
        for part in parts:
            if re.match(tag_pattern, part):
                # This is a tag we want to preserve
                tags_to_preserve.append(part)
                result.append(f"__TAG_{len(tags_to_preserve)-1}__")
            else:
                # Escape this part
                result.append(escape(part))
        
        # Reconstruct with preserved tags
        text = ''.join(result)
        for i, tag in enumerate(tags_to_preserve):
            text = text.replace(f"__TAG_{i}__", tag)
        
        return text
    
    def safe_paragraph(text, style):
        """Create a paragraph with comprehensive error handling"""
        try:
            cleaned = sanitize_for_paragraph(text)
            return Paragraph(cleaned, style)
        except Exception as e:
            # Fallback: create plain text paragraph
            try:
                plain_text = re.sub(r'<[^>]*>', '', text)
                return Paragraph(escape(plain_text), style)
            except:
                # Ultimate fallback
                return Paragraph("(Content could not be rendered)", style)
    
    def parse_color(color_str):
        """Parse color from various formats to ReportLab color"""
        if not color_str:
            return None
        
        try:
            color_str = color_str.strip().lower()
            
            # Hex color
            if color_str.startswith('#'):
                return colors.HexColor(color_str)
            
            # RGB format: rgb(r, g, b)
            rgb_match = re.match(r'rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', color_str)
            if rgb_match:
                r, g, b = map(int, rgb_match.groups())
                return colors.Color(r/255.0, g/255.0, b/255.0)
            
            # Named colors
            color_map = {
                'red': colors.red,
                'green': colors.green,
                'blue': colors.blue,
                'yellow': colors.yellow,
                'orange': colors.orange,
                'gray': colors.gray,
                'grey': colors.grey,
                'white': colors.white,
                'black': colors.black,
            }
            if color_str in color_map:
                return color_map[color_str]
            
        except Exception:
            pass
        
        return None
    
    def extract_cell_color(cell):
        """Extract background color from cell style attribute"""
        style_attr = cell.get('style', '')
        
        # Look for background-color or background
        bg_patterns = [
            r'background-color\s*:\s*([^;]+)',
            r'background\s*:\s*([^;]+)',
        ]
        
        for pattern in bg_patterns:
            match = re.search(pattern, style_attr, re.IGNORECASE)
            if match:
                color_value = match.group(1).strip()
                color = parse_color(color_value)
                if color:
                    return color
        
        return None
    
    def process_table(table_html):
        """Process HTML table and convert to ReportLab Table with colors"""
        try:
            soup = BeautifulSoup(table_html, 'html.parser')
            table_elem = soup.find('table')
            
            if not table_elem:
                return None
            
            table_data = []
            cell_colors = []
            
            # Process all rows
            for tr in table_elem.find_all('tr'):
                row_data = []
                row_colors = []
                
                for cell in tr.find_all(['th', 'td']):
                    # Get cell text and clean it
                    cell_text = cell.get_text().strip()
                    cell_para = safe_paragraph(cell_text, cell_style)
                    row_data.append(cell_para)
                    
                    # Extract color
                    color = extract_cell_color(cell)
                    row_colors.append(color)
                
                if row_data:
                    table_data.append(row_data)
                    cell_colors.append(row_colors)
            
            if not table_data:
                return None
            
            # Calculate column widths
            num_cols = len(table_data[0])
            available_width = 7 * inch
            col_widths = [available_width / num_cols] * num_cols
            
            # Create table
            pdf_table = Table(table_data, colWidths=col_widths, repeatRows=1)
            
            # Base style commands
            style_commands = [
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ]
            
            # Apply cell-specific colors
            for row_idx, row_colors in enumerate(cell_colors):
                for col_idx, color in enumerate(row_colors):
                    if color:
                        style_commands.append(
                            ('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), color)
                        )
            
            # Default header background if no colors specified
            if all(c is None for c in cell_colors[0]):
                style_commands.append(
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f2f2f2'))
                )
            
            pdf_table.setStyle(TableStyle(style_commands))
            return pdf_table
            
        except Exception as e:
            print(f"Error processing table: {e}")
            return None
    
    def process_markdown_content(content):
        """Process markdown content and extract tables"""
        # Find all tables
        table_pattern = r'<table[^>]*>.*?</table>'
        tables = list(re.finditer(table_pattern, content, flags=re.DOTALL | re.IGNORECASE))
        
        segments = []
        current_pos = 0
        
        for match in tables:
            # Text before table
            text_before = content[current_pos:match.start()]
            if text_before.strip():
                segments.append(('text', text_before))
            
            # Table
            segments.append(('table', match.group(0)))
            current_pos = match.end()
        
        # Remaining text
        if current_pos < len(content):
            remaining = content[current_pos:]
            if remaining.strip():
                segments.append(('text', remaining))
        
        return segments
    
    def add_text_block(text):
        """Add text block with proper markdown parsing"""
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Heading level 1 (#)
            if line.startswith('# ') and not line.startswith('##'):
                text = line[2:].strip()
                story.append(safe_paragraph(text, heading1_style))
            
            # Heading level 2 (##)
            elif line.startswith('## ') and not line.startswith('###'):
                text = line[3:].strip()
                story.append(safe_paragraph(text, heading2_style))
            
            # Heading level 3 (###)
            elif line.startswith('### '):
                text = line[4:].strip()
                story.append(safe_paragraph(text, heading3_style))
            
            # Bullet points
            elif line.startswith(('* ', '- ', '• ')):
                text = line.lstrip('*-• ').strip()
                story.append(safe_paragraph(f"• {text}", bullet_style))
            
            # Numbered lists
            elif re.match(r'^\d+\.\s', line):
                text = re.sub(r'^\d+\.\s*', '', line)
                story.append(safe_paragraph(f"  {text}", bullet_style))
            
            # Regular paragraph
            else:
                story.append(safe_paragraph(line, normal_style))
    
    # Build the PDF
    try:
        # Title
        story.append(safe_paragraph("Sales & Market Intelligence Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Question section
        story.append(safe_paragraph("<b>Question:</b>", heading2_style))
        story.append(safe_paragraph(user_query, normal_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Analysis section
        story.append(safe_paragraph("<b>Analysis:</b>", heading2_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Process content
        segments = process_markdown_content(markdown_response)
        
        for seg_type, seg_content in segments:
            if seg_type == 'text':
                add_text_block(seg_content)
            elif seg_type == 'table':
                table = process_table(seg_content)
                if table:
                    story.append(Spacer(1, 0.1*inch))
                    story.append(table)
                    story.append(Spacer(1, 0.15*inch))
        
        # Build PDF
        # Add citations section if provided
        if citations and len(citations) > 0:
            story.append(safe_paragraph("<b>Sources & Citations</b>", heading1_style))
            story.append(Spacer(1, 0.10*inch))
            
            for idx, citation in enumerate(citations, 1):
                # Citation number
                story.append(safe_paragraph(f"<b>[{idx}]</b>", heading3_style))
                
                # Citation title
                story.append(safe_paragraph(citation.get('title', 'Untitled'), normal_style))
                
                # Citation URL
                url = citation.get('uri', '')
                story.append(safe_paragraph(f"<font color='#1f77b4'>{url}</font>", normal_style))
                
                # Citation snippet
                if citation.get('snippet'):
                    story.append(safe_paragraph(f"<i>{citation['snippet']}</i>", bullet_style))
                
                story.append(Spacer(1, 0.07*inch))
        
        # Build PDF
        doc.build(story)
        
    except Exception as e:
        # Fallback: create simple error report
        print(f"Error building PDF: {e}")
        story = [
            Paragraph("Sales & Market Intelligence Report", title_style),
            Spacer(1, 0.2*inch),
            Paragraph(f"<b>Question:</b> {escape(user_query)}", normal_style),
            Spacer(1, 0.2*inch),
            Paragraph("<b>Analysis:</b>", heading2_style),
            Spacer(1, 0.1*inch),
            Paragraph("An error occurred while generating the full report. Please try again or view the analysis in the web interface.", normal_style),
            Spacer(1, 0.1*inch),
            Paragraph(f"<i>Error details: {escape(str(e))}</i>", normal_style)
        ]
        doc.build(story)
    
    buffer.seek(0)
    return buffer

def perform_web_search(query: str, api_key: str, max_retries: int = 3) -> dict:
    """
    Perform highly accurate web search using Google Gemini with search grounding.
    
    Args:
        query: Search query string
        api_key: Google API key
        max_retries: Number of retry attempts for failed requests
    
    Returns:
        dict with success, error, text, citations, and metadata
    """
    import requests
    import time
    from datetime import datetime
    
    try:
        # Validate inputs
        if not query or not query.strip():
            return {
                "success": False,
                "error": "Query cannot be empty",
                "text": "",
                "citations": [],
                "metadata": {}
            }
        
        if not api_key or not api_key.strip():
            return {
                "success": False,
                "error": "API key is required",
                "text": "",
                "citations": [],
                "metadata": {}
            }
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"
        
        # Enhanced search query focused on top sources
        search_query = f"""Conduct a focused web search using ONLY the top 5 most authoritative and relevant sources about: {query}

        CRITICAL REQUIREMENTS:
        1. Use ONLY the 5 most credible, recent, and directly relevant sources
        2. Prioritize: official sources, peer-reviewed publications, authoritative news outlets, government data, industry leaders
        3. Cross-reference key facts across these top sources
        4. Include specific dates, numbers, and data points with source attribution
        5. Note any conflicts between sources
        6. Synthesize information efficiently - focus on signal, not noise

        STRUCTURE YOUR RESPONSE:
        • Executive Summary: 2-3 sentences of key verified findings
        • Core Facts: Most important verified data points with dates and sources
        • Key Insights: Critical trends or developments supported by evidence
        • Main Players: Leading organizations/entities (if relevant)
        • Notable Challenges: Verified issues or concerns
        • Latest Developments: Most recent updates (with specific dates)
        • Source Quality: Brief note on the authority/credibility of sources used

        ACCURACY STANDARDS:
        - Every major claim must reference its source
        - Include specific dates for time-sensitive information
        - Use exact numbers with context and units
        - Explicitly state "according to [source]..." for attribution
        - Note information recency (e.g., "as of January 2025...")
        - Focus on the most important, actionable information"""
        
        payload = {
            "contents": [{
                "parts": [{"text": search_query}]
            }],
            "tools": [{
                "google_search": {}
            }],
            "generationConfig": {
                "temperature": 0.1,  # Lower temperature for more factual responses
                "topP": 0.8,
                "topK": 20,
                "maxOutputTokens": 8192,
                "candidateCount": 1
            }
        }
        
        # Retry logic for robustness
        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url, 
                    json=payload,
                    timeout=30,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    break
                elif response.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2  # Exponential backoff
                        time.sleep(wait_time)
                        continue
                    last_error = f"Rate limit exceeded after {max_retries} attempts"
                elif response.status_code >= 500:  # Server error
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    last_error = f"Server error: {response.status_code}"
                else:
                    last_error = f"API error {response.status_code}: {response.text}"
                    break
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    continue
                last_error = "Request timeout after multiple attempts"
            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {str(e)}"
                break
        
        if response.status_code != 200 or last_error:
            return {
                "success": False,
                "error": last_error or f"Unknown error (status: {response.status_code})",
                "text": "",
                "citations": [],
                "metadata": {"attempts": attempt + 1}
            }
        
        data = response.json()
        
        # Enhanced extraction with validation
        result_text = ""
        citations = []
        search_queries_used = []
        grounding_support_score = 0.0
        
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            
            # Extract text content with validation
            if "content" in candidate and "parts" in candidate["content"]:
                text_parts = []
                for part in candidate["content"]["parts"]:
                    if "text" in part and part["text"].strip():
                        text_parts.append(part["text"].strip())
                result_text = "\n\n".join(text_parts)
            
            # Extract comprehensive grounding metadata
            if "groundingMetadata" in candidate:
                grounding = candidate["groundingMetadata"]
                
                # Get grounding support score (confidence measure)
                if "groundingSupport" in grounding:
                    support = grounding["groundingSupport"]
                    if isinstance(support, (int, float)):
                        grounding_support_score = float(support)
                
                # Extract search queries that were executed
                if "searchQueries" in grounding:
                    search_queries_used = grounding["searchQueries"]
                
                # Extract top 5 most relevant grounding chunks (citations)
                seen_uris = set()
                if "groundingChunks" in grounding:
                    for chunk in grounding["groundingChunks"]:
                        # Stop after collecting 5 unique sources
                        if len(citations) >= 5:
                            break
                            
                        if "web" in chunk:
                            web_info = chunk["web"]
                            uri = web_info.get("uri", "")
                            
                            # Deduplicate by URI
                            if uri and uri not in seen_uris:
                                seen_uris.add(uri)
                                
                                citation = {
                                    "title": web_info.get("title", "Untitled").strip(),
                                    "uri": uri,
                                    "snippet": web_info.get("snippet", "")[:300].strip()
                                }
                                
                                # Extract domain for credibility assessment
                                try:
                                    from urllib.parse import urlparse
                                    domain = urlparse(uri).netloc
                                    citation["domain"] = domain
                                except:
                                    citation["domain"] = "unknown"
                                
                                citations.append(citation)
                
                # Sort citations by relevance (if available) or keep order
                if "groundingSupports" in grounding:
                    # Citations are already in relevance order from API
                    pass
        
        # Validate results
        if not result_text and not citations:
            return {
                "success": False,
                "error": "No search results or citations found. Query may be too specific or no recent information available.",
                "text": "",
                "citations": [],
                "metadata": {
                    "attempts": attempt + 1,
                    "search_queries_used": search_queries_used
                }
            }
        
        # Quality scoring
        quality_score = 0.0
        if result_text:
            quality_score += 0.4
        if citations:
            quality_score += 0.3 * min(len(citations) / 5, 1.0)  # Up to 5 citations
        if grounding_support_score > 0:
            quality_score += 0.3 * grounding_support_score
        
        return {
            "success": True,
            "error": None,
            "text": result_text,
            "citations": citations,
            "metadata": {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "citation_count": len(citations),
                "grounding_support_score": grounding_support_score,
                "quality_score": round(quality_score, 2),
                "search_queries_used": search_queries_used,
                "attempts": attempt + 1
            }
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "text": "",
            "citations": [],
            "metadata": {
                "traceback": traceback.format_exc(),
                "error_type": type(e).__name__
            }
        }
