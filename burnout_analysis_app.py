"""
Student Burnout Risk Analysis - Enhanced Version
Features: CSV Upload, Database, Counselor Dashboard, PDF Reports
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Student Burnout Analysis - Enhanced",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Unique Professional Design
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        padding: 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Header Styles */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        letter-spacing: -0.5px;
        padding: 0.5rem 0;
    }
    
    h2 {
        color: #2d3748;
        font-weight: 600;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
        padding-left: 1rem;
        border-left: 4px solid #667eea;
    }
    
    h3 {
        color: #4a5568;
        font-weight: 500;
        margin-top: 1.5rem;
    }
    
    /* Card Styles */
    .stMetric {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.15);
    }
    
    .stMetric label {
        color: #718096 !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #2d3748 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Slider Styles */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Input Styles */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Alert Styles */
    .stAlert {
        border-radius: 12px;
        border: none;
        padding: 1.25rem;
        font-weight: 500;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #d4f4dd 0%, #b8f1cc 100%);
        color: #22543d;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef5e7 0%, #fdeaa8 100%);
        color: #744210;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee 0%, #fdd 100%);
        color: #742a2a;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #e6f7ff 0%, #bae7ff 100%);
        color: #003a8c;
    }
    
    /* Dataframe Styles */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Expander Styles */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        font-weight: 600;
        color: #2d3748;
        padding: 1rem 1.5rem;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea;
        background: #f7fafc;
    }
    
    /* Sidebar Styles */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white;
        -webkit-text-fill-color: white;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: white;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 0.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #718096;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Custom Metric Cards */
    .metric-card {
        background: white;
        padding: 1.75rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.15);
    }
    
    .metric-card h3 {
        color: #718096;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 0.5rem 0;
    }
    
    .metric-card h1 {
        color: #2d3748;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        background: none;
        -webkit-text-fill-color: #2d3748;
    }
    
    .metric-card.risk-high {
        border-left: 5px solid #f56565;
        background: linear-gradient(135deg, #fff5f5 0%, #ffffff 100%);
    }
    
    .metric-card.risk-high h1 {
        color: #c53030;
        -webkit-text-fill-color: #c53030;
    }
    
    .metric-card.risk-medium {
        border-left: 5px solid #ed8936;
        background: linear-gradient(135deg, #fffaf0 0%, #ffffff 100%);
    }
    
    .metric-card.risk-medium h1 {
        color: #c05621;
        -webkit-text-fill-color: #c05621;
    }
    
    .metric-card.risk-low {
        border-left: 5px solid #48bb78;
        background: linear-gradient(135deg, #f0fff4 0%, #ffffff 100%);
    }
    
    .metric-card.risk-low h1 {
        color: #2f855a;
        -webkit-text-fill-color: #2f855a;
    }
    
    /* Section Divider */
    hr {
        margin: 2.5rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e0, transparent);
    }
    
    /* Download Button Special */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        box-shadow: 0 4px 12px rgba(72, 187, 120, 0.3);
    }
    
    .stDownloadButton > button:hover {
        box-shadow: 0 8px 20px rgba(72, 187, 120, 0.4);
    }
    
    /* Checkbox Styles */
    .stCheckbox {
        padding: 0.5rem 0;
    }
    
    .stCheckbox > label {
        font-weight: 500;
        color: #2d3748;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #cbd5e0;
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: #f7fafc;
    }
    
    /* Plotly Charts */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Badge Styles */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-success {
        background: #c6f6d5;
        color: #22543d;
    }
    
    .badge-warning {
        background: #feebc8;
        color: #744210;
    }
    
    .badge-danger {
        background: #fed7d7;
        color: #742a2a;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== DATABASE FUNCTIONS ====================
def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('burnout_analysis.db')
    c = conn.cursor()
    
    # Students table
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  student_id TEXT UNIQUE,
                  name TEXT,
                  email TEXT,
                  department TEXT,
                  year INTEGER,
                  created_at TEXT)''')
    
    # Assessments table
    c.execute('''CREATE TABLE IF NOT EXISTS assessments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  student_id TEXT,
                  assessment_date TEXT,
                  sleep_duration REAL,
                  study_hours REAL,
                  screen_time REAL,
                  stress_levels REAL,
                  physical_activity REAL,
                  social_interaction REAL,
                  risk_score INTEGER,
                  burnout_probability REAL,
                  risk_category TEXT,
                  FOREIGN KEY (student_id) REFERENCES students(student_id))''')
    
    # Interventions table
    c.execute('''CREATE TABLE IF NOT EXISTS interventions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  student_id TEXT,
                  intervention_date TEXT,
                  intervention_type TEXT,
                  notes TEXT,
                  counselor_name TEXT,
                  FOREIGN KEY (student_id) REFERENCES students(student_id))''')
    
    conn.commit()
    conn.close()

def save_student(student_id, name, email, department, year):
    """Save student to database"""
    conn = sqlite3.connect('burnout_analysis.db')
    c = conn.cursor()
    try:
        c.execute('''INSERT OR REPLACE INTO students 
                     (student_id, name, email, department, year, created_at)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (student_id, name, email, department, year, datetime.now().isoformat()))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving student: {e}")
        return False
    finally:
        conn.close()

def save_assessment(student_id, data, risk_score, probability, risk_category):
    """Save assessment to database"""
    conn = sqlite3.connect('burnout_analysis.db')
    c = conn.cursor()
    try:
        c.execute('''INSERT INTO assessments 
                     (student_id, assessment_date, sleep_duration, study_hours, 
                      screen_time, stress_levels, physical_activity, social_interaction,
                      risk_score, burnout_probability, risk_category)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (student_id, datetime.now().isoformat(), 
                   data['sleep_duration'], data['study_hours'], data['screen_time'],
                   data['stress_levels'], data['physical_activity'], data['social_interaction'],
                   risk_score, probability, risk_category))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving assessment: {e}")
        return False
    finally:
        conn.close()

def get_all_students():
    """Get all students from database"""
    conn = sqlite3.connect('burnout_analysis.db')
    df = pd.read_sql_query("SELECT * FROM students", conn)
    conn.close()
    return df

def get_student_assessments(student_id):
    """Get all assessments for a student"""
    conn = sqlite3.connect('burnout_analysis.db')
    df = pd.read_sql_query(
        "SELECT * FROM assessments WHERE student_id = ? ORDER BY assessment_date DESC",
        conn, params=(student_id,))
    conn.close()
    return df

def get_all_assessments():
    """Get all assessments"""
    conn = sqlite3.connect('burnout_analysis.db')
    df = pd.read_sql_query("SELECT * FROM assessments ORDER BY assessment_date DESC", conn)
    conn.close()
    return df

def save_intervention(student_id, intervention_type, notes, counselor_name):
    """Save intervention to database"""
    conn = sqlite3.connect('burnout_analysis.db')
    c = conn.cursor()
    c.execute('''INSERT INTO interventions 
                 (student_id, intervention_date, intervention_type, notes, counselor_name)
                 VALUES (?, ?, ?, ?, ?)''',
              (student_id, datetime.now().isoformat(), intervention_type, notes, counselor_name))
    conn.commit()
    conn.close()

# ==================== CSV UPLOAD FUNCTIONS ====================
def validate_csv(df):
    """Validate uploaded CSV has required columns"""
    required_columns = ['sleep_duration', 'study_hours', 'screen_time', 
                       'stress_levels', 'physical_activity', 'social_interaction']
    
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    
    # Check data types
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column '{col}' must be numeric"
    
    return True, "Valid"

def load_csv_data(uploaded_file):
    """Load and validate CSV data"""
    try:
        df = pd.read_csv(uploaded_file)
        is_valid, message = validate_csv(df)
        
        if not is_valid:
            st.error(f"❌ CSV Validation Error: {message}")
            return None
        
        # Add burnout classification if not present
        if 'burnout' not in df.columns:
            # Calculate burnout score
            df['burnout_score'] = (
                (8 - df['sleep_duration']) * 0.15 +
                (df['study_hours'] - 3) * 0.1 +
                (df['screen_time'] - 5) * 0.08 +
                (df['stress_levels'] - 5) * 0.2 +
                (5 - df['physical_activity']) * 0.05 +
                (5 - df['social_interaction']) * 0.05
            )
            df['burnout_score'] = (df['burnout_score'] - df['burnout_score'].min()) / \
                                 (df['burnout_score'].max() - df['burnout_score'].min())
            df['burnout'] = (df['burnout_score'] > 0.6).astype(int)
        
        st.success(f"✅ Successfully loaded {len(df)} student records!")
        return df
        
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

# ==================== PDF REPORT GENERATION ====================
def generate_pdf_report(student_data, risk_score, probability, risk_category, student_info=None):
    """Generate PDF assessment report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("Student Burnout Risk Assessment Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Student Info
    if student_info:
        story.append(Paragraph("Student Information", heading_style))
        info_data = [
            ['Student ID:', student_info.get('student_id', 'N/A')],
            ['Name:', student_info.get('name', 'N/A')],
            ['Department:', student_info.get('department', 'N/A')],
            ['Assessment Date:', datetime.now().strftime('%B %d, %Y')]
        ]
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Risk Summary
    story.append(Paragraph("Risk Assessment Summary", heading_style))
    
    # Risk color
    risk_color = colors.red if probability > 0.7 else colors.orange if probability > 0.4 else colors.green
    
    summary_data = [
        ['Risk Score:', f'{risk_score}/16'],
        ['Risk Category:', risk_category],
        ['Burnout Probability:', f'{probability*100:.1f}%']
    ]
    summary_table = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (1, 2), (1, 2), risk_color),
        ('TEXTCOLOR', (1, 2), (1, 2), colors.white),
        ('FONTNAME', (1, 2), (1, 2), 'Helvetica-Bold'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Lifestyle Factors
    story.append(Paragraph("Lifestyle Factors Analysis", heading_style))
    
    lifestyle_data = [
        ['Factor', 'Current Value', 'Recommended Range', 'Status'],
        ['Sleep Duration', f'{student_data["sleep_duration"]:.1f} hrs', '7-9 hrs', 
         '✓' if student_data["sleep_duration"] >= 7 else '✗'],
        ['Study Hours', f'{student_data["study_hours"]:.1f} hrs', '3-6 hrs',
         '✓' if 3 <= student_data["study_hours"] <= 6 else '✗'],
        ['Screen Time', f'{student_data["screen_time"]:.1f} hrs', '3-6 hrs',
         '✓' if 3 <= student_data["screen_time"] <= 6 else '✗'],
        ['Stress Level', f'{student_data["stress_levels"]:.1f}/10', '1-5',
         '✓' if student_data["stress_levels"] <= 5 else '✗'],
        ['Physical Activity', f'{student_data["physical_activity"]:.1f} hrs/wk', '3+ hrs',
         '✓' if student_data["physical_activity"] >= 3 else '✗'],
        ['Social Interaction', f'{student_data["social_interaction"]:.1f} hrs', '3+ hrs',
         '✓' if student_data["social_interaction"] >= 3 else '✗'],
    ]
    
    lifestyle_table = Table(lifestyle_data, colWidths=[2*inch, 1.5*inch, 1.8*inch, 0.7*inch])
    lifestyle_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    story.append(lifestyle_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Recommendations
    story.append(Paragraph("Personalized Recommendations", heading_style))
    
    recommendations = []
    if student_data['sleep_duration'] < 7:
        recommendations.append("• Increase sleep duration to 7-9 hours per night")
    if student_data['study_hours'] > 7:
        recommendations.append("• Reduce study hours and focus on quality over quantity")
    if student_data['screen_time'] > 8:
        recommendations.append("• Limit recreational screen time to reduce eye strain")
    if student_data['stress_levels'] > 6:
        recommendations.append("• Practice stress management techniques (meditation, deep breathing)")
    if student_data['physical_activity'] < 3:
        recommendations.append("• Increase physical activity to at least 3-4 hours per week")
    if student_data['social_interaction'] < 3:
        recommendations.append("• Enhance social connections with friends and family")
    
    if not recommendations:
        recommendations.append("• Excellent! Continue maintaining your healthy lifestyle habits")
    
    for rec in recommendations:
        story.append(Paragraph(rec, styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Footer
    footer_text = f"""
    <para align=center>
    <font size=9 color="#666666">
    This report was generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>
    For support, please contact your campus counseling center.<br/>
    <b>Emergency:</b> If you're in crisis, call the National Suicide Prevention Lifeline: 988
    </font>
    </para>
    """
    story.append(Paragraph(footer_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# ==================== DATA GENERATION (keeping original) ====================
@st.cache_data
def generate_student_data(n_samples=500, random_state=42):
    """Generate synthetic student lifestyle data"""
    np.random.seed(random_state)
    
    sleep_duration = np.clip(np.random.normal(6.5, 1.5, n_samples), 3, 10)
    study_hours = np.clip(np.random.normal(5, 2, n_samples), 1, 12)
    screen_time = np.clip(np.random.normal(7, 2.5, n_samples), 2, 14)
    
    stress_base = 5 + (8 - sleep_duration) * 0.5 + (study_hours - 5) * 0.3 + (screen_time - 7) * 0.2
    stress_levels = np.clip(stress_base + np.random.normal(0, 1, n_samples), 1, 10)
    
    physical_activity = np.clip(np.random.normal(3, 1.5, n_samples), 0, 10)
    social_interaction = np.clip(np.random.normal(4, 1.5, n_samples), 0, 8)
    
    burnout_score = (
        (8 - sleep_duration) * 0.15 +
        (study_hours - 3) * 0.1 +
        (screen_time - 5) * 0.08 +
        (stress_levels - 5) * 0.2 +
        (5 - physical_activity) * 0.05 +
        (5 - social_interaction) * 0.05
    )
    
    burnout_score = (burnout_score - burnout_score.min()) / (burnout_score.max() - burnout_score.min())
    burnout_score = np.clip(burnout_score + np.random.normal(0, 0.1, n_samples), 0, 1)
    burnout_binary = (burnout_score > 0.6).astype(int)
    
    df = pd.DataFrame({
        'sleep_duration': sleep_duration,
        'study_hours': study_hours,
        'screen_time': screen_time,
        'stress_levels': stress_levels,
        'physical_activity': physical_activity,
        'social_interaction': social_interaction,
        'burnout_score': burnout_score,
        'burnout': burnout_binary
    })
    
    return df

# ==================== MODEL TRAINING ====================
@st.cache_resource
def train_model(df):
    """Train logistic regression model"""
    features = ['sleep_duration', 'study_hours', 'screen_time', 'stress_levels',
                'physical_activity', 'social_interaction']
    X = df[features]
    y = df['burnout']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    return model, scaler, features, X_test, y_test, y_pred, y_pred_proba

# ==================== RISK CALCULATION ====================
def calculate_risk_score(student_data):
    """Calculate risk score based on thresholds"""
    score = 0
    
    if student_data['sleep_duration'] < 6:
        score += 3
    elif student_data['sleep_duration'] < 7:
        score += 2
        
    if student_data['study_hours'] > 8:
        score += 3
    elif student_data['study_hours'] > 6:
        score += 2
        
    if student_data['screen_time'] > 9:
        score += 3
    elif student_data['screen_time'] > 6:
        score += 2
        
    if student_data['stress_levels'] > 7:
        score += 3
    elif student_data['stress_levels'] > 5:
        score += 2
        
    if student_data['physical_activity'] < 1.5:
        score += 2
    elif student_data['physical_activity'] < 3:
        score += 1
        
    if student_data['social_interaction'] < 1.5:
        score += 2
    elif student_data['social_interaction'] < 3:
        score += 1
        
    return score

def get_risk_category(score):
    """Get risk category based on score"""
    if score <= 4:
        return "Low Risk", "#4caf50"
    elif score <= 8:
        return "Medium Risk", "#ff9800"
    else:
        return "High Risk", "#f44336"

# ==================== VISUALIZATION FUNCTIONS ====================
def plot_correlation_matrix(df):
    """Create interactive correlation heatmap"""
    features = ['sleep_duration', 'study_hours', 'screen_time', 'stress_levels', 
                'physical_activity', 'social_interaction', 'burnout_score']
    corr_matrix = df[features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[f.replace('_', ' ').title() for f in corr_matrix.columns],
        y=[f.replace('_', ' ').title() for f in corr_matrix.columns],
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Correlation Matrix: Student Lifestyle Factors",
        height=600,
        xaxis_title="",
        yaxis_title="",
        font=dict(size=12)
    )
    
    return fig

def plot_roc_curve(y_test, y_pred_proba):
    """Create ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='#1f77b4', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="ROC Curve: Burnout Prediction Model",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        hovermode='x unified'
    )
    
    return fig

# ==================== COUNSELOR DASHBOARD ====================
def show_counselor_dashboard():
    """Counselor dashboard for monitoring all students"""
    st.markdown("""
    <div style='background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); 
                padding: 2rem; border-radius: 20px; margin-bottom: 2rem;
                box-shadow: 0 8px 24px rgba(72, 187, 120, 0.2);'>
        <h1 style='color: white; margin: 0; font-size: 2rem; -webkit-text-fill-color: white;'>
            👨‍⚕️ Counselor Control Center
        </h1>
        <p style='color: rgba(255, 255, 255, 0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
            Monitor, track, and intervene across your entire student population
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get all students and assessments
    students_df = get_all_students()
    assessments_df = get_all_assessments()
    
    if len(students_df) == 0:
        st.info("📝 No students in the system yet. Students will appear here after their first assessment.")
        return
    
    # Overview metrics with new professional design
    st.markdown("""
    <div style='background: white; padding: 1.5rem; border-radius: 16px; 
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-bottom: 2rem;'>
        <h3 style='color: #2d3748; margin: 0 0 1rem 0; font-size: 1.25rem;'>
            📊 Population Overview
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_students = len(students_df)
    
    if len(assessments_df) > 0:
        # Get latest assessment for each student
        latest_assessments = assessments_df.groupby('student_id').first().reset_index()
        
        high_risk = len(latest_assessments[latest_assessments['risk_category'] == 'High Risk'])
        medium_risk = len(latest_assessments[latest_assessments['risk_category'] == 'Medium Risk'])
        low_risk = len(latest_assessments[latest_assessments['risk_category'] == 'Low Risk'])
        avg_probability = latest_assessments['burnout_probability'].mean() * 100
    else:
        high_risk = medium_risk = low_risk = 0
        avg_probability = 0
    
    with col1:
        st.metric("Total Students", total_students)
    with col2:
        st.metric("🔴 High Risk", high_risk)
    with col3:
        st.metric("🟡 Medium Risk", medium_risk)
    with col4:
        st.metric("🟢 Low Risk", low_risk)
    
    st.markdown("---")
    
    # Risk distribution chart
    if len(assessments_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Distribution")
            risk_counts = latest_assessments['risk_category'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.4,
                marker=dict(colors=['#4caf50', '#ff9800', '#f44336'])
            )])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Average Burnout Probability by Department")
            if 'department' in students_df.columns:
                dept_stats = latest_assessments.merge(students_df[['student_id', 'department']], 
                                                      on='student_id', how='left')
                dept_avg = dept_stats.groupby('department')['burnout_probability'].mean() * 100
                
                fig = go.Figure(data=[go.Bar(
                    x=dept_avg.index,
                    y=dept_avg.values,
                    marker_color='#1f77b4'
                )])
                fig.update_layout(
                    height=350,
                    yaxis_title="Avg Burnout Probability (%)",
                    xaxis_title="Department"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Student list with filtering
    st.subheader("🎓 Student List & Monitoring")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.selectbox("Filter by Risk", 
                                   ["All", "High Risk", "Medium Risk", "Low Risk"])
    
    with col2:
        if 'department' in students_df.columns:
            departments = ["All"] + list(students_df['department'].unique())
            dept_filter = st.selectbox("Filter by Department", departments)
        else:
            dept_filter = "All"
    
    with col3:
        search = st.text_input("🔍 Search by Name/ID")
    
    # Filter data
    if len(assessments_df) > 0:
        display_df = latest_assessments.merge(
            students_df[['student_id', 'name', 'department', 'email']], 
            on='student_id', 
            how='left'
        )
        
        # Remove any rows with NaN values (students without proper data)
        display_df = display_df.dropna(subset=['risk_category', 'burnout_probability'])
        
        if risk_filter != "All":
            display_df = display_df[display_df['risk_category'] == risk_filter]
        
        if dept_filter != "All":
            display_df = display_df[display_df['department'] == dept_filter]
        
        if search:
            display_df = display_df[
                display_df['name'].str.contains(search, case=False, na=False) |
                display_df['student_id'].str.contains(search, case=False, na=False)
            ]
        
        # Display table
        if len(display_df) > 0:
            display_cols = ['student_id', 'name', 'department', 'risk_category', 
                           'burnout_probability', 'assessment_date']
            display_df['burnout_probability'] = (display_df['burnout_probability'] * 100).round(1)
            display_df = display_df[display_cols].sort_values('burnout_probability', ascending=False)
            
            st.dataframe(
                display_df.style.background_gradient(subset=['burnout_probability'], cmap='RdYlGn_r'),
                use_container_width=True,
                height=400
            )
            
            # Individual student details
            st.markdown("---")
            st.subheader("📋 Individual Student Details")
            
            # Safely format student selection
            def format_student_name(student_id):
                try:
                    student_row = display_df[display_df['student_id'] == student_id]
                    if len(student_row) > 0 and 'name' in student_row.columns:
                        name = student_row['name'].values[0]
                        return f"{student_id} - {name}"
                    else:
                        return student_id
                except:
                    return student_id
            
            selected_student = st.selectbox(
                "Select Student",
                display_df['student_id'].tolist(),
                format_func=format_student_name
            )
            
            if selected_student:
                student_assessments = get_student_assessments(selected_student)
                
                # Safely get student info with error handling
                student_info_df = students_df[students_df['student_id'] == selected_student]
                if len(student_info_df) == 0:
                    st.error(f"⚠️ Student {selected_student} not found in the database.")
                    return
                
                student_info = student_info_df.iloc[0]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Assessment History**")
                    if len(student_assessments) > 0:
                        # Trend chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=student_assessments['assessment_date'],
                            y=student_assessments['burnout_probability'] * 100,
                            mode='lines+markers',
                            name='Burnout Risk',
                            line=dict(color='#1f77b4', width=2)
                        ))
                        fig.update_layout(
                            height=300,
                            yaxis_title="Burnout Probability (%)",
                            xaxis_title="Date",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(student_assessments[['assessment_date', 'risk_category', 
                                                         'burnout_probability', 'risk_score']], 
                                   use_container_width=True)
                
                with col2:
                    st.write("**Quick Actions**")
                    
                    # Add intervention
                    with st.expander("➕ Add Intervention"):
                        intervention_type = st.selectbox(
                            "Intervention Type",
                            ["Counseling Session", "Wellness Workshop", "Referral", 
                             "Follow-up Call", "Email Check-in", "Other"]
                        )
                        notes = st.text_area("Notes")
                        counselor_name = st.text_input("Counselor Name")
                        
                        if st.button("Save Intervention"):
                            if counselor_name and notes:
                                save_intervention(selected_student, intervention_type, 
                                                notes, counselor_name)
                                st.success("✅ Intervention recorded!")
                            else:
                                st.error("Please fill in all fields")
                    
                    # Send alert
                    if st.button("📧 Send Alert Email"):
                        st.info(f"Email alert would be sent to: {student_info['email']}")
                    
                    # Generate report - only if assessments exist
                    if len(student_assessments) > 0:
                        latest = student_assessments.iloc[0]
                        student_data = {
                            'sleep_duration': latest['sleep_duration'],
                            'study_hours': latest['study_hours'],
                            'screen_time': latest['screen_time'],
                            'stress_levels': latest['stress_levels'],
                            'physical_activity': latest['physical_activity'],
                            'social_interaction': latest['social_interaction']
                        }
                        
                        pdf_buffer = generate_pdf_report(
                            student_data,
                            latest['risk_score'],
                            latest['burnout_probability'],
                            latest['risk_category'],
                            student_info.to_dict()
                        )
                        
                        st.download_button(
                            label="📄 Download Report",
                            data=pdf_buffer,
                            file_name=f"burnout_report_{selected_student}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.warning("⚠️ No assessments available for this student yet.")
        else:
            st.info("No students match the current filters.")
    else:
        st.info("No assessments recorded yet.")

# ==================== INDIVIDUAL ASSESSMENT PAGE ====================
def show_individual_assessment(model, scaler, features):
    """Individual assessment page with database integration"""
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 20px; margin-bottom: 2rem;
                box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2);'>
        <h1 style='color: white; margin: 0; font-size: 2rem; -webkit-text-fill-color: white;'>
            👤 Individual Risk Assessment
        </h1>
        <p style='color: rgba(255, 255, 255, 0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
            Complete lifestyle evaluation and personalized recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Student registration with new design
    with st.expander("📝 Student Registration (Optional - Save Your Data)", expanded=False):
        st.markdown("""
        <p style='color: #718096; margin-bottom: 1.5rem;'>
            Register to track your progress over time and receive personalized insights
        </p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            student_id = st.text_input("Student ID *", placeholder="e.g., S12345")
            name = st.text_input("Full Name *", placeholder="e.g., John Doe")
            email = st.text_input("Email Address *", placeholder="e.g., john@university.edu")
        with col2:
            department = st.selectbox("Department", 
                ["Computer Science", "Engineering", "Business", "Arts", "Science", "Medicine", "Law", "Other"])
            year = st.selectbox("Academic Year", [1, 2, 3, 4, 5, "Graduate"])
            st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("💾 Register Student Profile", use_container_width=True):
            if student_id and name and email:
                if save_student(student_id, name, email, department, year):
                    st.success("✅ Profile created successfully! You can now save assessments.")
            else:
                st.error("⚠️ Please complete all required fields (*)")
    
    st.markdown("<br>", unsafe_allow_html=True)
    # Lifestyle data input section with new design
    st.markdown("""
    <div style='background: white; padding: 2rem; border-radius: 16px; 
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-bottom: 2rem;'>
        <h2 style='color: #2d3748; margin: 0 0 0.5rem 0; border: none; padding: 0;'>
            📋 Lifestyle Assessment
        </h2>
        <p style='color: #718096; margin: 0 0 2rem 0;'>
            Please provide accurate information about your daily habits over the past week
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**😴 Sleep & Rest**")
        sleep_duration = st.slider("Sleep Duration (hours/day)", 3.0, 10.0, 7.0, 0.5)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("**📚 Academic Work**")
        study_hours = st.slider("Study Hours (hours/day)", 1.0, 12.0, 5.0, 0.5)
    
    with col2:
        st.markdown("**📱 Screen & Technology**")
        screen_time = st.slider("Screen Time (hours/day)", 2.0, 14.0, 7.0, 0.5)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("**😰 Stress & Pressure**")
        stress_levels = st.slider("Stress Level (1-10 scale)", 1.0, 10.0, 5.0, 0.5)
    
    with col3:
        st.markdown("**🏃 Physical Activity**")
        physical_activity = st.slider("Exercise (hours/week)", 0.0, 10.0, 3.0, 0.5)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("**👥 Social Connection**")
        social_interaction = st.slider("Social Time (hours/day)", 0.0, 8.0, 4.0, 0.5)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Save options with new design
    col1, col2 = st.columns([2, 1])
    with col1:
        save_to_db = st.checkbox("💾 Save this assessment to my profile", value=False)
        if save_to_db:
            save_student_id = st.text_input("Enter your Student ID", key="save_id", placeholder="e.g., S12345")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Center the button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        assess_button = st.button("🔍 Analyze My Burnout Risk", type="primary", use_container_width=True)
    
    if assess_button:
        student_data = {
            'sleep_duration': sleep_duration,
            'study_hours': study_hours,
            'screen_time': screen_time,
            'stress_levels': stress_levels,
            'physical_activity': physical_activity,
            'social_interaction': social_interaction
        }
        
        # Calculate risk score
        risk_score = calculate_risk_score(student_data)
        risk_category, risk_color = get_risk_category(risk_score)
        
        # Model prediction
        student_df = pd.DataFrame([student_data])
        student_scaled = scaler.transform(student_df[features])
        burnout_probability = model.predict_proba(student_scaled)[0][1]
        
        # Save to database if requested
        if save_to_db and save_student_id:
            if save_assessment(save_student_id, student_data, risk_score, 
                             burnout_probability, risk_category):
                st.success("✅ Assessment saved to database!")
        
        # Display results with new professional design
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Results header
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h2 style='color: #2d3748; font-size: 1.75rem; margin: 0; border: none; padding: 0;'>
                📊 Your Assessment Results
            </h2>
            <p style='color: #718096; margin-top: 0.5rem;'>
                Based on your lifestyle factors, here's your comprehensive risk analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Risk Score</h3>
                <h1>{risk_score}<span style='font-size: 1.5rem; color: #718096;'>/16</span></h1>
                <p style='color: #718096; margin: 0.5rem 0 0 0; font-size: 0.875rem;'>
                    Comprehensive risk index
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            risk_class = "risk-high" if "High" in risk_category else "risk-medium" if "Medium" in risk_category else "risk-low"
            st.markdown(f"""
            <div class='metric-card {risk_class}'>
                <h3>Risk Category</h3>
                <h1 style='font-size: 1.75rem;'>{risk_category}</h1>
                <p style='color: #718096; margin: 0.5rem 0 0 0; font-size: 0.875rem;'>
                    Current risk classification
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Burnout Probability</h3>
                <h1>{burnout_probability*100:.1f}<span style='font-size: 1.5rem; color: #718096;'>%</span></h1>
                <p style='color: #718096; margin: 0.5rem 0 0 0; font-size: 0.875rem;'>
                    ML-predicted likelihood
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=burnout_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Burnout Risk Meter"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "#e8f5e9"},
                    {'range': [40, 70], 'color': "#fff3e0"},
                    {'range': [70, 100], 'color': "#ffebee"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations with new design
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 16px; 
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin: 2rem 0;'>
            <h2 style='color: #2d3748; margin: 0 0 1.5rem 0; border: none; padding: 0;'>
                💡 Personalized Action Plan
            </h2>
        """, unsafe_allow_html=True)
        
        recommendations = []
        if sleep_duration < 7:
            recommendations.append({
                'icon': '😴',
                'title': 'Improve Sleep Hygiene',
                'desc': 'Increase sleep duration to 7-9 hours per night. Better sleep reduces burnout risk by up to 25%.',
                'priority': 'high'
            })
        if study_hours > 7:
            recommendations.append({
                'icon': '📚',
                'title': 'Optimize Study Time',
                'desc': 'Reduce study hours and focus on quality over quantity. Use techniques like Pomodoro.',
                'priority': 'medium'
            })
        if screen_time > 8:
            recommendations.append({
                'icon': '📱',
                'title': 'Reduce Screen Time',
                'desc': 'Limit recreational screen time to reduce eye strain and mental fatigue.',
                'priority': 'medium'
            })
        if stress_levels > 6:
            recommendations.append({
                'icon': '🧘',
                'title': 'Stress Management',
                'desc': 'Practice daily stress reduction: meditation, deep breathing, or mindfulness exercises.',
                'priority': 'high'
            })
        if physical_activity < 3:
            recommendations.append({
                'icon': '🏃',
                'title': 'Increase Physical Activity',
                'desc': 'Aim for at least 3-4 hours of exercise per week. Even light walking helps.',
                'priority': 'medium'
            })
        if social_interaction < 3:
            recommendations.append({
                'icon': '👥',
                'title': 'Enhance Social Connections',
                'desc': 'Spend more quality time with friends and family. Social support is protective.',
                'priority': 'medium'
            })
        
        if recommendations:
            for i, rec in enumerate(recommendations):
                priority_color = '#f56565' if rec['priority'] == 'high' else '#ed8936'
                priority_label = 'HIGH PRIORITY' if rec['priority'] == 'high' else 'RECOMMENDED'
                
                st.markdown(f"""
                <div style='background: #f7fafc; padding: 1.25rem; border-radius: 12px; 
                            margin-bottom: 1rem; border-left: 4px solid {priority_color};'>
                    <div style='display: flex; align-items: start; gap: 1rem;'>
                        <span style='font-size: 2rem;'>{rec['icon']}</span>
                        <div style='flex: 1;'>
                            <div style='display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;'>
                                <h4 style='margin: 0; color: #2d3748; font-size: 1.1rem;'>{rec['title']}</h4>
                                <span style='background: {priority_color}; color: white; padding: 0.25rem 0.5rem; 
                                            border-radius: 4px; font-size: 0.7rem; font-weight: 600;'>
                                    {priority_label}
                                </span>
                            </div>
                            <p style='margin: 0; color: #718096; line-height: 1.6;'>{rec['desc']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%); 
                        padding: 2rem; border-radius: 12px; text-align: center;'>
                <h3 style='color: #22543d; margin: 0 0 0.5rem 0;'>🎉 Excellent Balance!</h3>
                <p style='color: #2f855a; margin: 0;'>
                    Your lifestyle habits are well-balanced. Keep up the great work!
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # PDF Report generation with new design
        st.markdown("<br>", unsafe_allow_html=True)
        
        student_info = None
        if save_to_db and save_student_id:
            students = get_all_students()
            if len(students) > 0 and save_student_id in students['student_id'].values:
                student_info = students[students['student_id'] == save_student_id].iloc[0].to_dict()
        
        pdf_buffer = generate_pdf_report(student_data, risk_score, burnout_probability, 
                                         risk_category, student_info)
        
        # Center the download button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="📄 Download Complete PDF Report",
                data=pdf_buffer,
                file_name=f"burnout_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# ==================== MAIN APP ====================
def main():
    # Initialize database
    init_database()
    
    # Header with new professional design
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0 3rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>🧠 Student Burnout Analysis</h1>
        <p style='font-size: 1.25rem; color: #718096; font-weight: 500;'>
            Advanced Risk Assessment & Wellness Monitoring System
        </p>
        <div style='display: flex; justify-content: center; gap: 1rem; margin-top: 1.5rem;'>
            <span class='badge badge-success'>✓ CSV Upload</span>
            <span class='badge badge-success'>✓ Database</span>
            <span class='badge badge-success'>✓ Counselor Dashboard</span>
            <span class='badge badge-success'>✓ PDF Reports</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar with new professional design
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0 2rem 0;'>
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        width: 80px; height: 80px; border-radius: 20px; 
                        margin: 0 auto 1rem auto; display: flex; 
                        align-items: center; justify-content: center;
                        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);'>
                <span style='font-size: 3rem;'>🧠</span>
            </div>
            <h2 style='color: white; margin: 0; font-size: 1.5rem;'>Burnout Analysis</h2>
            <p style='color: #cbd5e0; font-size: 0.875rem; margin-top: 0.5rem;'>Professional Wellness Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Role selection with enhanced styling
        st.markdown("<h3 style='color: white; font-size: 1.1rem; margin-bottom: 1rem;'>👤 Select Your Role</h3>", 
                   unsafe_allow_html=True)
        role = st.radio("", ["Student", "Counselor"], label_visibility="collapsed")
        
        st.markdown("---")
        
        if role == "Student":
            st.markdown("<h3 style='color: white; font-size: 1.1rem; margin-bottom: 1rem;'>📱 Navigate</h3>", 
                       unsafe_allow_html=True)
            page = st.radio(
                "",
                ["👤 Assessment", "📊 Dashboard", "🔍 Analytics"],
                label_visibility="collapsed"
            )
        else:
            page = "👨‍⚕️ Counselor Dashboard"
        
        st.markdown("---")
        
        # CSV Upload Section with new design
        st.markdown("<h3 style='color: white; font-size: 1.1rem; margin-bottom: 1rem;'>📁 Data Upload</h3>", 
                   unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=['csv'], label_visibility="collapsed")
        
        if uploaded_file is not None:
            df = load_csv_data(uploaded_file)
            if df is not None:
                st.session_state['uploaded_data'] = df
                st.markdown(f"""
                <div style='background: rgba(72, 187, 120, 0.2); padding: 1rem; 
                            border-radius: 8px; border-left: 3px solid #48bb78;'>
                    <p style='color: #c6f6d5; margin: 0; font-weight: 600;'>
                        ✓ {len(df)} records loaded
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data source selection
        if 'uploaded_data' not in st.session_state:
            st.markdown("<h3 style='color: white; font-size: 1.1rem; margin-bottom: 1rem;'>🎲 Generate Data</h3>", 
                       unsafe_allow_html=True)
            n_samples = st.slider("Sample Size", 100, 1000, 500, 50, label_visibility="collapsed")
            
            if st.button("🔄 Regenerate", use_container_width=True):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
        
        st.markdown("---")
        
        # About section with new design
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.1); padding: 1.25rem; 
                    border-radius: 12px; backdrop-filter: blur(10px);'>
            <h4 style='color: white; margin: 0 0 0.75rem 0; font-size: 0.95rem;'>ℹ️ About</h4>
            <p style='color: #cbd5e0; font-size: 0.85rem; line-height: 1.6; margin: 0;'>
                Advanced analytics platform for student mental health monitoring and early intervention.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load data
    if 'uploaded_data' in st.session_state:
        df = st.session_state['uploaded_data']
    else:
        df = generate_student_data(n_samples=n_samples)
    
    # Train model
    model, scaler, features, X_test, y_test, y_pred, y_pred_proba = train_model(df)
    
    # Route to pages
    if role == "Counselor":
        show_counselor_dashboard()
    elif page == "👤 Assessment":
        show_individual_assessment(model, scaler, features)
    elif page == "📊 Dashboard":
        show_dashboard(df, model, features)
    elif page == "🔍 Analytics":
        show_data_analysis(df)

def show_dashboard(df, model, features):
    """Dashboard page"""
    st.markdown("""
    <div style='background: white; padding: 2rem; border-radius: 20px; margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); border-left: 5px solid #667eea;'>
        <h1 style='color: #2d3748; margin: 0; font-size: 2rem; -webkit-text-fill-color: #2d3748;'>
            📊 Analytics Dashboard
        </h1>
        <p style='color: #718096; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
            Real-time insights and population health metrics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        burnout_rate = df['burnout'].mean() * 100
        st.metric("Burnout Rate", f"{burnout_rate:.1f}%", f"{len(df[df['burnout']==1])} students")
    
    with col2:
        avg_stress = df['stress_levels'].mean()
        st.metric("Avg Stress Level", f"{avg_stress:.1f}/10", "High" if avg_stress > 6 else "Moderate")
    
    with col3:
        avg_sleep = df['sleep_duration'].mean()
        st.metric("Avg Sleep", f"{avg_sleep:.1f} hrs", "Low" if avg_sleep < 7 else "Good")
    
    with col4:
        avg_study = df['study_hours'].mean()
        st.metric("Avg Study Time", f"{avg_study:.1f} hrs", "High" if avg_study > 7 else "Normal")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Burnout Distribution")
        burnout_counts = df['burnout'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=['No Burnout', 'Burnout'],
            values=burnout_counts.values,
            hole=0.4,
            marker=dict(colors=['#2ecc71', '#e74c3c'])
        )])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Feature Correlations")
        st.plotly_chart(plot_correlation_matrix(df), use_container_width=True, height=350)

def show_data_analysis(df):
    """Data analysis page"""
    st.markdown("""
    <div style='background: white; padding: 2rem; border-radius: 20px; margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); border-left: 5px solid #48bb78;'>
        <h1 style='color: #2d3748; margin: 0; font-size: 2rem; -webkit-text-fill-color: #2d3748;'>
            🔍 Statistical Analysis
        </h1>
        <p style='color: #718096; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
            Deep dive into correlations and population statistics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📋 Descriptive Stats", "🔗 Correlations"])
    
    with tab1:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)
        
        st.subheader("Burnout vs No Burnout Comparison")
        comparison = df.groupby('burnout').agg({
            'sleep_duration': ['mean', 'std'],
            'study_hours': ['mean', 'std'],
            'screen_time': ['mean', 'std'],
            'stress_levels': ['mean', 'std'],
            'physical_activity': ['mean', 'std'],
            'social_interaction': ['mean', 'std']
        }).round(2)
        st.dataframe(comparison, use_container_width=True)
    
    with tab2:
        st.subheader("Correlation Analysis")
        st.plotly_chart(plot_correlation_matrix(df), use_container_width=True)

if __name__ == "__main__":
    main()


