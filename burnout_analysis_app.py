"""
Student Burnout Risk Analysis - Interactive Web Application
Beautiful UI/UX with real-time visualizations and model predictions
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
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Student Burnout Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left-color: #4caf50;
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    h2 {
        color: #2c3e50;
        font-weight: 600;
        margin-top: 2rem;
    }
    h3 {
        color: #34495e;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# Data Generation Function
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

# Model Training Function
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

# Visualization Functions
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

def plot_feature_distributions(df):
    """Create distribution plots comparing burnout vs no burnout"""
    features = ['sleep_duration', 'study_hours', 'screen_time', 
                'stress_levels', 'physical_activity', 'social_interaction']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f.replace('_', ' ').title() for f in features]
    )
    
    colors = ['#2ecc71', '#e74c3c']  # Green for no burnout, Red for burnout
    
    for idx, feature in enumerate(features):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        for burnout_status, color in zip([0, 1], colors):
            data = df[df['burnout'] == burnout_status][feature]
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name='Burnout' if burnout_status == 1 else 'No Burnout',
                    marker_color=color,
                    opacity=0.7,
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=700,
        title_text="Feature Distributions by Burnout Status",
        barmode='overlay',
        font=dict(size=11)
    )
    
    return fig

def plot_box_comparison(df):
    """Create box plots comparing features by burnout status"""
    features = ['sleep_duration', 'study_hours', 'screen_time', 
                'stress_levels', 'physical_activity', 'social_interaction']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f.replace('_', ' ').title() for f in features]
    )
    
    for idx, feature in enumerate(features):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        for burnout_status, color in zip([0, 1], ['#2ecc71', '#e74c3c']):
            data = df[df['burnout'] == burnout_status][feature]
            fig.add_trace(
                go.Box(
                    y=data,
                    name='Burnout' if burnout_status == 1 else 'No Burnout',
                    marker_color=color,
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=700,
        title_text="Box Plot Comparison: Burnout vs No Burnout",
        font=dict(size=11)
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

def plot_feature_importance(model, features):
    """Create feature importance plot"""
    importance = np.abs(model.coef_[0])
    feature_names = [f.replace('_', ' ').title() for f in features]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=feature_names,
        orientation='h',
        marker=dict(
            color=importance,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Importance")
        )
    ))
    
    fig.update_layout(
        title="Feature Importance (Absolute Coefficients)",
        xaxis_title="Importance",
        yaxis_title="Features",
        height=400
    )
    
    return fig

def plot_confusion_matrix(y_test, y_pred):
    """Create confusion matrix heatmap"""
    cm = confusion_matrix(y_test, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: No Burnout', 'Predicted: Burnout'],
        y=['Actual: No Burnout', 'Actual: Burnout'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        height=400
    )
    
    return fig

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

# Main Application
def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>🧠 Student Burnout Risk Analysis System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; color: #666;'>Statistical Modeling & Early Detection with Interactive Visualizations</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/mental-state.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["📊 Dashboard", "🔍 Data Analysis", "🤖 Model Performance", "👤 Individual Assessment", "📈 Visualizations"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This application uses statistical modeling to identify early warning signs of student burnout based on lifestyle factors.")
        
        st.markdown("### Dataset Info")
        n_samples = st.slider("Number of Students", 100, 1000, 500, 50)
        
        if st.button("🔄 Regenerate Data"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    
    # Load data and train model
    df = generate_student_data(n_samples=n_samples)
    model, scaler, features, X_test, y_test, y_pred, y_pred_proba = train_model(df)
    
    # Page routing
    if page == "📊 Dashboard":
        show_dashboard(df, model, features)
    elif page == "🔍 Data Analysis":
        show_data_analysis(df)
    elif page == "🤖 Model Performance":
        show_model_performance(model, features, X_test, y_test, y_pred, y_pred_proba)
    elif page == "👤 Individual Assessment":
        show_individual_assessment(model, scaler, features)
    elif page == "📈 Visualizations":
        show_visualizations(df)

def show_dashboard(df, model, features):
    """Dashboard page with key metrics"""
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        burnout_rate = df['burnout'].mean() * 100
        st.metric(
            label="Burnout Rate",
            value=f"{burnout_rate:.1f}%",
            delta=f"{len(df[df['burnout']==1])} students"
        )
    
    with col2:
        avg_stress = df['stress_levels'].mean()
        st.metric(
            label="Avg Stress Level",
            value=f"{avg_stress:.1f}/10",
            delta="High" if avg_stress > 6 else "Moderate"
        )
    
    with col3:
        avg_sleep = df['sleep_duration'].mean()
        st.metric(
            label="Avg Sleep",
            value=f"{avg_sleep:.1f} hrs",
            delta="Low" if avg_sleep < 7 else "Good"
        )
    
    with col4:
        avg_study = df['study_hours'].mean()
        st.metric(
            label="Avg Study Time",
            value=f"{avg_study:.1f} hrs",
            delta="High" if avg_study > 7 else "Normal"
        )
    
    st.markdown("---")
    
    # Charts
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
        st.subheader("Risk Categories")
        df['risk_score'] = df.apply(lambda row: calculate_risk_score(row.to_dict()), axis=1)
        df['risk_category'] = pd.cut(df['risk_score'], bins=[-1, 4, 8, 20],
                                       labels=['Low Risk', 'Medium Risk', 'High Risk'])
        risk_counts = df['risk_category'].value_counts()
        
        fig = go.Figure(data=[go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            marker=dict(color=['#4caf50', '#ff9800', '#f44336'])
        )])
        fig.update_layout(height=350, yaxis_title="Number of Students")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("Top Risk Factors")
    st.plotly_chart(plot_feature_importance(model, features), use_container_width=True)

def show_data_analysis(df):
    """Data analysis page"""
    st.header("🔍 Statistical Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📋 Descriptive Stats", "🔗 Correlations", "📊 Comparisons"])
    
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
        
        # Correlation with burnout score
        features = ['sleep_duration', 'study_hours', 'screen_time', 'stress_levels',
                   'physical_activity', 'social_interaction', 'burnout_score']
        corr_matrix = df[features].corr()
        burnout_corr = corr_matrix['burnout_score'].sort_values(ascending=False)
        
        st.write("**Correlation with Burnout Score:**")
        corr_df = pd.DataFrame({
            'Feature': burnout_corr.index,
            'Correlation': burnout_corr.values
        })
        st.dataframe(corr_df.round(3), use_container_width=True)
        
        st.plotly_chart(plot_correlation_matrix(df), use_container_width=True)
    
    with tab3:
        st.subheader("Feature Comparisons")
        st.plotly_chart(plot_box_comparison(df), use_container_width=True)

def show_model_performance(model, features, X_test, y_test, y_pred, y_pred_proba):
    """Model performance page"""
    st.header("🤖 Model Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auc_score = roc_auc_score(y_test, y_pred_proba)
        st.metric("ROC-AUC Score", f"{auc_score:.3f}")
        
        accuracy = (y_pred == y_test).mean()
        st.metric("Accuracy", f"{accuracy*100:.1f}%")
    
    with col2:
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        st.metric("Precision (Burnout)", f"{report['1']['precision']:.2f}")
        st.metric("Recall (Burnout)", f"{report['1']['recall']:.2f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROC Curve")
        st.plotly_chart(plot_roc_curve(y_test, y_pred_proba), use_container_width=True)
    
    with col2:
        st.subheader("Confusion Matrix")
        st.plotly_chart(plot_confusion_matrix(y_test, y_pred), use_container_width=True)
    
    st.subheader("Detailed Classification Report")
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
    st.dataframe(report_df.round(2), use_container_width=True)
    
    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame({
        'Feature': [f.replace('_', ' ').title() for f in features],
        'Coefficient': model.coef_[0],
        'Abs_Coefficient': np.abs(model.coef_[0])
    }).sort_values('Abs_Coefficient', ascending=False)
    
    fig = go.Figure(go.Bar(
        x=coef_df['Coefficient'],
        y=coef_df['Feature'],
        orientation='h',
        marker=dict(
            color=coef_df['Coefficient'],
            colorscale='RdBu',
            cmid=0
        )
    ))
    fig.update_layout(
        title="Model Coefficients (Impact on Burnout Prediction)",
        xaxis_title="Coefficient Value",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def show_individual_assessment(model, scaler, features):
    """Individual assessment page"""
    st.header("👤 Individual Student Risk Assessment")
    
    st.markdown("Enter student lifestyle data to assess burnout risk:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sleep_duration = st.slider("Sleep Duration (hours/day)", 3.0, 10.0, 7.0, 0.5)
        study_hours = st.slider("Study Hours (hours/day)", 1.0, 12.0, 5.0, 0.5)
    
    with col2:
        screen_time = st.slider("Screen Time (hours/day)", 2.0, 14.0, 7.0, 0.5)
        stress_levels = st.slider("Stress Level (1-10)", 1.0, 10.0, 5.0, 0.5)
    
    with col3:
        physical_activity = st.slider("Physical Activity (hours/week)", 0.0, 10.0, 3.0, 0.5)
        social_interaction = st.slider("Social Interaction (hours/day)", 0.0, 8.0, 4.0, 0.5)
    
    if st.button("🔍 Assess Burnout Risk", type="primary"):
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
        
        # Display results
        st.markdown("---")
        st.subheader("Assessment Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;'>
                <h3 style='color: #666; margin: 0; font-size: 16px;'>Risk Score</h3>
                <h1 style='color: #2c3e50; margin: 10px 0; font-size: 48px;'>{risk_score}/16</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid {risk_color};'>
                <h3 style='color: #666; margin: 0; font-size: 16px;'>Risk Category</h3>
                <h1 style='color: {risk_color}; margin: 10px 0; font-size: 48px;'>{risk_category}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;'>
                <h3 style='color: #666; margin: 0; font-size: 16px;'>Burnout Probability</h3>
                <h1 style='color: #2c3e50; margin: 10px 0; font-size: 48px;'>{burnout_probability*100:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
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
        
        # Recommendations
        st.subheader("📋 Personalized Recommendations")
        
        recommendations = []
        if sleep_duration < 7:
            recommendations.append("😴 **Increase sleep duration** to 7-9 hours per night")
        if study_hours > 7:
            recommendations.append("📚 **Reduce study hours** and focus on quality over quantity")
        if screen_time > 8:
            recommendations.append("📱 **Limit screen time** to reduce eye strain and mental fatigue")
        if stress_levels > 6:
            recommendations.append("🧘 **Practice stress management** techniques like meditation or yoga")
        if physical_activity < 3:
            recommendations.append("🏃 **Increase physical activity** to at least 3-4 hours per week")
        if social_interaction < 3:
            recommendations.append("👥 **Enhance social connections** with friends and family")
        
        if recommendations:
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.success("✅ Great job! Your lifestyle habits are well-balanced. Keep it up!")

def show_visualizations(df):
    """Visualizations page"""
    st.header("📈 Interactive Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "📦 Box Plots", "🎯 Scatter Plots"])
    
    with tab1:
        st.subheader("Feature Distributions by Burnout Status")
        st.plotly_chart(plot_feature_distributions(df), use_container_width=True)
    
    with tab2:
        st.subheader("Box Plot Comparisons")
        st.plotly_chart(plot_box_comparison(df), use_container_width=True)
    
    with tab3:
        st.subheader("Scatter Plot Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X-axis", 
                ['sleep_duration', 'study_hours', 'screen_time', 
                 'stress_levels', 'physical_activity', 'social_interaction'],
                index=0)
        with col2:
            y_feature = st.selectbox("Y-axis",
                ['sleep_duration', 'study_hours', 'screen_time', 
                 'stress_levels', 'physical_activity', 'social_interaction'],
                index=3)
        
        fig = px.scatter(
            df, x=x_feature, y=y_feature,
            color='burnout',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            labels={'burnout': 'Burnout Status'},
            title=f"{x_feature.replace('_', ' ').title()} vs {y_feature.replace('_', ' ').title()}",
            opacity=0.6,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
