README
On
STUDENT BURNOUT RISK ANALYSIS APP
(A Project on Data and Statistical Modelling)


* Submitted by:
Name: Jyot Bhavnani                   Name: Diya Ajudiya 
Roll No.: 25BCE383                      Roll No.: 25BCE074




* Overview:
This web application analyzes student burnout risk using synthetic lifestyle data and a logistic regression model. It provides interactive dashboards, visualizations, and personalized assessments for early detection of burnout.


Key features include real-time metrics, model performance evaluation, and risk scoring based on factors like sleep, stress, and activity levels.


Primary Goal:
Early Detection & Intervention: Identify students at high burnout risk through statistical modeling of daily habits (sleep, study hours, screen time, stress levels, physical activity, social interaction), enabling educators and counselors to implement targeted interventions before academic performance declines.


* Use case:
Designed for educators, counselors, and students to identify burnout early through lifestyle patterns. Users input data for individual predictions or explore dataset-wide insights via interactive plots and stats.
Supports proactive interventions in academic settings by highlighting high-risk students via color-coded categories (low, medium, high).


* Unique Selling Point:
* Interactive visualization
   * Plotly charts for correlations
   * ROC curves
   *  Feature distributions
   * gauges showing burnout probability
* Real-time Predictions
   * Logistic regression model with personalized risk scores and recommendations
* Multi-View Analytics
   * 5 interconnected pages provide cohort analysis, individual assessment, and model diagnostics in one app
* Real-Time Data Refresh
   * Slider adjusts sample size (100-1000 students) with instant model retraining and visualization updates.


* Features:


* Dashboard: Burnout rate, stress averages, pie/bar charts for distributions.
* Data Analysis: Descriptive stats, correlations, box plots comparing burnout vs. no-burnout groups.
* Model Performance: ROC-AUC, confusion matrix, classification reports, feature importance.
* Individual Assessment: Sliders for inputs, risk meter, probability gauge, tailored advice.
* Visualizations: Histograms, scatter plots, customizable by user selection


* Data Generation:
Synthetic dataset mimics student habits: sleep duration (3-10 hrs), study hours (1-12), screen time (2-14), stress (1-10), activity (0-10), social interaction (0-8). Burnout binary derived from normalized scores.
* Code Structure:
1. Imports & Configuration
Core libraries: Streamlit, NumPy, Pandas, Scikit-learn, Plotl
Custom CSS styling for professional UI/UX
Warning suppression for clean output


2.Core Data Science Functions 
generatestudentdata(): Synthetic dataset generation (500 students default()
trainmodel(): Complete ML pipeline (split, scale, LogisticRegression)
calculateriskscore(): Rule-based risk scoring (threshold penalties)
10+ Plotly visualization functions


3. Page Handler Functions (5 main modules)
showdashboard(): KPI metrics, pie charts, feature importance
showdataanalysis(): Descriptive stats, correlations, box plots
showmodelperformance(): ROC-AUC, classification report, model diagnostics
showindividualassessment(): Input sliders, risk gauge, recommendations
showvisualizations(): Interactive histograms, scatter plots
4. Main Application Logic
Sidebar navigation: st.radio() with 5 page options
Dynamic data loading based on sample size slider 
Conditional page rendering based on user selection
Cache clearing on "Regenerate Data" button








* Limitations & Future Enhancements:




Limitations:
Uses synthetic data; train on real data for production. Model assumes linear relationships—consider advanced ML for non-linear patterns.


Future Enhancements:
The Student Burnout Risk Analysis App has strong potential for evolution into an enterprise-grade data science platform for student wellness.


ML & Modeling Upgrades
Advanced Algorithms: Implement Random Forest, XGBoost, or Neural Networks for improved AUC (>0.85 target) and non-linear pattern detection.
​


Deep Learning: Add LSTM models for time-series analysis of longitudinal student data (weekly habit tracking).


Ensemble Methods: Combine logistic regression with gradient boosting for robust predictions across diverse student demographics.
