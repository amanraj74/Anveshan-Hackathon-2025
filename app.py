import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from churn_model import ChurnPredictor
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Churn Prediction Dashboard")
st.markdown("""
This dashboard helps you predict and analyze customer churn using machine learning.
Upload your data to get started!
""")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'data' not in st.session_state:
    st.session_state.data = None

# Sidebar
st.sidebar.header("Upload Data")

# Add tabs for training and prediction
tab1, tab2 = st.sidebar.tabs(["Train Model", "Make Predictions"])

with tab1:
    st.subheader("Upload Training Data")
    train_file = st.file_uploader("Choose a training CSV file with 'churn' column", type="csv", key="train")
    
    if train_file is not None:
        train_data = pd.read_csv(train_file)
        if 'churn' not in train_data.columns:
            st.error("Training file must contain a 'churn' column!")
        else:
            st.session_state.data = train_data
            with st.spinner("Training model..."):
                model = ChurnPredictor()
                X = train_data.drop('churn', axis=1)
                y = train_data['churn']
                auc_score = model.train(X, y)
                st.success(f"Model trained successfully! AUC-ROC Score: {auc_score:.4f}")
                st.session_state.model = model
                
                # Save the trained model
                model.save_model("churn_model.joblib")
                st.info("Model saved successfully!")

with tab2:
    st.subheader("Upload Data for Prediction")
    test_file = st.file_uploader("Choose a CSV file for prediction", type="csv", key="test")
    
    if test_file is not None:
        if st.session_state.model is None:
            # Try to load saved model
            if os.path.exists("churn_model.joblib"):
                st.session_state.model = ChurnPredictor.load_model("churn_model.joblib")
                st.success("Loaded previously trained model!")
            else:
                st.error("No trained model available. Please train a model first!")
                st.stop()
        
        test_data = pd.read_csv(test_file)
        st.session_state.data = test_data
        
        with st.spinner("Generating predictions..."):
            try:
                predictions = st.session_state.model.predict(test_data)
                st.session_state.predictions = predictions
                st.success("Predictions generated successfully!")
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
                st.session_state.predictions = None

# Main content
if st.session_state.data is not None and st.session_state.predictions is not None:
    # Create two columns for the main metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn probability distribution
        fig_dist = px.histogram(
            x=st.session_state.predictions,
            nbins=50,
            title="Churn Probability Distribution",
            labels={'x': 'Churn Probability', 'y': 'Count'}
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Churn vs Retain pie chart
        churn_threshold = 0.5
        churn_counts = pd.Series(st.session_state.predictions > churn_threshold).value_counts()
        fig_pie = px.pie(
            values=churn_counts.values,
            names=['Retain', 'Churn'],
            title="Churn vs Retain Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Top 10 risk customers
    st.subheader("Top 10 High-Risk Customers")
    risk_df = pd.DataFrame({
        'Customer ID': st.session_state.data.index,
        'Churn Probability': st.session_state.predictions
    })
    risk_df = risk_df.sort_values('Churn Probability', ascending=False).head(10)
    st.dataframe(risk_df)
    
    # Feature importance
    if st.session_state.model is not None:
        st.subheader("Feature Importance")
        feature_importance = st.session_state.model.get_feature_importance()
        fig_importance = px.bar(
            feature_importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title="Top 10 Most Important Features"
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Download predictions
    st.download_button(
        label="Download Predictions",
        data=pd.DataFrame({
            'Customer ID': st.session_state.data.index,
            'Churn Probability': st.session_state.predictions
        }).to_csv(index=False),
        file_name="churn_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Please use the sidebar to train a model or make predictions!") 