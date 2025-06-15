import streamlit as st
import pandas as pd
import plotly.express as px
from churn_model import ChurnPredictor
import os

st.set_page_config(page_title="Churn Prediction Dashboard", page_icon="📊", layout="wide")

# Set Streamlit dark theme (if config.toml is not used)
st.markdown("""
    <style>
        .main {background-color: #1A1A2E; color: #FFFFFF;}
        .stButton>button {background-color: #00CFFF; color: white;}
        .stDownloadButton>button {background-color: #00CFFF; color: white;}
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Churn Dashboard")
st.sidebar.markdown("Upload your CSV to get started.")

# Tabs for workflow
tab1, tab2 = st.sidebar.tabs(["Train Model", "Make Predictions"])

# Session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'customer_ids' not in st.session_state:
    st.session_state.customer_ids = None

# Train Model Tab
with tab1:
    st.subheader("Upload Training Data")
    train_file = st.file_uploader("CSV with 'churned' column", type="csv", key="train")
    if train_file is not None:
        with st.spinner("Loading data..."):
            train_data = pd.read_csv(train_file)
            if 'churned' not in train_data.columns and 'churn' not in train_data.columns:
                st.error("Training file must contain a 'churned' or 'churn' column!")
            else:
                # Standardize the column name
                if 'churn' in train_data.columns:
                    train_data.rename(columns={'churn': 'churned'}, inplace=True)
                st.session_state.customer_ids = train_data['customerID'] if 'customerID' in train_data.columns else train_data.index
                st.session_state.data = train_data
                
                # Show data preview
                st.write("Data Preview:")
                st.dataframe(train_data.head())
                
                # Training progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, status):
                    progress_bar.progress(progress)
                    status_text.text(status)
                
                with st.spinner("Training model..."):
                    model = ChurnPredictor()
                    X = train_data.drop('churned', axis=1)
                    y = train_data['churned']
                    
                    # Update progress
                    update_progress(0.3, "Preprocessing data...")
                    
                    # Train model
                    update_progress(0.5, "Training model...")
                    auc = model.train(X, y)
                    
                    # Save model
                    update_progress(0.8, "Saving model...")
                    model.save("churn_model.joblib")
                    
                    # Complete
                    update_progress(1.0, "Training complete!")
                    st.success(f"Model trained! Mean CV AUC-ROC: {auc:.4f}")
                    st.session_state.model = model
                    st.info("Model saved.")

# Make Predictions Tab
with tab2:
    st.subheader("Upload Data for Prediction")
    test_file = st.file_uploader("CSV for prediction", type="csv", key="test")
    if test_file is not None:
        test_data = pd.read_csv(test_file)
        st.session_state.customer_ids = test_data['customerID'] if 'customerID' in test_data.columns else test_data.index
        st.session_state.data = test_data
        if st.session_state.model is None:
            if os.path.exists("churn_model.joblib"):
                st.session_state.model = ChurnPredictor.load("churn_model.joblib")
                st.success("Loaded trained model.")
            else:
                st.error("No trained model available. Please train a model first!")
                st.stop()
        with st.spinner("Generating predictions..."):
            try:
                predictions = st.session_state.model.predict(test_data)
                st.session_state.predictions = predictions
                st.success("Predictions generated!")
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.session_state.predictions = None

# Main Dashboard
st.title("📊 Churn Prediction Dashboard")
st.markdown("Predict and visualize customer churn. Upload your data to get started!")

if st.session_state.data is not None and st.session_state.predictions is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Probability Distribution")
        fig = px.histogram(
            x=st.session_state.predictions,
            nbins=30,
            color_discrete_sequence=["#00CFFF"],
            labels={'x': 'Churn Probability', 'y': 'Count'}
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="#FF4B4B", annotation_text="Threshold 0.5")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Distribution of predicted churn probabilities.")

    with col2:
        st.subheader("Churn vs Retain")
        churn_labels = ["Churn" if p > 0.5 else "Retain" for p in st.session_state.predictions]
        pie = px.pie(
            names=churn_labels,
            color_discrete_sequence=["#FF4B4B", "#00CFFF"],
            title="Churn vs Retain"
        )
        st.plotly_chart(pie, use_container_width=True)
        st.caption("Proportion of users predicted to churn vs. retain.")

    st.subheader("Top 10 High-Risk Customers")
    risk_df = pd.DataFrame({
        'Customer ID': st.session_state.customer_ids,
        'Churn Probability': st.session_state.predictions
    })
    risk_df = risk_df.sort_values('Churn Probability', ascending=False).head(10)
    st.dataframe(risk_df, use_container_width=True)
    st.caption("Users most at risk of churning.")

    st.subheader("Feature Importance")
    try:
        feature_importance = st.session_state.model.get_feature_importance()
        fig_importance = px.bar(
            feature_importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            color='importance',
            color_continuous_scale='Blues'
        )
        fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_importance, use_container_width=True)
        st.caption("Top features influencing churn.")
    except Exception as e:
        st.warning(f"Feature importance could not be displayed: {e}")

    # SHAP explainability for top-1 customer
    st.subheader("Why is this customer at risk?")
    try:
        shap_values = st.session_state.model.get_shap_values(st.session_state.data.head(1))
        st.write("SHAP values for the highest risk customer:")
        st.json(dict(zip(st.session_state.model.feature_names, shap_values[0])))
    except Exception as e:
        st.info("SHAP explainability available for training data only.")

    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=pd.DataFrame({
            'Customer ID': st.session_state.customer_ids,
            'Churn Probability': st.session_state.predictions
        }).to_csv(index=False),
        file_name="churn_predictions.csv",
        mime="text/csv"
    )
    st.caption("Download the full predictions for further analysis or submission.")

    if st.button("🔄 Reset Dashboard"):
        st.session_state.model = None
        st.session_state.predictions = None
        st.session_state.data = None
        st.session_state.customer_ids = None
        st.rerun()
else:
    st.info("👈 Upload a CSV file to begin. Use the sidebar to train or predict.")
    st.markdown("---")
    st.markdown("**Tips:**\n- Use the 'Train Model' tab to train on labeled data.\n- Use the 'Make Predictions' tab for new/unlabeled data.\n- Download your predictions for submission.\n- See feature importance for explainability.")
    st.markdown("---")
    st.caption("Made with ❤️ for DS-2 Hackathon | [GitHub](https://github.com/amanraj74/masai-hackathon.git)") 