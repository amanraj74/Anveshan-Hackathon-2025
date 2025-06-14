# Churn Prediction Dashboard

A beautiful, robust, and explainable dashboard for predicting customer churn in fintech.

## 🚀 Overview
- Predicts which customers are likely to churn using advanced machine learning (XGBoost).
- Interactive dashboard for uploading data, visualizing churn risk, and downloading predictions.
- Model explainability with feature importance and SHAP values.

## 🛠️ Setup
1. Clone this repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run app.py
   ```

## 📊 Dashboard Features
- **Upload & Parse:** Accepts CSVs with behavioral/transactional features. Handles missing/noisy data.
- **AUC-ROC Optimized:** Trains a binary classifier for 30-day churn.
- **Visuals:**
  - Churn probability histogram
  - Churn vs Retain pie chart
  - Top-10 risk table
  - Feature importance bar chart
  - SHAP explainability for top customer
- **Download:** Export predictions as CSV.
- **Reset:** Start over with one click.

## 🧠 Model
- XGBoost classifier, cross-validated for best AUC-ROC
- Robust preprocessing (missing values, encoding, scaling)
- Feature importance and SHAP explainability

## 🖼️ Screenshots
![Dashboard Screenshot](image.png)

## 🎥 Video Demo
[Watch the demo](https://your-demo-link.com)

## 👥 Team
- SARTHAK RAHA
- [GitHub](https://github.com/your-repo)

## 📄 License
MIT

## Project Structure

- `model.py`: Core ML model implementation
- `app.py`: Streamlit dashboard application
- `utils.py`: Helper functions
- `requirements.txt`: Project dependencies

## Model Performance

The model is optimized for AUC-ROC score and includes:
- Feature importance analysis
- SHAP value explanations
- Cross-validation results

## Data

The project uses the telco dataset with behavioral and transactional features. The data is preprocessed to handle missing values and categorical variables.

## License

MIT License
