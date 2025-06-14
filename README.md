# Churn Prediction Dashboard

A machine learning-powered tool for predicting customer churn in the fintech space, featuring an interactive dashboard for visualization and analysis.

## Features

- Churn probability prediction using XGBoost
- Interactive dashboard with visualizations
- Model explainability using SHAP values
- Export functionality for predictions
- Mobile-responsive design

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run app.py
   ```

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

## Dashboard Features

- Churn probability distribution
- Churn vs Retain pie chart
- Top 10 risk customers table
- Download predictions functionality
- Model explainability visualizations

## Data

The project uses the telco dataset with behavioral and transactional features. The data is preprocessed to handle missing values and categorical variables.

## Team

[Your team information here]

## License

MIT License 