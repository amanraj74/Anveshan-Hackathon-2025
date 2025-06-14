import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import shap
import joblib

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def preprocess_data(self, df, is_training=True):
        """Preprocess the data for model training or prediction."""
        df = df.copy()
        
        # Handle missing values
        df = df.fillna(df.mean() if is_training else 0)
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        
        # Scale numerical features
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if is_training:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            
        return df
    
    def train(self, X, y):
        """Train the XGBoost model."""
        self.feature_names = X.columns
        
        # Preprocess the data
        X_processed = self.preprocess_data(X, is_training=True)
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # Initialize the model
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric='auc'
        )
        
        # Train the model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Calculate and print performance metrics
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)
        print(f"Validation AUC-ROC Score: {auc_score:.4f}")
        
        return auc_score
    
    def predict(self, X):
        """Generate churn probabilities for new data."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_processed = self.preprocess_data(X, is_training=False)
        return self.model.predict_proba(X_processed)[:, 1]
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        return feature_importance.sort_values('importance', ascending=False)
    
    def get_shap_values(self, X):
        """Calculate SHAP values for model explainability."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_processed = self.preprocess_data(X, is_training=False)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_processed)
        return shap_values
    
    def save_model(self, path):
        """Save the model and preprocessing components."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load_model(cls, path):
        """Load a saved model."""
        model_data = joblib.load(path)
        predictor = cls()
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.label_encoders = model_data['label_encoders']
        predictor.feature_names = model_data['feature_names']
        return predictor 