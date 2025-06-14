import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import shap
import joblib
from functools import lru_cache

class ChurnPredictor:
    """
    Churn prediction model with robust preprocessing and explainability.
    """
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None

    def preprocess(self, df, training=True):
        """
        Preprocesses the input DataFrame for model training or prediction.
        """
        df = df.copy()
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
        
        # Fill missing numerics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean() if training else 0)
        
        # Encode categoricals (except target)
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col == 'churned':
                continue
            if training:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        
        # Scale numerics
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if training:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df

    @lru_cache(maxsize=32)
    def _get_cached_preprocess(self, data_hash, training):
        """
        Cached preprocessing implementation.
        """
        return self.preprocess(pd.read_json(data_hash), training)

    def train(self, X, y):
        """
        Trains the XGBoost model and prints cross-validated AUC-ROC.
        Optimized for faster training in Streamlit.
        """
        # Convert DataFrame to JSON string for caching
        data_hash = X.to_json()
        X_proc = self._get_cached_preprocess(data_hash, True)
        self.feature_names = X_proc.columns
        
        # Reduced number of CV folds for faster training
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Optimized XGBoost parameters for faster training
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,  # Reduced from 200
            learning_rate=0.1,  # Increased from 0.07
            max_depth=4,  # Reduced from 5
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc',
            tree_method='hist',  # Faster training method
            n_jobs=-1  # Use all available cores
        )
        
        # Quick cross-validation
        aucs = cross_val_score(model, X_proc, y, cv=skf, scoring='roc_auc', n_jobs=-1)
        print(f"Mean CV AUC-ROC: {aucs.mean():.4f} ± {aucs.std():.4f}")
        
        # Fit on all data
        model.fit(X_proc, y)
        self.model = model
        return aucs.mean()

    def predict(self, X):
        """
        Predicts churn probability for new data.
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        # Convert DataFrame to JSON string for caching
        data_hash = X.to_json()
        X_proc = self._get_cached_preprocess(data_hash, False)
        return self.model.predict_proba(X_proc)[:, 1]

    def get_feature_importance(self):
        """
        Returns feature importance as a DataFrame.
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def get_shap_values(self, X):
        """
        Returns SHAP values for explainability.
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        X_proc = self.preprocess(X, training=False)
        explainer = shap.TreeExplainer(self.model)
        return explainer.shap_values(X_proc)

    def save(self, path):
        """
        Saves the model and preprocessing objects.
        """
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, path)

    @classmethod
    def load(cls, path):
        """
        Loads a saved model.
        """
        data = joblib.load(path)
        obj = cls()
        obj.model = data['model']
        obj.scaler = data['scaler']
        obj.label_encoders = data['label_encoders']
        obj.feature_names = data['feature_names']
        return obj 