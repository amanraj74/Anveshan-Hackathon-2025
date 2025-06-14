import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import shap
import joblib
from functools import lru_cache
from imblearn.over_sampling import SMOTE

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
        Trains the XGBoost model with advanced tuning and prints cross-validated AUC-ROC.
        """
        X_proc = self.preprocess(X, training=True)
        self.feature_names = X_proc.columns

        # Handle class imbalance with SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_proc, y)

        # Hyperparameter grid for RandomizedSearchCV
        param_dist = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 8],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2, 0.5],
            'min_child_weight': [1, 3, 5, 7],
            'scale_pos_weight': [1, 2, 5, 10]
        }

        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            tree_method='hist',
            random_state=42,
            n_jobs=-1
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            xgb_clf, param_distributions=param_dist, n_iter=20,
            scoring='roc_auc', n_jobs=-1, cv=skf, verbose=1, random_state=42
        )
        search.fit(X_res, y_res)

        print(f"Best params: {search.best_params_}")
        print(f"Best CV AUC-ROC: {search.best_score_:.4f}")

        # Fit best model on all data with early stopping
        best_model = search.best_estimator_
        best_model.fit(
            X_res, y_res,
            eval_set=[(X_res, y_res)],
            early_stopping_rounds=20,
            verbose=False
        )
        self.model = best_model
        return search.best_score_

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