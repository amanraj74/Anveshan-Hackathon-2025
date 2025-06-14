import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib

class EnsembleChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None

    def feature_engineering(self, df):
        df = df.copy()
        # Example features
        if 'TotalCharges' in df.columns and 'tenure' in df.columns:
            df['AvgChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
        if 'SeniorCitizen' in df.columns and 'InternetService' in df.columns:
            df['IsSeniorAndFiber'] = ((df['SeniorCitizen'] == 1) & (df['InternetService'] == 'Fiber optic')).astype(int)
        if 'tenure' in df.columns:
            df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72, np.inf], labels=[1,2,3,4,5])
        return df

    def preprocess(self, df, training=True):
        df = self.feature_engineering(df)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean() if training else 0)
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col == 'churned':
                continue
            if training:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if training:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        return df

    def train(self, X, y):
        X_proc = self.preprocess(X, training=True)
        self.feature_names = X_proc.columns
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_proc, y)
        # Base models
        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='auc', tree_method='hist', random_state=42, n_jobs=-1,
            n_estimators=200, learning_rate=0.07, max_depth=5, subsample=0.8, colsample_bytree=0.8
        )
        lgbm_clf = lgb.LGBMClassifier(
            objective='binary', metric='auc', random_state=42, n_jobs=-1,
            n_estimators=200, learning_rate=0.07, max_depth=5, subsample=0.8, colsample_bytree=0.8
        )
        rf_clf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42, n_jobs=-1)
        # Stacking ensemble
        ensemble = StackingClassifier(
            estimators=[
                ('xgb', xgb_clf),
                ('lgbm', lgbm_clf),
                ('rf', rf_clf)
            ],
            final_estimator=xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='auc', tree_method='hist', random_state=42, n_jobs=-1
            ),
            n_jobs=-1,
            passthrough=True
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = cross_val_score(ensemble, X_res, y_res, cv=skf, scoring='roc_auc', n_jobs=-1)
        print(f"Mean CV AUC-ROC (Ensemble): {aucs.mean():.4f} ± {aucs.std():.4f}")
        ensemble.fit(X_res, y_res)
        self.model = ensemble
        return aucs.mean()

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        X_proc = self.preprocess(X, training=False)
        return self.model.predict_proba(X_proc)[:, 1]

    def save(self, path):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls()
        obj.model = data['model']
        obj.scaler = data['scaler']
        obj.label_encoders = data['label_encoders']
        obj.feature_names = data['feature_names']
        return obj 