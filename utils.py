import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_pred_proba):
    """
    Evaluate model performance using various metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    metrics = {}
    
    # Calculate AUC-ROC
    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
    
    return metrics

def plot_precision_recall_curve(y_true, y_pred_proba):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP={avg_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    return plt

def plot_feature_correlations(df, target_col='churn'):
    """
    Plot correlation matrix for numerical features.
    
    Args:
        df: DataFrame containing features
        target_col: Target column name
    """
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    return plt

def get_optimal_threshold(y_true, y_pred_proba):
    """
    Find optimal threshold for classification based on F1 score.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        float: Optimal threshold
    """
    from sklearn.metrics import f1_score
    
    thresholds = np.arange(0.1, 1.0, 0.1)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    return optimal_threshold

def format_predictions(predictions, customer_ids):
    """
    Format predictions into a DataFrame with customer IDs.
    
    Args:
        predictions: Array of prediction probabilities
        customer_ids: Array of customer IDs
        
    Returns:
        DataFrame: Formatted predictions
    """
    return pd.DataFrame({
        'Customer ID': customer_ids,
        'Churn Probability': predictions
    }).sort_values('Churn Probability', ascending=False) 