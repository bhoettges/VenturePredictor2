import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def plot_feature_importance(feature_names, importance_scores, title="Feature Importance", filename="feature_importance.png"):
    """
    Plot feature importance scores
    
    Args:
        feature_names (list): List of feature names
        importance_scores (np.array): Feature importance scores
        title (str): Plot title
        filename (str): Output filename
    """
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(title)
    plt.tight_layout()
    
    # Save to output folder
    output_path = os.path.join('output', filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance plot to: {output_path}")

def plot_predictions(y_true, y_pred, title="Actual vs Predicted Values", filename="predictions.png"):
    """
    Plot actual vs predicted values
    
    Args:
        y_true (np.array): True target values
        y_pred (np.array): Predicted target values
        title (str): Plot title
        filename (str): Output filename
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.tight_layout()
    
    # Save to output folder
    output_path = os.path.join('output', filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved predictions plot to: {output_path}")

def plot_residuals(y_true, y_pred, title="Residual Plot", filename="residuals.png"):
    """
    Plot residuals
    
    Args:
        y_true (np.array): True target values
        y_pred (np.array): Predicted target values
        title (str): Plot title
        filename (str): Output filename
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.tight_layout()
    
    # Save to output folder
    output_path = os.path.join('output', filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved residuals plot to: {output_path}")

def plot_correlation_matrix(df, title="Correlation Matrix", filename="correlation_matrix.png"):
    """
    Plot correlation matrix
    
    Args:
        df (pd.DataFrame): Input dataframe
        title (str): Plot title
        filename (str): Output filename
    """
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title(title)
    plt.tight_layout()
    
    # Save to output folder
    output_path = os.path.join('output', filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation matrix to: {output_path}") 