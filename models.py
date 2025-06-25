import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

class ModelTrainer:
    def __init__(self):
        self.models = {
            'xgboost_multi': MultiOutputRegressor(
                XGBRegressor(
                    objective='reg:squarederror', 
                    n_estimators=100, 
                    learning_rate=0.1, 
                    random_state=42
                )
            ),
            'random_forest': MultiOutputRegressor(
                RandomForestRegressor(n_estimators=100, random_state=42)
            ),
            'svr': SVR(kernel='rbf')
        }
        self.trained_models = {}
        
    def train_model(self, model_name, X_train, y_train):
        """
        Train a specific model
        
        Args:
            model_name (str): Name of the model to train
            X_train (np.array): Training features
            y_train (np.array): Training target
            
        Returns:
            object: Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        return model
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a trained model
        
        Args:
            model_name (str): Name of the model to evaluate
            X_test (np.array): Test features
            y_test (np.array): Test target
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
            
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        # For multi-output models, flatten the predictions
        if len(y_pred.shape) > 1:
            y_pred_flat = y_pred.flatten()
            y_test_flat = y_test.flatten()
        else:
            y_pred_flat = y_pred
            y_test_flat = y_test
        
        metrics = {
            'mse': mean_squared_error(y_test_flat, y_pred_flat),
            'rmse': np.sqrt(mean_squared_error(y_test_flat, y_pred_flat)),
            'mae': mean_absolute_error(y_test_flat, y_pred_flat),
            'r2': r2_score(y_test_flat, y_pred_flat)
        }
        
        return metrics
    
    def evaluate_model_detailed(self, model_name, X_test, y_test, test_companies=None, test_quarters=None):
        """
        Evaluate a trained model with detailed analysis including company counts and quarterly performance
        
        Args:
            model_name (str): Name of the model to evaluate
            X_test (np.array): Test features
            y_test (np.array): Test target
            test_companies (np.array): Company IDs for test set (optional)
            test_quarters (np.array): Quarter numbers for test set (optional)
            
        Returns:
            dict: Dictionary containing detailed evaluation metrics
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
            
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        # For multi-output models, handle differently
        if len(y_pred.shape) > 1:
            y_pred_flat = y_pred.flatten()
            y_test_flat = y_test.flatten()
            
            # Calculate quarterly R² scores for multi-output
            quarterly_r2 = {}
            for i in range(y_pred.shape[1]):
                quarter_r2 = r2_score(y_test[:, i], y_pred[:, i])
                quarterly_r2[f'Q{i+1}_r2'] = quarter_r2
        else:
            y_pred_flat = y_pred
            y_test_flat = y_test
            quarterly_r2 = {}
            
            # Calculate quarterly R² scores for single output
            if test_quarters is not None:
                for quarter in [1, 2, 3, 4]:
                    quarter_mask = test_quarters == quarter
                    if np.sum(quarter_mask) > 0:
                        quarter_r2 = r2_score(y_test[quarter_mask], y_pred[quarter_mask])
                        quarterly_r2[f'Q{quarter}_r2'] = quarter_r2
        
        # Basic metrics
        metrics = {
            'mse': mean_squared_error(y_test_flat, y_pred_flat),
            'rmse': np.sqrt(mean_squared_error(y_test_flat, y_pred_flat)),
            'mae': mean_absolute_error(y_test_flat, y_pred_flat),
            'r2': r2_score(y_test_flat, y_pred_flat)
        }
        
        # Company counts
        if test_companies is not None:
            unique_companies = len(np.unique(test_companies))
            metrics['unique_companies'] = unique_companies
        
        metrics['quarterly_r2'] = quarterly_r2
        
        return metrics
    
    def get_feature_importance(self, model_name):
        """
        Get feature importance for a trained model
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            np.array: Feature importance scores
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
            
        model = self.trained_models[model_name]
        
        if model_name == 'xgboost_multi':
            # For MultiOutput XGBoost, average feature importance across all estimators
            feature_importance = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
            return feature_importance
        elif hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_)
        else:
            raise ValueError(f"Model {model_name} does not support feature importance")