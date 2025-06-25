from data_loader import load_data, preprocess_data, split_data, get_processed_dataframe, create_timeseries_dataset
from models import ModelTrainer
from visualization import (
    plot_feature_importance,
    plot_predictions,
    plot_residuals,
    plot_correlation_matrix
)
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

def main():
    # Load and preprocess data
    df = load_data('202402_Copy.xlsx')
    print(f"Original dataset shape: {df.shape}")
    
    # Get processed dataframe for detailed analysis
    processed_df = get_processed_dataframe(df)
    print(f"Processed dataset shape: {processed_df.shape}")
    
    # Create time-series dataset for XGBoost MultiOutput
    X_timeseries, Y_timeseries = create_timeseries_dataset(processed_df)
    print(f"Time-series dataset shape: X={X_timeseries.shape}, Y={Y_timeseries.shape}")
    
    # Split time-series data
    X_train, X_test, Y_train, Y_test = train_test_split(X_timeseries, Y_timeseries, test_size=0.2, random_state=42)
    
    # Convert DataFrames to numpy arrays for compatibility
    X_train = X_train.values if hasattr(X_train, 'values') else X_train
    X_test = X_test.values if hasattr(X_test, 'values') else X_test
    Y_train = Y_train.values if hasattr(Y_train, 'values') else Y_train
    Y_test = Y_test.values if hasattr(Y_test, 'values') else Y_test
    
    # Initialize model trainer
    trainer = ModelTrainer()
    
    # Train and evaluate models
    models_to_train = ['xgboost_multi', 'random_forest', 'svr']
    results = {}
    
    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()} model...")

        if model_name in ['xgboost_multi', 'random_forest']:
            # Use time-series data for multi-output models
            trainer.train_model(model_name, X_train, Y_train)
            metrics = trainer.evaluate_model_detailed(model_name, X_test, Y_test)
            results[model_name] = metrics

            # Print detailed results
            print(f"\nModel Performance (Predicting ARR YoY Growth for Q1–Q4):")
            print(f"MAE: {metrics['mae']:.2f} | R² Score: {metrics['r2']:.2f}")

            # Print quarterly performance
            for quarter in [1, 2, 3, 4]:
                quarter_key = f'Q{quarter}_r2'
                if quarter_key in metrics['quarterly_r2']:
                    print(f"Q{quarter} R² Score: {metrics['quarterly_r2'][quarter_key]:.2f}")

            # Get predictions for plotting
            Y_pred = trainer.trained_models[model_name].predict(X_test)

            # Plot results for MultiOutput models
            plot_predictions(Y_test.flatten(), Y_pred.flatten(), 
                           f"{model_name} - Actual vs Predicted (Q1-Q4)", f"{model_name}_predictions.png")
            plot_residuals(Y_test.flatten(), Y_pred.flatten(), 
                          f"{model_name} - Residuals (Q1-Q4)", f"{model_name}_residuals.png")

            # Save the model
            joblib.dump(trainer.trained_models[model_name], f"{model_name}_model.pkl")

        else:
            # For SVR or other single-output models, use the original single-output approach
            X_single = processed_df.drop(['id_company', 'ARR YoY Growth (in %)', 'Quarter Num'], axis=1)
            y_single = processed_df['ARR YoY Growth (in %)']
            companies = processed_df['id_company'].values
            quarters = processed_df['Quarter Num'].values

            # Split data while preserving company and quarter information
            X_train_single, X_test_single, y_train_single, y_test_single, companies_train, companies_test, quarters_train, quarters_test = train_test_split(
                X_single, y_single, companies, quarters, test_size=0.2, random_state=42
            )

            # Convert to numpy arrays
            X_train_single = X_train_single.values if hasattr(X_train_single, 'values') else X_train_single
            X_test_single = X_test_single.values if hasattr(X_test_single, 'values') else X_test_single
            y_train_single = y_train_single.values if hasattr(y_train_single, 'values') else y_train_single
            y_test_single = y_test_single.values if hasattr(y_test_single, 'values') else y_test_single

            # Print company counts for single-output models
            if model_name == 'svr':  # Only print once for SVR
                unique_companies_train = len(np.unique(companies_train))
                unique_companies_test = len(np.unique(companies_test))
                print(f"\nUnique Companies in Training Set: {unique_companies_train}")
                print(f"Unique Companies in Testing Set: {unique_companies_test}")

            trainer.train_model(model_name, X_train_single, y_train_single)
            metrics = trainer.evaluate_model_detailed(model_name, X_test_single, y_test_single, companies_test, quarters_test)
            results[model_name] = metrics

            # Print detailed results
            print(f"\nModel Performance (Predicting ARR YoY Growth):")
            print(f"MAE: {metrics['mae']:.2f} | R² Score: {metrics['r2']:.2f}")

            # Print quarterly performance
            for quarter in [1, 2, 3, 4]:
                quarter_key = f'Q{quarter}_r2'
                if quarter_key in metrics['quarterly_r2']:
                    print(f"Q{quarter} R² Score: {metrics['quarterly_r2'][quarter_key]:.2f}")

            # Get predictions
            y_pred = trainer.trained_models[model_name].predict(X_test_single)

            # Plot results with unique filenames
            plot_predictions(y_test_single, y_pred, f"{model_name} - Actual vs Predicted", f"{model_name}_predictions.png")
            plot_residuals(y_test_single, y_pred, f"{model_name} - Residuals", f"{model_name}_residuals.png")

        # Plot feature importance if available
        try:
            importance = trainer.get_feature_importance(model_name)
            if model_name in ['xgboost_multi', 'random_forest']:
                feature_names = X_timeseries.columns.tolist()
            else:
                feature_names = X_single.columns.tolist()
            plot_feature_importance(feature_names, importance, f"{model_name} - Feature Importance", f"{model_name}_feature_importance.png")
        except ValueError:
            print(f"Feature importance not available for {model_name}")
    
    # Plot correlation matrix
    plot_correlation_matrix(processed_df, "Correlation Matrix", "correlation_matrix.png")
    
    # Print final summary
    print(f"\n{'='*50}")
    print("FINAL MODEL COMPARISON:")
    print(f"{'='*50}")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")

if __name__ == "__main__":
    main() 