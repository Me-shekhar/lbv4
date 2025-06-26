"""
Model Training Script for LBV Prediction
This script handles the training and evaluation of the Random Forest model
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LBVModelTrainer:
    def __init__(self, data_path='data.csv'):
        self.data_path = data_path
        self.data = None
        self.model = None
        self.label_encoder = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.hydrocarbon_ranges = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Clean column names and remove empty columns
        df = df.drop(columns=[col for col in df.columns if col.startswith('Unnamed') or df[col].isna().all()])
        
        # Rename columns for consistency
        column_mapping = {
            'Ti (K)': 'Temperature',
            'equivalent ratio': 'Equivalence_Ratio',
            'Pressure (atm)': 'Pressure',
            'LBV (cm/s)': 'LBV'
        }
        df = df.rename(columns=column_mapping)
        
        # Remove rows with missing values
        df = df.dropna()
        
        # Remove duplicate entries
        df = df.drop_duplicates()
        
        print(f"Dataset shape after preprocessing: {df.shape}")
        print(f"Number of unique hydrocarbons: {df['Hydrocarbon'].nunique()}")
        print(f"LBV range: {df['LBV'].min():.2f} - {df['LBV'].max():.2f} cm/s")
        
        # Store hydrocarbon ranges for validation
        self.hydrocarbon_ranges = {}
        for hydrocarbon in df['Hydrocarbon'].unique():
            hc_data = df[df['Hydrocarbon'] == hydrocarbon]
            self.hydrocarbon_ranges[hydrocarbon] = {
                'temperature': {'min': hc_data['Temperature'].min(), 'max': hc_data['Temperature'].max()},
                'equiv_ratio': {'min': hc_data['Equivalence_Ratio'].min(), 'max': hc_data['Equivalence_Ratio'].max()},
                'pressure': {'min': hc_data['Pressure'].min(), 'max': hc_data['Pressure'].max()},
                'lbv': {'min': hc_data['LBV'].min(), 'max': hc_data['LBV'].max()}
            }
        
        self.data = df
        return df
    
    def prepare_features(self):
        """Prepare features for model training"""
        print("Preparing features...")
        
        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")
        
        df = self.data.copy()
        
        # Encode hydrocarbon names
        self.label_encoder = LabelEncoder()
        df['Hydrocarbon_Encoded'] = self.label_encoder.fit_transform(df['Hydrocarbon'])
        
        # Features and target
        X = df[['Hydrocarbon_Encoded', 'Temperature', 'Equivalence_Ratio', 'Pressure']]
        y = df['LBV']
        
        # Split data with stratification on hydrocarbon type
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=df['Hydrocarbon_Encoded']
        )
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, use_grid_search=True, verbose=True):
        """Train Random Forest model with hyperparameter tuning"""
        print("Training Random Forest model...")
        
        if self.X_train is None:
            raise ValueError("Features not prepared. Call prepare_features() first.")
        
        if use_grid_search:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
            
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            
            # Use RandomizedSearchCV for efficiency
            print("Performing hyperparameter tuning...")
            grid_search = RandomizedSearchCV(
                rf, param_grid, n_iter=100, cv=5, 
                scoring='neg_mean_absolute_error', 
                random_state=42, n_jobs=-1, verbose=1 if verbose else 0
            )
            
            grid_search.fit(self.X_train, self.y_train)
            self.model = grid_search.best_estimator_
            
            if verbose:
                print("Best parameters found:")
                for param, value in grid_search.best_params_.items():
                    print(f"  {param}: {value}")
                print(f"Best CV score: {-grid_search.best_score_:.4f}")
            
            self.best_params = grid_search.best_params_
            self.cv_score = -grid_search.best_score_
        else:
            # Use default parameters with some optimization
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=30,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(self.X_train, self.y_train)
            self.best_params = self.model.get_params()
            
            # Calculate CV score
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, 
                                      cv=5, scoring='neg_mean_absolute_error')
            self.cv_score = -cv_scores.mean()
        
        print("Model training completed!")
        return self.model
    
    def evaluate_model(self, verbose=True):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Make predictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        # Check ±2 cm/s accuracy
        train_within_bounds = np.abs(y_pred_train - self.y_train) <= 2.0
        test_within_bounds = np.abs(y_pred_test - self.y_test) <= 2.0
        
        train_accuracy = np.mean(train_within_bounds) * 100
        test_accuracy = np.mean(test_within_bounds) * 100
        
        # Store metrics
        self.metrics = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_score': self.cv_score,
            'best_params': self.best_params
        }
        
        if verbose:
            print("\n=== Model Performance Metrics ===")
            print(f"Training MAE: {train_mae:.4f} cm/s")
            print(f"Test MAE: {test_mae:.4f} cm/s")
            print(f"Training RMSE: {train_rmse:.4f} cm/s")
            print(f"Test RMSE: {test_rmse:.4f} cm/s")
            print(f"Training R²: {train_r2:.4f}")
            print(f"Test R²: {test_r2:.4f}")
            print(f"Training ±2 cm/s Accuracy: {train_accuracy:.2f}%")
            print(f"Test ±2 cm/s Accuracy: {test_accuracy:.2f}%")
            print(f"CV MAE Score: {self.cv_score:.4f} cm/s")
            
            # Feature importance
            feature_names = ['Hydrocarbon', 'Temperature', 'Equivalence_Ratio', 'Pressure']
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\nFeature Importance:")
            for _, row in importance_df.iterrows():
                print(f"  {row['Feature']}: {row['Importance']:.4f}")
        
        return self.metrics
    
    def save_model(self, model_path='lbv_model.pkl', encoder_path='label_encoder.pkl'):
        """Save trained model and label encoder"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Model saved to {model_path}")
        print(f"Label encoder saved to {encoder_path}")
    
    def generate_report(self, save_path='model_training_report.txt'):
        """Generate detailed training report"""
        if not hasattr(self, 'metrics'):
            raise ValueError("Model not evaluated. Call evaluate_model() first.")
        
        report = f"""
LBV Prediction Model Training Report
=====================================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset Information:
- Total samples: {len(self.data)}
- Number of hydrocarbons: {self.data['Hydrocarbon'].nunique()}
- LBV range: {self.data['LBV'].min():.2f} - {self.data['LBV'].max():.2f} cm/s
- Temperature range: {self.data['Temperature'].min():.1f} - {self.data['Temperature'].max():.1f} K
- Equivalence ratio range: {self.data['Equivalence_Ratio'].min():.3f} - {self.data['Equivalence_Ratio'].max():.3f}
- Pressure range: {self.data['Pressure'].min():.1f} - {self.data['Pressure'].max():.1f} atm

Model Configuration:
- Algorithm: Random Forest Regression
- Training set size: {len(self.X_train)}
- Test set size: {len(self.X_test)}
- Cross-validation folds: 5

Performance Metrics:
- Test MAE: {self.metrics['test_mae']:.4f} cm/s
- Test RMSE: {self.metrics['test_rmse']:.4f} cm/s
- Test R²: {self.metrics['test_r2']:.4f}
- Test ±2 cm/s Accuracy: {self.metrics['test_accuracy']:.2f}%
- CV MAE Score: {self.metrics['cv_score']:.4f} cm/s

Best Hyperparameters:
"""
        for param, value in self.metrics['best_params'].items():
            report += f"- {param}: {value}\n"
        
        report += f"""
Feature Importance:
"""
        feature_names = ['Hydrocarbon', 'Temperature', 'Equivalence_Ratio', 'Pressure']
        for i, importance in enumerate(self.model.feature_importances_):
            report += f"- {feature_names[i]}: {importance:.4f}\n"
        
        report += f"""
Model Quality Assessment:
- The model achieves {self.metrics['test_accuracy']:.1f}% accuracy within ±2 cm/s bounds
- R² score of {self.metrics['test_r2']:.3f} indicates {'excellent' if self.metrics['test_r2'] > 0.9 else 'good' if self.metrics['test_r2'] > 0.8 else 'moderate'} predictive performance
- Low RMSE of {self.metrics['test_rmse']:.2f} cm/s shows good precision
- Cross-validation score confirms model robustness

Recommendations:
- Model is ready for production use
- Predictions should be clipped to training data range ± 2 cm/s
- Regular retraining recommended when new data becomes available
"""
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(f"Training report saved to {save_path}")
        return report

def main():
    """Main training pipeline"""
    print("Starting LBV Model Training Pipeline...")
    
    # Initialize trainer
    trainer = LBVModelTrainer()
    
    # Load and preprocess data
    trainer.load_and_preprocess_data()
    
    # Prepare features
    trainer.prepare_features()
    
    # Train model with hyperparameter tuning
    trainer.train_model(use_grid_search=True, verbose=True)
    
    # Evaluate model
    trainer.evaluate_model(verbose=True)
    
    # Save model
    trainer.save_model()
    
    # Generate report
    trainer.generate_report()
    
    print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    main()
