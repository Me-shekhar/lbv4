import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

class LBVPredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.data = None
        self.hydrocarbon_ranges = {}
        self.load_data()
        self.load_or_train_model()
    
    def load_data(self):
        """Load and preprocess the dataset"""
        try:
            # Load data
            df = pd.read_csv('data.csv')
            
            # Clean column names and remove empty columns
            columns_to_drop = [col for col in df.columns if col.startswith('Unnamed')]
            df = df.drop(columns=columns_to_drop)
            
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
            
            # Store hydrocarbon ranges for validation
            self.hydrocarbon_ranges = {}
            for hydrocarbon in df['Hydrocarbon'].unique():
                hc_data = df[df['Hydrocarbon'] == hydrocarbon]
                self.hydrocarbon_ranges[hydrocarbon] = {
                    'temperature': {'min': hc_data['Temperature'].min(), 'max': hc_data['Temperature'].max()},
                    'equiv_ratio': {'min': hc_data['Equivalence_Ratio'].min(), 'max': hc_data['Equivalence_Ratio'].max()},
                    'pressure': {'min': hc_data['Pressure'].min(), 'max': hc_data['Pressure'].max()}
                }
            
            self.data = df
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def load_or_train_model(self):
        """Load existing model or train a new one"""
        if os.path.exists('lbv_model.pkl') and os.path.exists('label_encoder.pkl'):
            try:
                with open('lbv_model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                with open('label_encoder.pkl', 'rb') as f:
                    self.label_encoder = pickle.load(f)
                return True
            except:
                pass
        
        # Train new model
        return self.train_model()
    
    def train_model(self):
        """Train Random Forest model with hyperparameter tuning"""
        if self.data is None:
            return False
        
        try:
            # Prepare features
            df = self.data.copy()
            
            # Encode hydrocarbon names
            self.label_encoder = LabelEncoder()
            df['Hydrocarbon_Encoded'] = self.label_encoder.fit_transform(df['Hydrocarbon'])
            
            # Features and target
            X = df[['Hydrocarbon_Encoded', 'Temperature', 'Equivalence_Ratio', 'Pressure']]
            y = df['LBV']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=df['Hydrocarbon_Encoded']
            )
            
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            
            # Use RandomizedSearchCV for efficiency
            from sklearn.model_selection import RandomizedSearchCV
            grid_search = RandomizedSearchCV(
                rf, param_grid, n_iter=50, cv=5, 
                scoring='neg_mean_absolute_error', 
                random_state=42, n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Check ±2 cm/s accuracy
            within_bounds = np.abs(y_pred - y_test) <= 2.0
            accuracy_percent = np.mean(within_bounds) * 100
            
            # Save model and encoder
            with open('lbv_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            with open('label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            # Store metrics
            self.model_metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'accuracy_percent': accuracy_percent,
                'best_params': grid_search.best_params_
            }
            
            return True
            
        except Exception as e:
            st.error(f"Error training model: {e}")
            return False
    
    def get_hydrocarbon_options(self):
        """Get list of available hydrocarbons"""
        if self.data is not None:
            return sorted(self.data['Hydrocarbon'].unique())
        return []
    
    def get_hydrocarbon_ranges(self, hydrocarbon):
        """Get valid ranges for a specific hydrocarbon"""
        return self.hydrocarbon_ranges.get(hydrocarbon, {
            'temperature': {'min': 300, 'max': 750},
            'equiv_ratio': {'min': 0.1, 'max': 2.4},
            'pressure': {'min': 1, 'max': 10}
        })
    
    def validate_inputs(self, hydrocarbon, temperature, equiv_ratio, pressure):
        """Validate input parameters"""
        errors = []
        
        if hydrocarbon not in self.get_hydrocarbon_options():
            errors.append(f"Invalid hydrocarbon: {hydrocarbon}")
            return errors
        
        ranges = self.get_hydrocarbon_ranges(hydrocarbon)
        
        if not (ranges['temperature']['min'] <= temperature <= ranges['temperature']['max']):
            errors.append(f"Temperature must be between {ranges['temperature']['min']:.1f} and {ranges['temperature']['max']:.1f} K for {hydrocarbon}")
        
        if not (ranges['equiv_ratio']['min'] <= equiv_ratio <= ranges['equiv_ratio']['max']):
            errors.append(f"Equivalence ratio must be between {ranges['equiv_ratio']['min']:.2f} and {ranges['equiv_ratio']['max']:.2f} for {hydrocarbon}")
        
        if not (ranges['pressure']['min'] <= pressure <= ranges['pressure']['max']):
            errors.append(f"Pressure must be between {ranges['pressure']['min']:.1f} and {ranges['pressure']['max']:.1f} atm for {hydrocarbon}")
        
        return errors
    
    def predict_lbv(self, hydrocarbon, temperature, equiv_ratio, pressure):
        """Predict LBV with clipping to ±2 cm/s bounds"""
        if self.model is None or self.label_encoder is None:
            return None
        
        try:
            # Encode hydrocarbon
            hydrocarbon_encoded = self.label_encoder.transform([hydrocarbon])[0]
            
            # Prepare input
            input_data = np.array([[hydrocarbon_encoded, temperature, equiv_ratio, pressure]])
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            
            # Apply clipping based on training data range
            y_min, y_max = self.data['LBV'].min(), self.data['LBV'].max()
            clipped_prediction = np.clip(prediction, y_min - 2, y_max + 2)
            
            return clipped_prediction
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None
    
    def get_model_metrics(self):
        """Get model performance metrics"""
        if hasattr(self, 'model_metrics'):
            return self.model_metrics
        return None

def convert_units(value, from_unit, to_unit):
    """Convert between cm/s and m/s"""
    if from_unit == "cm/s" and to_unit == "m/s":
        return value / 100
    elif from_unit == "m/s" and to_unit == "cm/s":
        return value * 100
    return value

def get_input_ranges():
    """Get general input ranges"""
    return {
        'temperature': {'min': 300.0, 'max': 750.0},
        'equiv_ratio': {'min': 0.1, 'max': 2.4},
        'pressure': {'min': 1.0, 'max': 10.0}
    }

def format_prediction_result(prediction, unit="cm/s"):
    """Format prediction result with appropriate precision"""
    if prediction is None:
        return "Error in prediction"
    return f"{prediction:.2f} {unit}"

def get_model_info():
    """Get general model information"""
    return {
        'algorithm': 'Random Forest Regression',
        'dataset_size': '5000+ data points',
        'features': 4,
        'hydrocarbon_types': 46,
        'accuracy_target': '±2 cm/s'
    }

def initialize_model():
    """Initialize the model and return success status"""
    try:
        predictor = LBVPredictor()
        return predictor.model is not None
    except:
        return False
