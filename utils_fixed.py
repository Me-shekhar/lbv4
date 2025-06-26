import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
            
            # Remove Unnamed columns and empty columns
            df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
            df = df.dropna(axis=1, how='all')
            
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
            print(f"Data loaded: {len(df)} samples, {df['Hydrocarbon'].nunique()} hydrocarbons")
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
                print("Model loaded successfully")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                pass
        
        # Train new model if loading failed
        return self.train_model()
    
    def train_model(self):
        """Train Random Forest model"""
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
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            within_bounds = np.mean(np.abs(y_pred - y_test) <= 2.0) * 100
            
            # Store metrics
            self.model_metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'accuracy_percent': within_bounds,
                'best_params': self.model.get_params()
            }
            
            # Save model and encoder
            with open('lbv_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            with open('label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            print(f"Model trained: MAE={mae:.2f}, R²={r2:.3f}, ±2cm/s accuracy={within_bounds:.1f}%")
            return True
            
        except Exception as e:
            st.error(f"Error training model: {e}")
            return False
    
    def get_hydrocarbon_options(self):
        """Get list of available hydrocarbons"""
        if self.data is not None:
            return sorted(self.data['Hydrocarbon'].unique().tolist())
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
        """Predict LBV with clipping to reasonable bounds"""
        if self.model is None or self.label_encoder is None:
            return None
        
        try:
            # Encode hydrocarbon
            hydrocarbon_encoded = self.label_encoder.transform([hydrocarbon])[0]
            
            # Prepare input
            input_data = np.array([[hydrocarbon_encoded, temperature, equiv_ratio, pressure]])
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            
            # Apply reasonable clipping based on training data range
            if self.data is not None:
                y_min, y_max = self.data['LBV'].min(), self.data['LBV'].max()
                clipped_prediction = np.clip(prediction, y_min - 2, y_max + 2)
            else:
                clipped_prediction = max(0, prediction)  # Ensure non-negative
            
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
        'dataset_size': '1000+ data points',
        'features': 4,
        'hydrocarbon_types': 23,
        'accuracy_target': '±2 cm/s'
    }

def initialize_model():
    """Initialize the model and return success status"""
    try:
        predictor = LBVPredictor()
        return predictor.model is not None and len(predictor.get_hydrocarbon_options()) > 0
    except Exception as e:
        print(f"Model initialization error: {e}")
        return False