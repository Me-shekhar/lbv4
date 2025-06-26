# Laminar Burning Velocity (LBV) Prediction Web App

## Overview

This is a Streamlit web application that predicts Laminar Burning Velocity (LBV) for hydrocarbon-air mixtures using machine learning. The app uses a Random Forest regression model trained on experimental data to predict LBV values based on input parameters like temperature, equivalence ratio, and pressure.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework
- **UI Components**: Modern teal/cyan themed interface with custom CSS
- **Visualization**: Plotly for interactive charts and graphs
- **Layout**: Wide layout with collapsible sidebar for better user experience

### Backend Architecture
- **Model**: Random Forest Regressor (scikit-learn)
- **Data Processing**: Pandas and NumPy for data manipulation
- **Model Training**: Automated training pipeline with hyperparameter tuning
- **Prediction Engine**: Custom LBVPredictor class with validation and bounds checking

### Data Storage
- **Primary Data**: CSV file (`data.csv`) containing hydrocarbon experimental data
- **Model Persistence**: Pickle format for trained model storage
- **Configuration**: TOML files for project dependencies and Streamlit settings

## Key Components

### 1. Main Application (`app.py`)
- Streamlit web interface with custom CSS styling
- Input forms for user parameters (temperature, equivalence ratio, pressure)
- Real-time prediction display with interactive visualizations
- Model performance metrics and validation information

### 2. Utility Module (`utils.py`)
- `LBVPredictor` class: Core prediction engine
- Data preprocessing and validation functions
- Model loading and training orchestration
- Input range validation and unit conversion utilities

### 3. Model Training (`model_trainer.py`)
- `LBVModelTrainer` class: Handles complete training pipeline
- Data preprocessing with cleaning and feature engineering
- Hyperparameter optimization using GridSearchCV/RandomizedSearchCV
- Model evaluation and performance metrics calculation

### 4. Data Processing
- Automatic handling of missing values and duplicates
- Label encoding for categorical variables (hydrocarbon types)
- Feature scaling and normalization
- Input validation against training data ranges

## Data Flow

1. **Data Loading**: CSV data is loaded and preprocessed on application startup
2. **Model Initialization**: Pre-trained model is loaded or training is triggered if model doesn't exist
3. **User Input**: Users provide temperature, equivalence ratio, and pressure values
4. **Validation**: Input parameters are validated against acceptable ranges
5. **Prediction**: Random Forest model generates LBV prediction
6. **Post-processing**: Results are formatted and bounds-checked (±2 cm/s constraint)
7. **Visualization**: Interactive charts display prediction results and confidence intervals

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and tools
- **plotly**: Interactive visualization library

### Additional Dependencies
- **matplotlib**: Static plotting (fallback visualizations)
- **seaborn**: Statistical data visualization
- **pickle**: Model serialization

### System Dependencies (Nix)
- **Python 3.11**: Runtime environment
- **Cairo, FFmpeg, FreeType**: Graphics and media processing
- **GTK3, Gobject-introspection**: GUI toolkit support
- **Qhull, TCL/TK**: Scientific computing and GUI libraries

## Deployment Strategy

### Replit Configuration
- **Environment**: Python 3.11 with Nix package manager
- **Auto-scaling**: Configured for automatic scaling based on demand
- **Port Configuration**: Streamlit server runs on port 5000
- **Workflow**: Parallel execution with dedicated Streamlit app workflow

### Streamlit Configuration
- **Server**: Headless mode for deployment
- **Theme**: Dark theme with custom teal/cyan color scheme
- **Network**: Bound to all interfaces (0.0.0.0) for external access

### Model Persistence
- **Training**: Automatic model training on first run if no saved model exists
- **Caching**: Streamlit caching for data loading and model initialization
- **Validation**: Input validation ensures predictions stay within acceptable bounds

## Changelog
- June 26, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.