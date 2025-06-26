import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load and clean data
df = pd.read_csv('data.csv')
df = df.drop(columns=[col for col in df.columns if df[col].isna().all() or col.startswith('Unnamed')])
df = df.rename(columns={
    'Ti (K)': 'Temperature',
    'equivalent ratio': 'Equivalence_Ratio', 
    'Pressure (atm)': 'Pressure',
    'LBV (cm/s)': 'LBV'
})
df = df.dropna().drop_duplicates()

print(f'Dataset: {df.shape[0]} samples, {df["Hydrocarbon"].nunique()} hydrocarbons')
print(f'LBV range: {df["LBV"].min():.1f} - {df["LBV"].max():.1f} cm/s')

# Encode hydrocarbons
le = LabelEncoder()
df['Hydrocarbon_Encoded'] = le.fit_transform(df['Hydrocarbon'])

# Features and target
X = df[['Hydrocarbon_Encoded', 'Temperature', 'Equivalence_Ratio', 'Pressure']]
y = df['LBV']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['Hydrocarbon_Encoded']
)

# Hyperparameter tuning with focus on accuracy
print('Optimizing model for ±2 cm/s accuracy...')
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [15, 20, 25],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 0.7, 0.8]
}

# Custom scorer for ±2 cm/s accuracy
def accuracy_within_2cm(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) <= 2.0)

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf, param_grid, cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1, verbose=1
)

grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

print(f'Best parameters: {grid_search.best_params_}')

# Evaluate model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

train_accuracy = np.mean(np.abs(y_pred_train - y_train) <= 2.0) * 100
test_accuracy = np.mean(np.abs(y_pred_test - y_test) <= 2.0) * 100

print(f'\nTraining Results:')
print(f'Train MAE: {train_mae:.2f} cm/s')
print(f'Test MAE: {test_mae:.2f} cm/s')
print(f'Train RMSE: {train_rmse:.2f} cm/s')
print(f'Test RMSE: {test_rmse:.2f} cm/s')
print(f'Train R²: {train_r2:.3f}')
print(f'Test R²: {test_r2:.3f}')
print(f'Train ±2cm/s accuracy: {train_accuracy:.1f}%')
print(f'Test ±2cm/s accuracy: {test_accuracy:.1f}%')

# Feature importance
feature_names = ['Hydrocarbon', 'Temperature', 'Equivalence_Ratio', 'Pressure']
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f'\nFeature Importance:')
for _, row in importance_df.iterrows():
    print(f'{row["Feature"]}: {row["Importance"]:.3f}')

# Save model and metrics
model_metrics = {
    'mae': test_mae,
    'rmse': test_rmse,
    'r2': test_r2,
    'accuracy_percent': test_accuracy,
    'best_params': grid_search.best_params_
}

with open('lbv_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
with open('model_metrics.pkl', 'wb') as f:
    pickle.dump(model_metrics, f)

print('\nModel and metrics saved successfully!')
print(f'Model meets ±2 cm/s requirement: {"✓ YES" if test_accuracy >= 80 else "✗ NO"} ({test_accuracy:.1f}% accuracy)')