import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
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

# Train model with good default parameters
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

print('Training model...')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
within_2cm = np.mean(np.abs(y_pred - y_test) <= 2.0) * 100

print(f'MAE: {mae:.2f} cm/s')
print(f'R²: {r2:.3f}')
print(f'±2cm/s accuracy: {within_2cm:.1f}%')

# Save model files
with open('lbv_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print('Model saved successfully!')