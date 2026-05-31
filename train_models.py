"""
HealthBite — Model Retraining Pipeline

This script demonstrates how the internal machine learning models are built.
It loads cleaned nutritional data, generates health labels based on a
heuristic scoring system (for demonstration purposes), and trains the
ensemble models used by the application.

Outputs:
  - models/classifier.pkl (VotingClassifier)
  - models/scaler.pkl (StandardScaler)
  - models/recommender.pkl (NearestNeighbors)
  - models/recommender_scaler.pkl (StandardScaler)
  - models/food_names.pkl (Pandas Series)
  - models/food_features.pkl (Pandas DataFrame)
"""

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

DATA_PATH = 'data/cleaned/foods_clean.csv'
MODELS_DIR = 'models'

# Feature columns must match ml_service.py FEATURE_COLS order exactly
FEATURE_COLS = ['calories', 'protein', 'carbs', 'fat', 'sugar', 'sodium', 'saturated_fat']

def generate_health_labels(df):
    """
    Generate synthetic 'Healthy' (1) vs 'Unhealthy' (0) labels.
    A real-world app would use dietitian-labeled data or strict WHO bounds.
    Heuristic: High sugar, high sodium, or high saturated fat = unhealthy.
    High protein = healthy buffer.
    """
    labels = []
    for _, row in df.iterrows():
        # Points for unhealthy traits (per 100g)
        penalty = 0
        if row['sugar'] > 15: penalty += 2
        if row['sugar'] > 25: penalty += 2
        if row['sodium'] > 400: penalty += 2
        if row['saturated_fat'] > 5: penalty += 2
        if row['fat'] > 20: penalty += 1
        
        # Buffer for healthy traits
        if row['protein'] > 10: penalty -= 1
        if row['fiber'] > 5: penalty -= 1
        
        # Final classification
        labels.append(1 if penalty < 3 else 0)
    
    return np.array(labels)

def train_and_save_models():
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}. Please ensure data exists.")
        
    df = pd.read_csv(DATA_PATH)
    
    # Ensure all feature columns exist, drop rows with NaNs
    df = df.dropna(subset=FEATURE_COLS + ['name'])
    
    print(f"Dataset size: {len(df)} foods.")
    
    # 1. Prepare data
    X = df[FEATURE_COLS].values
    y = generate_health_labels(df)
    
    # Check class balance
    healthy_count = np.sum(y == 1)
    unhealthy_count = len(y) - healthy_count
    print(f"Labels generated: {healthy_count} Healthy, {unhealthy_count} Unhealthy.")
    
    # 2. Train Classification Models
    print("Training classifiers...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    svm = SVC(kernel='rbf', probability=True, C=1.0)
    gnb = GaussianNB()
    
    ensemble = VotingClassifier(
        estimators=[('knn', knn), ('svm', svm), ('gnb', gnb)],
        voting='soft'
    )
    
    ensemble.fit(X_scaled, y)
    print(f"Ensemble training complete. Accuracy on training set: {ensemble.score(X_scaled, y):.3f}")
    
    # 3. Train Recommender (Nearest Neighbors)
    print("Training recommender...")
    rec_scaler = StandardScaler()
    # For recommendations, we might weight macros differently, but standard scaling is a good baseline
    X_rec_scaled = rec_scaler.fit_transform(X)
    
    recommender = NearestNeighbors(n_neighbors=10, metric='euclidean')
    recommender.fit(X_rec_scaled)
    
    # 4. Save Models
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Saving models to {MODELS_DIR}/...")
    
    def _save(obj, filename):
        with open(os.path.join(MODELS_DIR, filename), 'wb') as f:
            pickle.dump(obj, f)
            
    _save(ensemble, 'classifier.pkl')
    _save(scaler, 'scaler.pkl')
    _save(recommender, 'recommender.pkl')
    _save(rec_scaler, 'recommender_scaler.pkl')
    _save(df['name'].reset_index(drop=True), 'food_names.pkl')
    _save(df[FEATURE_COLS].reset_index(drop=True), 'food_features.pkl')
    
    print("Pipeline complete!")

if __name__ == '__main__':
    train_and_save_models()
