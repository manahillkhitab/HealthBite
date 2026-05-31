"""
HealthBite — ML Service Layer
Encapsulates all model inference logic, keeping app.py clean.
"""
import os
import pickle

import numpy as np

# ── Feature columns (must match training order) ──────────────────
FEATURE_COLS = ['calories', 'protein', 'carbs', 'fat', 'sugar', 'sodium', 'saturated_fat']

# ── Per-feature validation bounds (per 100g serving) ─────────────
FEATURE_BOUNDS = {
    'calories':      (0,   900),   # kcal  — pure fat is ~900 kcal/100g
    'protein':       (0,   100),   # g
    'carbs':         (0,   100),   # g
    'fat':           (0,   100),   # g
    'sugar':         (0,   100),   # g
    'sodium':        (0, 10000),   # mg    — very salty processed foods ~6g
    'saturated_fat': (0,   100),   # g
}

# Human-readable labels for the three ensemble sub-models
MODEL_LABELS = {
    'knn': 'KNN Classifier',
    'svm': 'Support Vector Machine',
    'gnb': 'Naive Bayes',
}

# ── Load models once at import time ──────────────────────────────
def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

try:
    classifier        = _load('models/classifier.pkl')
    scaler            = _load('models/scaler.pkl')
    recommender       = _load('models/recommender.pkl')
    recommender_scaler= _load('models/recommender_scaler.pkl')
    food_names        = _load('models/food_names.pkl')
    food_features     = _load('models/food_features.pkl')
    print("[OK] ML Service: all models loaded successfully")
except Exception as e:
    print(f"[ERROR] ML Service failed to load models: {e}")
    raise


# ── Public API ────────────────────────────────────────────────────

def validate_nutrition(data: dict) -> list[str]:
    """
    Validate raw form data against physiological bounds.
    Returns a list of human-readable error strings (empty = all good).
    """
    errors = []
    for field, (lo, hi) in FEATURE_BOUNDS.items():
        try:
            val = float(data.get(field, 0))
        except (TypeError, ValueError):
            errors.append(f"'{field}' must be a valid number.")
            continue
        if not (lo <= val <= hi):
            unit = 'mg' if field == 'sodium' else ('kcal' if field == 'calories' else 'g')
            errors.append(f"'{field}' must be between {lo} and {hi} {unit} per 100g.")
    return errors


def extract_features(data: dict) -> np.ndarray:
    """Convert validated form dict → 2D numpy array in training order."""
    return np.array([[float(data[col]) for col in FEATURE_COLS]])


def predict_health(features: np.ndarray) -> dict:
    """
    Run the ensemble classifier + individual sub-model votes.
    Returns a dict with prediction, confidence, and individual_models list.
    """
    features_scaled = scaler.transform(features)

    # Ensemble prediction
    raw_pred   = classifier.predict(features_scaled)[0]
    proba      = classifier.predict_proba(features_scaled)[0]
    confidence = round(float(max(proba)) * 100, 1)
    prediction = 'Healthy' if raw_pred == 1 else 'Unhealthy'

    # Per-model votes
    individual_models = []
    for key, estimator in classifier.named_estimators_.items():
        try:
            p = estimator.predict_proba(features_scaled)[0]
            pct = round(float(p[1]) * 100, 1)       # prob of Healthy
        except Exception:
            pct = confidence
        individual_models.append({
            'label':   MODEL_LABELS.get(key, key.upper()),
            'confidence': pct,
            'verdict': 'Healthy' if pct >= 50 else 'Unhealthy',
        })

    return {
        'prediction':       prediction,
        'confidence':       confidence,
        'individual_models': individual_models,
    }


def get_recommendations(features: np.ndarray, n: int = 5) -> list[dict]:
    """
    Find the n most nutritionally similar foods using KNN recommender.
    Returns a list of dicts with name, similarity %, and macro highlights.
    """
    features_scaled = recommender_scaler.transform(features)
    distances, indices = recommender.kneighbors(features_scaled, n_neighbors=n + 1)

    recs = []
    for i in range(n):
        idx        = indices[0][i]
        similarity = round((1 / (1 + distances[0][i])) * 100, 1)
        recs.append({
            'name':          str(food_names[idx]),
            'similarity':    similarity,
            'calories':      round(float(food_features.iloc[idx]['calories']), 1),
            'protein':       round(float(food_features.iloc[idx]['protein']), 1),
            'sugar':         round(float(food_features.iloc[idx]['sugar']), 1),
            'saturated_fat': round(float(food_features.iloc[idx]['saturated_fat']), 1),
        })
    return recs
