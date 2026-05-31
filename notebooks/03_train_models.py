"""
HealthBite - Notebook 3: Model Training
Run: python notebooks/03_train_models.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Libraries loaded successfully!")

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# ─────────────────────────────────────────────
# Cell 2: Load Cleaned Data
# ─────────────────────────────────────────────
df = pd.read_csv('data/cleaned/foods_clean.csv').dropna()
print(f"Loaded dataset: {df.shape}")

# ─────────────────────────────────────────────
# Cell 3: Create Health Labels
# ─────────────────────────────────────────────
def classify_healthy(row):
    """
    Classify food as healthy (1) or unhealthy (0)
    Based on WHO nutritional guidelines (adapted)
    """
    score = 0
    
    # Positive indicators (good for health)
    if pd.notnull(row.get('sugar')) and row['sugar'] < 10:
        score += 1
    if pd.notnull(row.get('sodium')) and row['sodium'] < 400:
        score += 1
    if pd.notnull(row.get('fiber')) and row['fiber'] > 3:
        score += 1
    if pd.notnull(row.get('calories')) and row['calories'] < 300:
        score += 1
    if pd.notnull(row.get('fat')) and row['fat'] < 10:
        score += 1
    if pd.notnull(row.get('protein')) and row['protein'] > 5:
        score += 1
    
    # Using threshold 3 since fiber is missing/zero in USDA.csv
    return 1 if score >= 3 else 0

df['healthy'] = df.apply(classify_healthy, axis=1)

print("\n" + "="*50)
print("HEALTH CLASSIFICATION DISTRIBUTION")
print("="*50)
healthy_count = df['healthy'].sum()
unhealthy_count = len(df) - healthy_count
healthy_pct = (healthy_count / len(df)) * 100

print(f"Healthy foods: {healthy_count} ({healthy_pct:.1f}%)")
print(f"Unhealthy foods: {unhealthy_count} ({100-healthy_pct:.1f}%)")

# ─────────────────────────────────────────────
# Cell 4: Prepare Training Data
# ─────────────────────────────────────────────
feature_cols = ['calories', 'protein', 'carbs', 'fat', 'sugar', 'sodium', 'saturated_fat']
# We replaced fiber with saturated_fat for the features since USDA.csv doesn't have it natively mapped well
X = df[feature_cols]
y = df['healthy']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ─────────────────────────────────────────────
# Cell 5: Scale Features
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─────────────────────────────────────────────
# Cell 6: Train Individual Models
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("TRAINING INDIVIDUAL MODELS")
print("="*50)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test_scaled))
print(f"KNN Accuracy: {knn_acc:.3f}")

svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
print(f"SVM Accuracy: {svm_acc:.3f}")

gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
gnb_acc = accuracy_score(y_test, gnb.predict(X_test_scaled))
print(f"Naive Bayes Accuracy: {gnb_acc:.3f}")

# ─────────────────────────────────────────────
# Cell 7: Train Ensemble Model
# ─────────────────────────────────────────────
ensemble = VotingClassifier(
    estimators=[('knn', knn), ('svm', svm), ('gnb', gnb)],
    voting='soft'
)
ensemble.fit(X_train_scaled, y_train)
y_pred = ensemble.predict(X_test_scaled)
ensemble_acc = accuracy_score(y_test, y_pred)

print(f"\n[TARGET] ENSEMBLE ACCURACY: {ensemble_acc:.3f} ({ensemble_acc*100:.1f}%)")

# ─────────────────────────────────────────────
# Cell 8: Detailed Evaluation
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['Unhealthy', 'Healthy']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Unhealthy', 'Healthy'],
            yticklabels=['Unhealthy', 'Healthy'])
plt.title('HealthBite - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('static/images/confusion_matrix.png', dpi=100)
plt.close()

# ─────────────────────────────────────────────
# Cell 10: Save Models
# ─────────────────────────────────────────────
pickle.dump(ensemble, open('models/classifier.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

# ─────────────────────────────────────────────
# Cell 11: Build Recommendation System
# ─────────────────────────────────────────────
food_features = df[feature_cols].values
food_names = df['name'].values

recommender_scaler = StandardScaler()
food_features_scaled = recommender_scaler.fit_transform(food_features)

knn_recommender = NearestNeighbors(n_neighbors=6, metric='euclidean')
knn_recommender.fit(food_features_scaled)

pickle.dump(knn_recommender, open('models/recommender.pkl', 'wb'))
pickle.dump(recommender_scaler, open('models/recommender_scaler.pkl', 'wb'))
pickle.dump(food_names, open('models/food_names.pkl', 'wb'))
pickle.dump(df[feature_cols], open('models/food_features.pkl', 'wb'))

print("\n" + "="*60)
print("[SUCCESS] MODEL TRAINING COMPLETE AND SAVED!")
print("="*60)
