"""
Script ini melakukan training Random Forest Classifier untuk prediksi diabetes
dan menyimpan semua artifact (model, encoder, metrik, dsb) yang dipakai oleh
dashboard Streamlit. Logika di sini PERSIS SAMA dengan yang ada di notebook
Google Colab (EAS_Diabetes_RandomForest.ipynb) supaya hasilnya konsisten.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

RANDOM_STATE = 42

# 1. LOAD DATA --------------------------------------------------------------
df = pd.read_csv("data/Diabetes_Prediction.csv", sep=";")

# 2. ENCODING KATEGORIKAL (mapping manual -> transparan & gampang dipakai lagi di Streamlit)
GENDER_MAP = {"Female": 0, "Male": 1, "Other": 2}
SMOKING_MAP = {"No Info": 0, "never": 1, "former": 2, "current": 3, "not current": 4, "ever": 5}

df_encoded = df.copy()
df_encoded["gender"] = df_encoded["gender"].map(GENDER_MAP)
df_encoded["smoking_history"] = df_encoded["smoking_history"].map(SMOKING_MAP)

FEATURE_COLUMNS = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"
]
TARGET_COLUMN = "diabetes"

X = df_encoded[FEATURE_COLUMNS]
y = df_encoded[TARGET_COLUMN]

# 3. SPLIT DATA ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 4. HYPERPARAMETER TUNING (RandomizedSearchCV) --------------------------------
param_dist = {
    "n_estimators": [150, 200, 300],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

base_model = RandomForestClassifier(
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)

search = RandomizedSearchCV(
    base_model, param_distributions=param_dist, n_iter=10,
    scoring="roc_auc", cv=3, random_state=RANDOM_STATE, n_jobs=-1, verbose=1
)
search.fit(X_train, y_train)
best_model = search.best_estimator_
print("Best params:", search.best_params_)

# 5. EVALUASI -------------------------------------------------------------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
    "best_params": search.best_params_,
    "n_train": len(X_train),
    "n_test": len(X_test),
}
print(json.dumps(metrics, indent=2))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred).tolist()
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

feature_importance = dict(zip(FEATURE_COLUMNS, best_model.feature_importances_.tolist()))
feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

# 6. SIMPAN SEMUA ARTIFACT -------------------------------------------------------
out = "artifacts"
joblib.dump(best_model, f"{out}/rf_model.pkl")
joblib.dump(FEATURE_COLUMNS, f"{out}/feature_columns.pkl")

with open(f"{out}/encoders.json", "w") as f:
    json.dump({"gender": GENDER_MAP, "smoking_history": SMOKING_MAP}, f, indent=2)

with open(f"{out}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open(f"{out}/confusion_matrix.json", "w") as f:
    json.dump(cm, f, indent=2)

with open(f"{out}/roc_curve.json", "w") as f:
    json.dump(roc_data, f, indent=2)

with open(f"{out}/feature_importance.json", "w") as f:
    json.dump(feature_importance, f, indent=2)

print("\nSemua artifact berhasil disimpan di:", out)
