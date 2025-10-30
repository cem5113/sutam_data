# train_hourly_model.py
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --- ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    classification_report
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ---- MODEL SEÇ (XGBoost veya LightGBM)
USE_XGBOOST = True
if USE_XGBOOST:
    from xgboost import XGBClassifier
else:
    from lightgbm import LGBMClassifier

# -----------------------------
# 1) Veri yükleme
# -----------------------------
DATA_PATH = Path("sf_crime_grid_full_labeled.parquet")
if not DATA_PATH.exists():
    raise SystemExit(f"Input yok: {DATA_PATH.resolve()}")

df = pd.read_parquet(DATA_PATH)

# Zorunlu kolonlar
for col in ["Y_label", "dt", "GEOID"]:
    if col not in df.columns:
        raise SystemExit(f"Gerekli kolon eksik: {col}")

# -----------------------------
# 2) Özellik seçimi
# -----------------------------
# Kimlik/zaman sütunlarını dışla
drop_cols = {"Y_label", "dt", "GEOID"}
all_cols = [c for c in df.columns if c not in drop_cols]

# Numerik ve kategorik ayrımı
num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in all_cols if not pd.api.types.is_numeric_dtype(df[c])]

print(f"Numerik: {len(num_cols)} | Kategorik: {len(cat_cols)}")

# -----------------------------
# 3) Zaman bazlı split (leakage güvenli)
#    Son %20'yi test yapalım
# -----------------------------
df = df.sort_values("dt").reset_index(drop=True)
cut_idx = int(len(df) * 0.8)
train = df.iloc[:cut_idx].copy()
test  = df.iloc[cut_idx:].copy()

X_train = train[all_cols]
y_train = train["Y_label"].astype(np.int8)
X_test  = test[all_cols]
y_test  = test["Y_label"].astype(np.int8)

# -----------------------------
# 4) Dengesizlik oranı → scale_pos_weight
# -----------------------------
pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
scale_pos_weight = float(max(1.0, neg / max(1, pos)))
print(f"[INFO] Train positive: {pos:,} | negative: {neg:,} | scale_pos_weight ≈ {scale_pos_weight:.2f}")

# -----------------------------
# 5) Pipeline (impute + OHE + model)  — SPARSE GÜVENCESİ
# -----------------------------
steps_num = [("impute", SimpleImputer(strategy="median"))]
num_tf = Pipeline(steps=steps_num)

steps_cat = [
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
]
cat_tf = Pipeline(steps=steps_cat)

transformers = []
if num_cols:
    transformers.append(("num", num_tf, num_cols))
if cat_cols:
    transformers.append(("cat", cat_tf, cat_cols))

pre = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
    sparse_threshold=1.0,   # <— Asla densify etme
    n_jobs=None
)

if USE_XGBOOST:
    clf = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        n_jobs=4,
        random_state=42,
    )
else:
    clf = LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary",
        is_unbalance=False,
        scale_pos_weight=scale_pos_weight,
        n_jobs=4,
        random_state=42,
    )

pipe = Pipeline([
    ("pre", pre),
    ("mdl", clf),
])

# -----------------------------
# 6) Eğitim
# -----------------------------
pipe.fit(X_train, y_train)

# -----------------------------
# 7) Değerlendirme (PR-AUC, F1@best_th, rapor)
# -----------------------------
proba = pipe.predict_proba(X_test)[:, 1].astype(np.float32)
ap = float(average_precision_score(y_test, proba))  # PR-AUC
print(f"\n[RESULT] PR-AUC (Average Precision): {ap:.4f}")

# En iyi F1 için threshold seç
prec, rec, th = precision_recall_curve(y_test, proba)
f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
best_idx = int(np.nanargmax(f1s))
best_th = float(th[max(0, best_idx-1)] if best_idx > 0 else 0.5)
y_pred_best = (proba >= best_th).astype(np.int8)
f1_best = float(f1_score(y_test, y_pred_best))

print(f"[RESULT] Best-F1 threshold ≈ {best_th:.4f} | F1 = {f1_best:.4f}")
print("\nClassification report (best-threshold):")
print(classification_report(y_test, y_pred_best, digits=4))

# -----------------------------
# 8) Çıktıları kaydet (opsiyonel)
# -----------------------------
OUT_DIR = Path(".")
(OUT_DIR / "models").mkdir(exist_ok=True)
(OUT_DIR / "reports").mkdir(exist_ok=True)

# Metrikler
metrics = {
    "pr_auc": ap,
    "f1_best": f1_best,
    "best_threshold": best_th,
    "positives_train": pos,
    "negatives_train": neg,
    "scale_pos_weight": scale_pos_weight,
}
with open(OUT_DIR / "reports" / "hourly_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# Öznitelik isimleri
try:
    feat_names = pipe.named_steps["pre"].get_feature_names_out().tolist()
    pd.Series(feat_names, name="feature").to_csv(OUT_DIR / "reports" / "feature_names.csv", index=False)
except Exception:
    pass

# Modeli kaydet (joblib opsiyonel)
try:
    import joblib
    joblib.dump(pipe, OUT_DIR / "models" / ("sutam_hourly_xgb.joblib" if USE_XGBOOST else "sutam_hourly_lgbm.joblib"))
except Exception as e:
    print(f"[WARN] Model kaydedilemedi: {e}")
