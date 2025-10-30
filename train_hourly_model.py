#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_block_model.py
- aggregate_windows.py çıktısı ile (1D veya 8H) model eğitir
- Zaman bazlı split (son %20 test), sızıntısız
- Kategorikler: year, month, day_of_week, (opsiyonel) block_id
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

USE_XGBOOST = True
if USE_XGBOOST:
    from xgboost import XGBClassifier
else:
    from lightgbm import LGBMClassifier

# ---- Girdi (1D veya 8H)
DATA_PATH = Path("sf_crime_grid_8h.parquet")  # 8H deneme için; günlük ise sf_crime_grid_1d.parquet yap
if not DATA_PATH.exists():
    raise SystemExit(f"Girdi yok: {DATA_PATH.resolve()}")

df = pd.read_parquet(DATA_PATH)
req = {"GEOID","t0","Y_label","crime_count","year","month","day_of_week"}
missing = req - set(df.columns)
if missing:
    raise SystemExit(f"Gerekli kolonlar eksik: {missing}")

# ---- Özellikler
DROP = {"GEOID","t0","Y_label"}  # kimlik ve hedef dışı
CAT = ["year","month","day_of_week"]
if "block_id" in df.columns:
    CAT.append("block_id")  # 8H için

# Numerikler: kalan tüm numerikler (count/priors dahil)
num_cols = [c for c in df.columns if (c not in DROP) and (c not in CAT)
            and pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in CAT if c in df.columns]

print(f"[INFO] Numerik={len(num_cols)} | Kategorik={len(cat_cols)}")
df = df.sort_values("t0").reset_index(drop=True)

# ---- Split (zaman bazlı: son %20 test)
cut = int(len(df) * 0.80)
train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

X_train = train[cat_cols + num_cols]
y_train = train["Y_label"].astype(np.int8).values
X_test  = test[cat_cols + num_cols]
y_test  = test["Y_label"].astype(np.int8).values

# ---- Pipeline (impute + OHE[sparse] + model)
num_tf = Pipeline([("impute", SimpleImputer(strategy="median"))])
cat_tf = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
])

pre = ColumnTransformer(
    transformers=[
        ("num", num_tf, num_cols),
        ("cat", cat_tf, cat_cols),
    ],
    remainder="drop",
    sparse_threshold=1.0,
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
        is_unbalance=True,   # 1D/8H’de oran daha iyi olsa da güvenli
        n_jobs=4,
        random_state=42,
    )

pipe = Pipeline([("pre", pre), ("mdl", clf)])

# ---- Eğitim
pipe.fit(X_train, y_train)

# ---- Değerlendirme
proba = pipe.predict_proba(X_test)[:, 1].astype(np.float32)
ap = float(average_precision_score(y_test, proba))
prec, rec, th = precision_recall_curve(y_test, proba)
f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
best_idx = int(np.nanargmax(f1s))
best_th  = float(th[max(0, best_idx-1)] if best_idx > 0 else 0.5)
y_pred_best = (proba >= best_th).astype(np.int8)
f1_best = float(f1_score(y_test, y_pred_best))

print(f"[RESULT] PR-AUC: {ap:.4f} | Best-F1@{best_th:.3f} = {f1_best:.4f}")
print("Classification report:")
print(classification_report(y_test, y_pred_best, digits=4))

# ---- Kaydet
OUT_DIR = Path(".")
(OUT_DIR / "models").mkdir(exist_ok=True)
(OUT_DIR / "reports").mkdir(exist_ok=True)

metrics = {
    "data_file": str(DATA_PATH),
    "n_train": int(len(train)), "n_test": int(len(test)),
    "n_features_num": len(num_cols),
    "n_features_cat": len(cat_cols),
    "pr_auc": ap, "f1_best": f1_best, "best_threshold": best_th,
}
with open(OUT_DIR / "reports" / ("block_metrics_8h.json" if "block_id" in df.columns else "block_metrics_1d.json"),
          "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

try:
    import joblib
    joblib.dump(pipe, OUT_DIR / "models" / ("sutam_block_8h.joblib" if "block_id" in df.columns else "sutam_block_1d.joblib"))
except Exception as e:
    print(f"[WARN] Model kaydı başarısız: {e}")
