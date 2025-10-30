# train_hourly_model.py
import pandas as pd
import numpy as np
from pathlib import Path

# --- ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, classification_report
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
df = pd.read_parquet(DATA_PATH)

# Zorunlu kolonlar
assert "Y_label" in df.columns, "Y_label kolonu yok."
assert "dt" in df.columns, "dt (saatlik zaman) kolonu yok."
assert "GEOID" in df.columns, "GEOID kolonu yok."

# -----------------------------
# 2) Özellik seçimi
# -----------------------------
# Kimlik/zaman sütunlarını dışla
drop_cols = {"Y_label", "dt", "GEOID"}
all_cols = [c for c in df.columns if c not in drop_cols]

# Basit bir kural: numerik ve kategorik kolonları ayır
num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in all_cols if (not pd.api.types.is_numeric_dtype(df[c]))]

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
y_train = train["Y_label"].astype(int)
X_test  = test[all_cols]
y_test  = test["Y_label"].astype(int)

# -----------------------------
# 4) Dengesizlik oranı → scale_pos_weight
# -----------------------------
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = max(1.0, neg / max(1, pos))  # güvenlik için max
print(f"[INFO] Train positive: {pos:,} | negative: {neg:,} | scale_pos_weight ≈ {scale_pos_weight:.2f}")

# -----------------------------
# 5) Pipeline (impute + OHE + model)
# -----------------------------
num_tf = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
])

cat_tf = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

pre = ColumnTransformer(
    transformers=[
        ("num", num_tf, num_cols),
        ("cat", cat_tf, cat_cols),
    ],
    remainder="drop"
)

if USE_XGBOOST:
    clf = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="aucpr",          # PR-AUC odaklı
        scale_pos_weight=scale_pos_weight,
        n_jobs=4,
        random_state=42,
    )
else:
    clf = LGBMClassifier(
        n_estimators=1200,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary",
        # LightGBM için iki yol:
        # 1) is_unbalance=True  (otomatik dengeleme)  -YA DA-
        # 2) scale_pos_weight=scale_pos_weight
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
proba = pipe.predict_proba(X_test)[:, 1]
ap = average_precision_score(y_test, proba)  # PR-AUC
print(f"\n[RESULT] PR-AUC (Average Precision): {ap:.4f}")

# En iyi F1 için threshold seç
prec, rec, th = precision_recall_curve(y_test, proba)
f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
best_idx = int(np.nanargmax(f1s))
best_th = th[max(0, best_idx-1)] if best_idx > 0 else 0.5  # PR-Curve shape farkından küçük kaydırma
y_pred_best = (proba >= best_th).astype(int)
f1_best = f1_score(y_test, y_pred_best)

print(f"[RESULT] Best-F1 threshold ≈ {best_th:.4f} | F1 = {f1_best:.4f}")
print("\nClassification report (best-threshold):")
print(classification_report(y_test, y_pred_best, digits=4))

# İsteğe bağlı: Recall@K (ör. her saat en riskli K GEOID'i seçmek istiyorsan
# ayrı bir değerlendirme metodu kurman gerekir; bu örnek genel sınıflandırma içindir.)
