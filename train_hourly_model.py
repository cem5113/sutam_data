# train_hourly_model.py  — SUTAM (GEOID×hour) model (revize)
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ── ML
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve, classification_report
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ── MODEL: XGBoost (hist) veya LightGBM
USE_XGBOOST = True
if USE_XGBOOST:
    from xgboost import XGBClassifier
else:
    from lightgbm import LGBMClassifier


# =========================
# 1) Veri yükle
# =========================
DATA_PATH = Path("sf_crime_grid_full_labeled.parquet")
if not DATA_PATH.exists():
    raise SystemExit(f"Input yok: {DATA_PATH.resolve()}")
df = pd.read_parquet(DATA_PATH)

# Zorunlu kolonlar
for col in ["Y_label", "dt", "GEOID"]:
    if col not in df.columns:
        raise SystemExit(f"Gerekli kolon eksik: {col}")

# Y_label dağılımını bilgi amaçlı yaz
y_rate = df["Y_label"].mean()
print(f"[INFO] Y_label oranı (1'ler): {100*y_rate:.4f}%  |  0'lar: {100*(1-y_rate):.4f}%")
print(f"[INFO] Satır: {len(df):,}  GEOID: {df['GEOID'].nunique():,}  Saat aralığı: {df['dt'].min()} → {df['dt'].max()}")


# =========================
# 2) Özellik seçimi (sızıntısız)
# =========================
# — DÜŞÜR (kullanma): kimlik/zaman + sızıntı/aynı-saat sayaçları
DROP = {"Y_label", "dt", "GEOID", "dt_local", "crime_count", "hr_cnt", "daily_cnt"}

# — KATEGORİK (OHE)
CAT_COLS = ["year", "month", "day_of_week", "hour", "season"]

# — SAYISAL (güvenli)
NUM_COLS = [
    # Komşuluk / çağrı geçmişi
    "neighbor_crime_24h", "neighbor_crime_72h", "neighbor_crime_7d",
    "911_geo_hr_last3d", "911_geo_hr_last7d", "311_request_count",
    # Hava
    "wx_tavg", "wx_tmin", "wx_tmax", "wx_prcp", "wx_temp_range",
    "wx_is_rainy", "wx_is_hot_day",
    # POI / mesafeler / yoğunluk
    "poi_total_count", "poi_risk_score",
    "poi_count_300m", "poi_risk_300m",
    "poi_count_600m", "poi_risk_600m",
    "poi_count_900m", "poi_risk_900m",
    "distance_to_police", "distance_to_government_building",
    "bus_stop_count", "train_stop_count", "population",
    # Priors (geçmişe bakan)
    "prior_cnt_3m", "prior_p_3m", "prior_cnt_12m", "prior_p_12m",
]

# Sadece var olanları kullan
present = set(df.columns)
cat_cols = [c for c in CAT_COLS if c in present]
num_cols = [c for c in NUM_COLS if c in present]
all_cols = cat_cols + num_cols
if not all_cols:
    raise SystemExit("Kullanılabilir özellik bulunamadı (all_cols boş).")

print(f"[INFO] Özellik sayısı → numerik: {len(num_cols)} | kategorik: {len(cat_cols)}")
print(f"[INFO] Düşülen kolonlar: {sorted(DROP.intersection(present))}")

# =========================
# 3) Zaman bazlı split (leakage güvenli)
# =========================
df = df.sort_values("dt").reset_index(drop=True)
cut_idx = int(len(df) * 0.80)
train = df.iloc[:cut_idx].copy()
test  = df.iloc[cut_idx:].copy()

X_train = train[all_cols].copy()
y_train = train["Y_label"].astype(np.int8)
X_test  = test[all_cols].copy()
y_test  = test["Y_label"].astype(np.int8)

# Küçük güvenlik: kategorikleri category dtype yap
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_test[c]  = X_test[c].astype("category")

# =========================
# 4) Dengesizlik → scale_pos_weight
# =========================
pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
scale_pos_weight = float(max(1.0, neg / max(1, pos)))
print(f"[INFO] Train positive: {pos:,} | negative: {neg:,} | scale_pos_weight ≈ {scale_pos_weight:.2f}")

# =========================
# 5) Pipeline (impute + OHE[sparse] + model)
# =========================
num_tf = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
])

cat_tf = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    # DİKKAT: sparse_output=True ve dtype=float32 → densify etme
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
])

transformers = []
if num_cols: transformers.append(("num", num_tf, num_cols))
if cat_cols: transformers.append(("cat", cat_tf, cat_cols))

pre = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
    sparse_threshold=1.0,  # asla dense'e dönme
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

# =========================
# 6) Eğitim
# =========================
pipe.fit(X_train, y_train)

# =========================
# 7) Değerlendirme
# =========================
proba = pipe.predict_proba(X_test)[:, 1].astype(np.float32)
ap = float(average_precision_score(y_test, proba))  # PR-AUC
print(f"\n[RESULT] PR-AUC (Average Precision): {ap:.4f}")

# En iyi F1 threshold
prec, rec, th = precision_recall_curve(y_test, proba)
f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
best_idx = int(np.nanargmax(f1s))
best_th = float(th[max(0, best_idx-1)] if best_idx > 0 else 0.5)
y_pred_best = (proba >= best_th).astype(np.int8)
f1_best = float(f1_score(y_test, y_pred_best))
print(f"[RESULT] Best-F1 threshold ≈ {best_th:.4f} | F1 = {f1_best:.4f}\n")

print("Classification report (best-threshold):")
print(classification_report(y_test, y_pred_best, digits=4))

# =========================
# 8) Çıktılar
# =========================
OUT_DIR = Path(".")
(OUT_DIR / "models").mkdir(exist_ok=True)
(OUT_DIR / "reports").mkdir(exist_ok=True)

metrics = {
    "pr_auc": ap,
    "f1_best": f1_best,
    "best_threshold": best_th,
    "positives_train": pos,
    "negatives_train": neg,
    "scale_pos_weight": scale_pos_weight,
    "n_features_numeric": len(num_cols),
    "n_features_categorical": len(cat_cols),
}
with open(OUT_DIR / "reports" / "hourly_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# Öznitelik isimleri (mümkünse)
try:
    feat_names = pipe.named_steps["pre"].get_feature_names_out().tolist()
    pd.Series(feat_names, name="feature").to_csv(OUT_DIR / "reports" / "feature_names.csv", index=False)
except Exception as e:
    print(f"[WARN] Özellik isimleri alınamadı: {e}")

# Modeli kaydet
try:
    import joblib
    joblib.dump(pipe, OUT_DIR / "models" / ("sutam_hourly_xgb.joblib" if USE_XGBOOST else "sutam_hourly_lgbm.joblib"))
except Exception as e:
    print(f"[WARN] Model kaydedilemedi: {e}")
