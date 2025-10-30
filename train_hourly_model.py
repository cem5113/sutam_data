# train_hourly_model_balanced.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ML
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve, classification_report
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

from imblearn.over_sampling import SMOTENC

# MODEL: XGBoost / LightGBM
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

for col in ["Y_label", "dt", "GEOID"]:
    if col not in df.columns:
        raise SystemExit(f"Gerekli kolon eksik: {col}")

y_rate = df["Y_label"].mean()
print(f"[INFO] Y_label oranı (1'ler): {100*y_rate:.4f}%  |  0'lar: {100*(1-y_rate):.4f}%")
print(f"[INFO] Satır: {len(df):,}  GEOID: {df['GEOID'].nunique():,}  Saat aralığı: {df['dt'].min()} → {df['dt'].max()}")


# =========================
# 2) Özellik seçimi
# =========================
DROP = {"Y_label", "dt", "GEOID", "dt_local", "crime_count", "hr_cnt", "daily_cnt"}

CAT_COLS = ["year", "month", "day_of_week", "hour", "season"]
NUM_COLS = [
    "neighbor_crime_24h","neighbor_crime_72h","neighbor_crime_7d",
    "911_geo_hr_last3d","911_geo_hr_last7d","311_request_count",
    "wx_tavg","wx_tmin","wx_tmax","wx_prcp","wx_temp_range","wx_is_rainy","wx_is_hot_day",
    "poi_total_count","poi_risk_score",
    "poi_count_300m","poi_risk_300m","poi_count_600m","poi_risk_600m","poi_count_900m","poi_risk_900m",
    "distance_to_police","distance_to_government_building",
    "bus_stop_count","train_stop_count","population",
    "prior_cnt_3m","prior_p_3m","prior_cnt_12m","prior_p_12m",
]

present = set(df.columns)
cat_cols = [c for c in CAT_COLS if c in present]
num_cols = [c for c in NUM_COLS if c in present]
all_cols = cat_cols + num_cols
if not all_cols:
    raise SystemExit("Kullanılabilir özellik yok.")

print(f"[INFO] Özellik sayısı → numerik: {len(num_cols)} | kategorik: {len(cat_cols)}")
print(f"[INFO] Düşülen kolonlar: {sorted(DROP.intersection(present))}")


# =========================
# 3) Zaman bazlı split (leakage güvenli)
# =========================
df = df.sort_values("dt").reset_index(drop=True)
cut_idx = int(len(df) * 0.80)
train = df.iloc[:cut_idx].copy()
test  = df.iloc[cut_idx:].copy()

X_train_raw = train[all_cols].copy()
y_train = train["Y_label"].astype(np.int8).values
X_test_raw  = test[all_cols].copy()
y_test  = test["Y_label"].astype(np.int8).values

# Kategorikleri kategori→kod (int) yap (SMOTE için OHE YAPMA)
for c in cat_cols:
    X_train_raw[c] = X_train_raw[c].astype("category").cat.codes.astype("int16")
    X_test_raw[c]  = X_test_raw[c].astype("category").cat.codes.astype("int16")


# =========================
# 4) Basit imputation (SMOTE öncesi)
# =========================
imp_num = SimpleImputer(strategy="median")
imp_cat = SimpleImputer(strategy="most_frequent")

X_train_imp = X_train_raw.copy()
if num_cols:
    X_train_imp[num_cols] = imp_num.fit_transform(X_train_imp[num_cols])
if cat_cols:
    X_train_imp[cat_cols] = imp_cat.fit_transform(X_train_imp[cat_cols])

# Test setine de aynı imputers
X_test_imp = X_test_raw.copy()
if num_cols:
    X_test_imp[num_cols] = imp_num.transform(X_test_imp[num_cols])
if cat_cols:
    X_test_imp[cat_cols] = imp_cat.transform(X_test_imp[cat_cols])


# =========================
# 5) Önce negatifleri 1:5 oranına indir, sonra SMOTENC ile arttır
# =========================
pos_idx = np.where(y_train == 1)[0]
neg_idx = np.where(y_train == 0)[0]

n_pos = len(pos_idx)
n_neg_target = min(len(neg_idx), n_pos * 5)  # 1:5
neg_idx_down = resample(neg_idx, replace=False, n_samples=n_neg_target, random_state=42)

sel_idx = np.concatenate([pos_idx, neg_idx_down])
X_train_bal = X_train_imp.iloc[sel_idx].reset_index(drop=True)
y_train_bal = y_train[sel_idx]

print(f"[INFO] Train set: pos={y_train.sum():,}  neg={(y_train==0).sum():,}  rate={100*y_train.mean():.4f}")
print(f"[INFO] SMOTE öncesi: pos={int((y_train_bal==1).sum()):,}  neg={int((y_train_bal==0).sum()):,}  n={len(X_train_bal):,}")

# SMOTENC — Kategorik kolon indeksleri
cat_idx = [all_cols.index(c) for c in cat_cols]
sm = SMOTENC(
    categorical_features=cat_idx,
    sampling_strategy=0.8,  # pozitifleri ~negatiflerin %80'i kadar yap
    k_neighbors=3,
    random_state=42,
)
X_sm, y_sm = sm.fit_resample(X_train_bal.values, y_train_bal)

print(f"[INFO] SMOTE sonrası: pos={int((y_sm==1).sum()):,}  neg={int((y_sm==0).sum()):,}  n={len(y_sm):,}")

# Resampled veriyi DataFrame’e geri sar ve kategorikleri tekrar int’e zorla
X_sm_df = pd.DataFrame(X_sm, columns=all_cols)
for c in cat_cols:
    X_sm_df[c] = np.rint(X_sm_df[c]).astype("int16")  # olası float sızıntısını temizle


# =========================
# 6) SMOTE sonrası: OHE + model pipe (sparse)
# =========================
cat_tf = Pipeline(steps=[
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
])
num_tf = "passthrough"

transformers = []
if num_cols: transformers.append(("num", num_tf, num_cols))
if cat_cols: transformers.append(("cat", cat_tf, cat_cols))

pre = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
    sparse_threshold=1.0,  # dense'e çevirmesin
)

if USE_XGBOOST:
    clf = XGBClassifier(
        n_estimators=700,
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
        scale_pos_weight=1.0,  # SMOTE sonrası 1.0
    )
else:
    clf = LGBMClassifier(
        n_estimators=1400,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary",
        is_unbalance=False,
        scale_pos_weight=1.0,
        n_jobs=4,
        random_state=42,
    )

pipe = Pipeline([
    ("pre", pre),
    ("mdl", clf),
])


# =========================
# 7) Eğitim
# =========================
pipe.fit(X_sm_df, y_sm)


# =========================
# 8) Değerlendirme
# =========================
proba = pipe.predict_proba(X_test_imp)[:, 1].astype(np.float32)
ap = float(average_precision_score(y_test, proba))
print(f"\n[RESULT] PR-AUC (Average Precision): {ap:.4f}")

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
# 9) Çıktılar
# =========================
OUT_DIR = Path(".")
(OUT_DIR / "models").mkdir(exist_ok=True)
(OUT_DIR / "reports").mkdir(exist_ok=True)

metrics = {
    "pr_auc": ap,
    "f1_best": f1_best,
    "best_threshold": best_th,
    "positives_train": int(y_train.sum()),
    "negatives_train": int((y_train==0).sum()),
    "after_downsample_pos": int((y_train_bal==1).sum()),
    "after_downsample_neg": int((y_train_bal==0).sum()),
    "after_smote_pos": int((y_sm==1).sum()),
    "after_smote_neg": int((y_sm==0).sum()),
    "n_features_numeric": len(num_cols),
    "n_features_categorical": len(cat_cols),
}
with open(OUT_DIR / "reports" / "hourly_metrics_balanced.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

try:
    feat_names = pipe.named_steps["pre"].get_feature_names_out().tolist()
    pd.Series(feat_names, name="feature").to_csv(OUT_DIR / "reports" / "feature_names_balanced.csv", index=False)
except Exception as e:
    print(f"[WARN] Özellik isimleri alınamadı: {e}")

try:
    import joblib
    joblib.dump(
        pipe,
        OUT_DIR / "models" / ("sutam_hourly_balanced_xgb.joblib" if USE_XGBOOST else "sutam_hourly_balanced_lgbm.joblib")
    )
except Exception as e:
    print(f"[WARN] Model kaydedilemedi: {e}")
