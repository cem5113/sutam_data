# train_hourly_model_balanced.py — SUTAM (GEOID×hour) | SMOTENC + Threshold Tuning
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

# Imbalance
from imblearn.over_sampling import SMOTENC

# Model seçimi
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
# 2) Özellik seçimi (sızıntısız)
# =========================
DROP = {"Y_label", "dt", "GEOID", "dt_local", "crime_count", "hr_cnt", "daily_cnt"}

CAT_COLS = ["year", "month", "day_of_week", "hour", "season"]

NUM_COLS = [
    "neighbor_crime_24h", "neighbor_crime_72h", "neighbor_crime_7d",
    "911_geo_hr_last3d", "911_geo_hr_last7d", "311_request_count",
    "wx_tavg", "wx_tmin", "wx_tmax", "wx_prcp", "wx_temp_range",
    "wx_is_rainy", "wx_is_hot_day",
    "poi_total_count", "poi_risk_score",
    "poi_count_300m", "poi_risk_300m",
    "poi_count_600m", "poi_risk_600m",
    "poi_count_900m", "poi_risk_900m",
    "distance_to_police", "distance_to_government_building",
    "bus_stop_count", "train_stop_count", "population",
    "prior_cnt_3m", "prior_p_3m", "prior_cnt_12m", "prior_p_12m",
]

present = set(df.columns)
cat_cols = [c for c in CAT_COLS if c in present]
num_cols = [c for c in NUM_COLS if c in present]
all_cols = cat_cols + num_cols
if not all_cols:
    raise SystemExit("Kullanılabilir özellik bulunamadı (all_cols boş).")

print(f"[INFO] Özellik sayısı → numerik: {len(num_cols)} | kategorik: {len(cat_cols)}")
print(f"[INFO] Düşülen kolonlar: {sorted(DROP.intersection(present))}")


# =========================
# 3) Zaman bazlı split (son %20 test)
# =========================
df = df.sort_values("dt").reset_index(drop=True)
cut_idx = int(len(df) * 0.80)
train = df.iloc[:cut_idx].copy()
test  = df.iloc[cut_idx:].copy()

X_train = train[all_cols].copy()
y_train = train["Y_label"].astype(np.int8)
X_test  = test[all_cols].copy()
y_test  = test["Y_label"].astype(np.int8)

# Kategorikleri category dtype yap (güvenli)
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_test[c]  = X_test[c].astype("category")

pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
print(f"[INFO] Train set: pos={pos:,}  neg={neg:,}  rate={pos/(pos+neg):.4f}")


# =========================
# 4) Bellek dostu örnekleme + SMOTENC (yalnızca TRAIN)
# =========================
# Büyük veri → önce negatiflerden sınırlı örnekleyip sonra SMOTE
NEG_PER_POS = 5            # her 1 pozitif için en fazla 5 negatif
TARGET_POS_RATE = 0.20     # SMOTE sonrası hedef ~%20 pozitif (sampling_strategy)

rng = np.random.default_rng(42)
pos_idx = np.flatnonzero(y_train.values == 1)
neg_idx = np.flatnonzero(y_train.values == 0)

target_neg = min(len(neg_idx), NEG_PER_POS * len(pos_idx))
if target_neg < len(neg_idx):
    neg_idx_sampled = rng.choice(neg_idx, size=target_neg, replace=False)
else:
    neg_idx_sampled = neg_idx

keep_idx = np.concatenate([pos_idx, neg_idx_sampled])
keep_idx.sort()

X_tr_small = X_train.iloc[keep_idx].copy()
y_tr_small = y_train.iloc[keep_idx].copy()

print(f"[INFO] SMOTE öncesi: pos={int((y_tr_small==1).sum()):,}  "
      f"neg={int((y_tr_small==0).sum()):,}  n={len(X_tr_small):,}")

# --- SMOTENC girişini hazırlama: kategorileri kodla, numerikleri doldur
code_maps = {}   # {col: {code:int -> label:any}}
label_maps = {}  # {col: {label:any -> code:int}}

# Kategorikler: eksikleri doldur, kategori yap, code’a çevir
for c in cat_cols:
    X_tr_small[c] = X_tr_small[c].cat.add_categories(["__missing__"])
    X_tr_small[c] = X_tr_small[c].fillna("__missing__").astype("category")
    labels = list(X_tr_small[c].cat.categories)
    label_to_code = {lab: i for i, lab in enumerate(labels)}
    code_to_label = {i: lab for lab, i in label_to_code.items()}
    label_maps[c] = label_to_code
    code_maps[c] = code_to_label
    X_tr_small[c] = X_tr_small[c].map(label_to_code).astype("int32")

# Numerikler: median ile doldur
for c in num_cols:
    if c in X_tr_small.columns:
        if X_tr_small[c].isna().any():
            med = float(X_tr_small[c].median()) if not X_tr_small[c].dropna().empty else 0.0
            X_tr_small[c] = X_tr_small[c].fillna(med)
        X_tr_small[c] = X_tr_small[c].astype("float32")

# SMOTENC categorical mask (all_cols sırası önemli)
cat_mask = [c in cat_cols for c in all_cols]
X_smote_input = X_tr_small[all_cols].to_numpy()

# sampling_strategy: hedef pozitif oran p → n_pos_after = p/(1-p) * n_neg_after
# imblearn burada oransal kabul eder (0<p<1) → minority:majority oranı
sampling_strategy = TARGET_POS_RATE

sm = SMOTENC(
    categorical_features=cat_mask,
    sampling_strategy=sampling_strategy,
    k_neighbors=5,
    n_jobs=4,
    random_state=42,
)

X_res, y_res = sm.fit_resample(X_smote_input, y_tr_small.to_numpy().astype(int))
print(f"[INFO] SMOTE sonrası: pos={int((y_res==1).sum()):,}  neg={int((y_res==0).sum()):,}  n={len(X_res):,}")

# Kodlardan tekrar etiketlere dön (kategorikler)
X_res_df = pd.DataFrame(X_res, columns=all_cols)
for c in cat_cols:
    inv = code_maps[c]
    X_res_df[c] = X_res_df[c].round().astype(int).map(inv).astype("category")

# Numerikleri tipe oturt
for c in num_cols:
    X_res_df[c] = pd.to_numeric(X_res_df[c], errors="coerce").astype("float32")


# =========================
# 5) Ön işleme + Model (OHE sparse)
# =========================
num_tf = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
])

cat_tf = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
])

transformers = []
if num_cols: transformers.append(("num", num_tf, num_cols))
if cat_cols: transformers.append(("cat", cat_tf, cat_cols))

pre = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
    sparse_threshold=1.0,   # asla densify etme
)

# SMOTE sonrası sınıf oranı zaten dengelendi → scale_pos_weight = 1.0
if USE_XGBOOST:
    clf = XGBClassifier(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=1.0,
        n_jobs=4,
        random_state=42,
    )
else:
    clf = LGBMClassifier(
        n_estimators=1400,
        learning_rate=0.05,
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
# 6) Eğitim (SMOTE’lu train set)
# =========================
pipe.fit(X_res_df, y_res.astype(int))

# =========================
# 7) Değerlendirme (değişmemiş TEST set)
# =========================
# Test kategorilerini category tut
for c in cat_cols:
    X_test[c] = X_test[c].astype("category")

proba = pipe.predict_proba(X_test)[:, 1].astype(np.float32)
ap = float(average_precision_score(y_test, proba))
print(f"\n[RESULT] PR-AUC (Average Precision) on TEST: {ap:.4f}")

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
    "pr_auc_test": ap,
    "f1_best_test": f1_best,
    "best_threshold": best_th,
    "train_pos_before": pos,
    "train_neg_before": neg,
    "smote_pos_after": int((y_res == 1).sum()),
    "smote_neg_after": int((y_res == 0).sum()),
    "neg_per_pos_sampling": NEG_PER_POS,
    "target_pos_rate_smote": TARGET_POS_RATE,
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
    joblib.dump(pipe, OUT_DIR / "models" / ("sutam_hourly_xgb_balanced.joblib" if USE_XGBOOST else "sutam_hourly_lgbm_balanced.joblib"))
except Exception as e:
    print(f"[WARN] Model kaydedilemedi: {e}")
