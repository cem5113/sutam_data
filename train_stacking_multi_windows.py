#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_stacking_multi_windows.py
- Her frekans için (1D,1W,1M) stacking: [XGB, LGBM, RF, LR] -> meta LR
- Zaman bazlı split (son %20 test) — sızıntı yok
- OHE: year, month, day_of_week, (varsa) block_id
- Numerikler: median impute + (opsiyonel) scale
- Çıktılar:
    * sutam_stack_{freq}.joblib
    * metrics_{freq}.json
    * pr_curve_{freq}.csv
    * classification_report_{freq}.txt
    * metrics_summary_multi_windows.json (güncellenir)
"""

from __future__ import annotations
import json, sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import joblib

# İsteğe bağlı; yoksa sadece uyarı verip o base learner’ı atlarız
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


FREQS = ["1d", "1w", "1m"]
GRID_PATH = "sf_crime_grid_{freq}.parquet"

OUT_MODEL = "sutam_stack_{freq}.joblib"
OUT_METRICS_JSON = "metrics_{freq}.json"
OUT_PR_CSV = "pr_curve_{freq}.csv"
OUT_CLS_TXT = "classification_report_{freq}.txt"
OUT_SUMMARY = "metrics_summary_multi_windows.json"

RANDOM_STATE = 42


def ensure_utc(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, utc=True)
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("UTC")
    return dt


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float, float, float]:
    p, r, th = precision_recall_curve(y_true, prob)
    # PRC 'th' uzunluğu  len(p)-1; F1'i th üzerinden hesapla
    f1s = (2 * p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-12)
    i = np.nanargmax(f1s)
    return float(th[i]), float(f1s[i]), float(p[i]), float(r[i])


def build_stacking_pipeline(num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    num_tr = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))  # sparse uyumlu
    ])
    cat_tr = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_tr, num_cols),
            ("cat", cat_tr, cat_cols)
        ],
        sparse_threshold=0.3
    )

    estimators = []
    if HAS_XGB:
        estimators.append(("xgb", XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss",
            tree_method="hist"
        )))
    if HAS_LGBM:
        estimators.append(("lgbm", LGBMClassifier(
            n_estimators=500, max_depth=-1, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, objective="binary"
        )))

    estimators.append(("rf", RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, n_jobs=-1, random_state=RANDOM_STATE
    )))
    estimators.append(("lr", LogisticRegression(
        max_iter=2000, solver="saga", penalty="l2", n_jobs=-1
    )))

    # Hiç XGB/LGBM yoksa yine de RF+LR ile stacking çalışır
    meta = LogisticRegression(max_iter=2000, solver="liblinear")

    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        stack_method="predict_proba",
        passthrough=True,            # base input features da meta’ya gider
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("model", clf)
    ])
    return pipe


def train_one(freq: str) -> Dict:
    path = GRID_PATH.format(freq=freq)
    df = pd.read_parquet(path)

    # Zorunlu kolonlar
    need = ["GEOID", "t0", "Y_label"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"[{freq}] '{c}' kolonu yok: {path}")

    # Tipler
    df["GEOID"] = df["GEOID"].astype(str)
    df["t0"] = ensure_utc(df["t0"])
    df = df.sort_values("t0").reset_index(drop=True)

    # Özellik seçimi
    # Numerikler: priors + (varsa) risk sayıları; 'crime_count' GELECEKTE bilinmeyeceği için eğitimde bile kullanmamak daha sağlıklı
    num_cols = [c for c in df.columns if c.startswith("prior_cnt_") or c.startswith("prior_p_")]
    # Kategorikler: takvim sütunları ve (varsa) block_id (GEOID'i bilinçli olarak OHE’lemiyoruz, yer öğrenmesini aşırılaştırabilir)
    # takvim kolonların yoksa üret
    if "year" not in df.columns:
        dt = df["t0"]
        df["year"] = dt.dt.year.astype("int16")
        df["month"] = dt.dt.month.astype("int8")
        df["day_of_week"] = dt.dt.dayofweek.astype("int8")

    cat_cols = [c for c in ["year", "month", "day_of_week", "block_id"] if c in df.columns]

    drop_cols = set(["Y_label", "t0", "GEOID", "crime_count"])
    X_cols = [c for c in df.columns if c not in drop_cols and (c in num_cols or c in cat_cols)]

    X = df[X_cols].copy()
    y = df["Y_label"].astype(int).values

    # Zaman bazlı split (son %20 test)
    n = len(df)
    cut = int(n * 0.8)
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y[:cut], y[cut:]

    pipe = build_stacking_pipeline(num_cols=[c for c in X_cols if c in num_cols],
                                   cat_cols=[c for c in X_cols if c in cat_cols])

    pipe.fit(X_train, y_train)

    # Tahmin & metrikler
    prob = pipe.predict_proba(X_test)[:, 1]
    ap = average_precision_score(y_test, prob)
    th, f1, p_at, r_at = best_f1_threshold(y_test, prob)
    y_pred = (prob >= th).astype(int)

    rep = classification_report(y_test, y_pred, digits=4)
    print(f"[RESULT][{freq.upper()}] PR-AUC={ap:.4f} | Best-F1@{th:.3f}={f1:.4f}")

    # Kaydet
    joblib.dump(pipe, OUT_MODEL.format(freq=freq))
    Path(".").joinpath(OUT_CLS_TXT.format(freq=freq)).write_text(rep, encoding="utf-8")
    pd.DataFrame({"prob": prob, "y_true": y_test}).to_csv(OUT_PR_CSV.format(freq=freq), index=False)

    metrics = {
        "freq": freq,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "ap": float(ap),
        "best_threshold": float(th),
        "best_f1": float(f1),
        "precision_at_best": float(p_at),
        "recall_at_best": float(r_at),
        "features": X_cols
    }
    Path(".").joinpath(OUT_METRICS_JSON.format(freq=freq)).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    allm = {}
    for f in FREQS:
        try:
            m = train_one(f)
            allm[f] = m
        except Exception as e:
            print(f"[WARN] {f} eğitiminde hata: {e}")
    Path(OUT_SUMMARY).write_text(json.dumps(allm, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
