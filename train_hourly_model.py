#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_hourly_model.py
- Saatlik full-grid (sf_crime_grid_full_labeled.parquet) veya aggregate dosyası ile eğitim
- Zaman bazlı split (son %20 test), sızıntısız
- Dengesizlik: scale_pos_weight + opsiyonel undersample
- Çıktı: models/*.joblib, reports/*.json
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier
import joblib

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("sf_crime_grid_full_labeled.parquet"))
    p.add_argument("--target", type=str, default="Y_label")
    p.add_argument("--undersample", type=float, default=0.3, help="Negatif sınıf oranı (0=kapalı, 0.3 öneri)")
    p.add_argument("--model-out", type=Path, default=Path("models/sutam_hourly.joblib"))
    p.add_argument("--report-out", type=Path, default=Path("reports/hourly_metrics.json"))
    return p.parse_args()

def main():
    args = parse_args()
    src = args.input
    if not src.exists():
        raise SystemExit(f"Girdi yok: {src.resolve()}")

    df = pd.read_parquet(src)

    # Zaman kolonu: hourly'de "dt", aggregatede "t0"
    time_col = "dt" if "dt" in df.columns else ("t0" if "t0" in df.columns else None)
    if time_col is None:
        raise SystemExit("Zaman kolonu bulunamadı (dt / t0).")

    req = {"GEOID", time_col, args.target}
    missing = req - set(df.columns)
    if missing:
        raise SystemExit(f"Gerekli kolonlar eksik: {missing}")

    df = df.sort_values(time_col).reset_index(drop=True)

    # Özellik seçimi
    drop_cols = {"GEOID", time_col, args.target}
    cat_cols = [c for c in ["year","month","day_of_week","hour","block_id"] if c in df.columns]
    num_cols = [c for c in df.columns
                if c not in drop_cols and c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])]

    # Zaman bazlı split
    cut = int(len(df) * 0.80)
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    # Dengesizlik bilgisi
    y_train = train[args.target].astype(np.int8).values
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = float(neg / max(1, pos)) if pos > 0 else 1.0

    # İsteğe bağlı basit undersample (negatiflerden kırp)
    if args.undersample and args.undersample > 0:
        neg_idx = train[train[args.target] == 0].index.to_numpy()
        pos_idx = train[train[args.target] == 1].index.to_numpy()
        keep_neg = int(len(neg_idx) * (1.0 - float(args.undersample)))
        if keep_neg < len(neg_idx):
            rng = np.random.default_rng(42)
            keep_idx = set(rng.choice(neg_idx, size=keep_neg, replace=False).tolist()) | set(pos_idx.tolist())
            train = train.loc[sorted(list(keep_idx))].copy()
            y_train = train[args.target].astype(np.int8).values
            pos = int((y_train == 1).sum()); neg = int((y_train == 0).sum())
            spw = float(neg / max(1, pos)) if pos > 0 else 1.0

    X_train = train[cat_cols + num_cols]
    X_test  = test[cat_cols + num_cols]
    y_test  = test[args.target].astype(np.int8).values

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
        scale_pos_weight=spw,
    )

    pipe = Pipeline([("pre", pre), ("mdl", clf)])
    pipe.fit(X_train, train[args.target].astype(np.int8).values)

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

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.model_out)

    metrics = {
        "data_file": str(src),
        "time_col": time_col,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "n_features_num": len(num_cols),
        "n_features_cat": len(cat_cols),
        "class_balance_train": {"pos": int(pos), "neg": int(neg), "scale_pos_weight": spw},
        "pr_auc": ap, "f1_best": f1_best, "best_threshold": best_th,
    }
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
