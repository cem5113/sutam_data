#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_multi_windows.py  (REVIZE — sızıntı kapalı, class-imbalance bilinçli)
- Girdi: sf_crime_grid_{3h,8h,1d,1w,1m}.parquet
- Split: zaman bazlı (son %20 test)
- Sızıntı koruma: crime_count/hr_cnt/daily_cnt, Y_label türevleri ve risk_*/metrics_* dışlanır
- Dengesizlik: scale_pos_weight otomatik; opsiyonel undersample train set'te
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

LEAK_PREFIXES = ("risk_", "metrics_")
LEAK_FORBIDDEN_SUBSTR = ("y_label", "ylabel", "label", "crime_count", "hr_cnt", "daily_cnt")

# ---- Model seçimi (spw destekli)
def make_estimator(which: str, spw: float = 1.0):
    if which.lower() == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=1200, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, objective="binary",
            is_unbalance=False,  # spw kullanacağız
            class_weight=None,
            n_jobs=4, random_state=42,
            scale_pos_weight=spw,
        )
    else:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, min_child_weight=1.0,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="aucpr",
            n_jobs=4, random_state=42,
            scale_pos_weight=spw,
        )

def undersample_negatives(X: pd.DataFrame, y: np.ndarray, ratio: float) -> Tuple[pd.DataFrame, np.ndarray]:
    """Train set'te negatifleri (y==0) 'ratio' oranında tutar (0<ratio<=1)."""
    if ratio <= 0 or ratio >= 1:
        return X, y
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(neg_idx) == 0 or len(pos_idx) == 0:
        return X, y
    keep_neg = np.random.RandomState(42).choice(neg_idx, size=int(len(neg_idx)*ratio), replace=False)
    keep = np.concatenate([pos_idx, keep_neg])
    keep.sort()
    return X.iloc[keep].reset_index(drop=True), y[keep]

def _drop_leaks(df: pd.DataFrame, base_drop: set, cat_cols: list) -> tuple[list, list]:
    """Sızıntı: risk_*/metrics_* önekleri ve yasak alt-stringleri uçur."""
    num_cols = []
    for c in df.columns:
        cl = c.lower()
        if (c in base_drop) or (c in cat_cols):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if any(cl.startswith(p) for p in LEAK_PREFIXES):
            continue
        if any(s in cl for s in LEAK_FORBIDDEN_SUBSTR):
            continue
        num_cols.append(c)
    safe_cat = []
    for c in cat_cols:
        cl = c.lower()
        if any(cl.startswith(p) for p in LEAK_PREFIXES):  # nadir ama garanti olsun
            continue
        if any(s in cl for s in LEAK_FORBIDDEN_SUBSTR):
            continue
        if c in df.columns:
            safe_cat.append(c)
    return num_cols, safe_cat

def train_one(freq: str, path: Path, model_type: str, undersample: float, out_models: Path, out_reports: Path) -> Dict:
    df = pd.read_parquet(path)
    req = {"GEOID", "t0", "Y_label", "year", "month", "day_of_week"}
    missing = req - set(df.columns)
    if missing:
        raise SystemExit(f"[{freq}] Eksik kolon(lar): {missing}")

    # Zaman sırası & split
    df["t0"] = pd.to_datetime(df["t0"], utc=True, errors="coerce")
    df = df.sort_values("t0").reset_index(drop=True)
    cut = int(len(df) * 0.80)
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    # Özellik seti
    DROP = {"GEOID", "t0", "Y_label", "dt_local",
            "crime_count", "hr_cnt", "daily_cnt"}  # sayaçları kesin dışarıda tut
    CAT  = ["year", "month", "day_of_week"]
    if "block_id" in df.columns:
        CAT.append("block_id")

    # Sızıntı filtreleri
    num_cols, cat_cols = _drop_leaks(df, DROP, CAT)

    if not (num_cols or cat_cols):
        raise SystemExit(f"[{freq}] Kullanılabilir feature bulunamadı (sızıntı filtresi çok katı olabilir).")

    X_train = train[cat_cols + num_cols]
    y_train = train["Y_label"].astype(np.int8).values
    X_test  = test[cat_cols + num_cols]
    y_test  = test["Y_label"].astype(np.int8).values

    # Class-imbalance: scale_pos_weight
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = float(neg / max(1, pos)) if pos > 0 else 1.0

    # Opsiyonel undersample (sadece train)
    X_train, y_train = undersample_negatives(X_train, y_train, undersample)

    # Pipeline
    num_tf = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
    ])
    transformers = []
    if num_cols: transformers.append(("num", num_tf, num_cols))
    if cat_cols: transformers.append(("cat", cat_tf, cat_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)

    clf = make_estimator(model_type, spw=spw)
    pipe = Pipeline([("pre", pre), ("mdl", clf)])

    # Eğitim
    pipe.fit(X_train, y_train)

    # Değerlendirme
    proba = pipe.predict_proba(X_test)[:, 1].astype(np.float32)
    ap = float(average_precision_score(y_test, proba))
    prec, rec, th = precision_recall_curve(y_test, proba)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
    # thresholds uzunluğu = len(prec)-1; en güvenlisi şu:
    best_idx = int(np.nanargmax(f1s[:-1])) if len(f1s) > 1 else 0
    best_th  = float(th[best_idx]) if len(th) else 0.5
    y_pred_best = (proba >= best_th).astype(np.int8)
    f1_best = float(f1_score(y_test, y_pred_best))

    # Confusion matrix (best_th)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
    cls_rep = classification_report(y_test, y_pred_best, digits=4)

    # Çıktıları yaz
    out_models.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
        joblib.dump(pipe, out_models / f"sutam_{freq.lower()}.joblib")
    except Exception as e:
        print(f"[WARN][{freq}] Model kaydı başarısız: {e}")

    # PR eğrisi CSV
    pr_df = pd.DataFrame({"precision": prec, "recall": rec})
    pr_df.to_csv(out_reports / f"pr_curve_{freq.lower()}.csv", index=False)

    # Metin raporu
    with open(out_reports / f"classification_report_{freq.lower()}.txt", "w", encoding="utf-8") as f:
        f.write(cls_rep)

    # JSON metrikler
    metrics = {
        "freq": freq,
        "data_file": str(path),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features_num": len(num_cols),
        "n_features_cat": len(cat_cols),
        "class_balance_train": {"pos": pos, "neg": neg, "scale_pos_weight": spw},
        "class_balance_test": {
            "y1_rate": float(y_test.mean()),
            "y1_count": int((y_test == 1).sum()),
            "y0_count": int((y_test == 0).sum())
        },
        "pr_auc": ap,
        "best_threshold": best_th,
        "f1_best": f1_best,
        "confusion_best": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "model_type": model_type,
        "undersample_neg_train": undersample,
        "features_used_sample": (cat_cols + num_cols)[:30],
    }
    with open(out_reports / f"metrics_{freq.lower()}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[RESULT][{freq}] PR-AUC={ap:.4f} | Best-F1@{best_th:.3f}={f1_best:.4f} | "
          f"test y=1 %{100*y_test.mean():.3f}")
    if ap > 0.95:
        print(f"[WARN][{freq}] PR-AUC aşırı yüksek — sızıntı şüphesi. Feature listesine tekrar bakın.")
    return metrics

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=Path("."), help="Parquet dizini")
    p.add_argument("--prefix", type=str, default="sf_crime_grid_", help="Dosya öneki")
    p.add_argument("--freqs", type=str, default="1D", help="Virgüllü liste (örn: 1D veya 3H,8H,1D)")
    p.add_argument("--model", type=str, choices=["xgb","lgbm"], default="xgb")
    p.add_argument("--undersample", type=float, default=0.0, help="Train set negatif undersample (0-1)")
    p.add_argument("--reports", type=Path, default=Path("reports"))
    p.add_argument("--models",  type=Path, default=Path("models"))
    return p.parse_args()

def main():
    args = parse_args()
    freqs: List[str] = [f.strip().upper() for f in args.freqs.split(",") if f.strip()]
    summary = []
    for f in freqs:
        p = args.dir / f"{args.prefix}{f.lower()}.parquet"
        if not p.exists():
            print(f"[SKIP] {f}: {p} yok")
            continue
        m = train_one(f, p, args.model, args.undersample, args.models, args.reports)
        summary.append(m)

    # Toplu özet
    args.reports.mkdir(parents=True, exist_ok=True)
    with open(args.reports / "metrics_summary_multi_windows.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
