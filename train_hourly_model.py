#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_hourly_model.py
- Saatlik full-grid (sf_crime_grid_full_labeled.parquet) veya aggregate (8H/1D) dosyası ile eğitim
- Zaman bazlı split (son %20 test), sızıntı korumalı
- Dengesizlik: scale_pos_weight (otomatik) + opsiyonel negatif undersample
- CLI:
    --input <parquet>                   : Girdi dosyası (default: sf_crime_grid_full_labeled.parquet)
    --target Y_label                    : Hedef kolon (default)
    --freq {hourly,8H,1D}               : Veri frekansı (default: hourly)
    --tz America/Los_Angeles            : Bilgi amaçlı (log), zorunlu değil
    --undersample 0.30                  : Negatif sınıfta kırpma oranı (0=kapalı)
    --model-out models/..joblib         : Çıktı model dosyası (default freq’e göre isimlenir)
    --report-out reports/..json         : Metrik çıktısı (default freq’e göre isimlenir)
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


# ===== CLI =====
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("sf_crime_grid_full_labeled.parquet"))
    p.add_argument("--target", type=str, default="Y_label")
    p.add_argument("--freq", type=str, choices=["hourly", "8H", "1D"], default="hourly")
    p.add_argument("--tz", type=str, default=None)
    p.add_argument("--undersample", type=float, default=0.0, help="Negatif sınıf kırpma oranı [0-0.9] (0=kapalı)")
    p.add_argument("--model-out", type=Path, default=None)
    p.add_argument("--report-out", type=Path, default=None)
    return p.parse_args()


# ===== Sızıntı koruma =====
FORBIDDEN_SUBSTRINGS = ["y_label", "ylabel", "label", "crime_count", "hr_cnt", "daily_cnt"]

PRIOR_OK_PREFIX = ("prior_cnt_", "prior_p_")  # shift(1) ile üretildiyse güvenli

def assert_no_leak(cols: list[str]):
    low = [c.lower() for c in cols]
    offenders = []
    for c in low:
        if any(bad in c for bad in FORBIDDEN_SUBSTRINGS):
            offenders.append(c)
    if offenders:
        raise SystemExit("❌ Potansiyel sızıntı: feature set içinde yasak kolon(lar): "
                         + ", ".join(sorted(set(offenders))))
    priors = [c for c in cols if c.startswith(PRIOR_OK_PREFIX)]
    if priors:
        print(f"[INFO] Priors (izinli): {sorted(priors)[:10]}{' ...' if len(priors)>10 else ''}")


# ===== Ana =====
def main():
    args = parse_args()

    # Varsayılan çıktı yollarını freq’e göre kur
    if args.model_out is None:
        args.model_out = Path(f"models/sutam_{args.freq}.joblib")
    if args.report_out is None:
        args.report_out = Path(f"reports/metrics_{args.freq}.json")

    src = args.input
    if not src.exists():
        raise SystemExit(f"Girdi yok: {src.resolve()}")

    df = pd.read_parquet(src)

    # Zaman kolonu: blok veride t0, hourly’de dt
    time_col = "t0" if "t0" in df.columns else ("dt" if "dt" in df.columns else None)
    if time_col is None:
        raise SystemExit("Zaman kolonu bulunamadı (dt / t0).")

    # Temel kontroller
    need = {"GEOID", time_col, args.target}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Gerekli kolon(lar) eksik: {miss}")

    # Sıralama (leakage-safe split)
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)

    # Özellik seçimi
    drop_cols = {"GEOID", time_col, args.target, "dt_local"}
    cat_candidates = ["year", "month", "day_of_week", "hour", "block_id"]
    # freq’e göre mantıklı kategorikler
    cat_cols = []
    for c in cat_candidates:
        if c in df.columns:
            if args.freq == "hourly" and c in ("hour", "year", "month", "day_of_week"):
                cat_cols.append(c)
            elif args.freq in ("8H", "1D"):
                # 8H ise block_id olabilir; 1D’de genelde yok ama varsa alabiliriz
                if c in ("year", "month", "day_of_week", "block_id"):
                    cat_cols.append(c)

    num_cols = [c for c in df.columns
                if (c not in drop_cols)
                and (c not in cat_cols)
                and pd.api.types.is_numeric_dtype(df[c])]

    feat_cols = cat_cols + num_cols
    if not feat_cols:
        raise SystemExit("Kullanılabilir feature bulunamadı.")

    # Sızıntı muhafazası
    assert_no_leak(feat_cols)

    # Zaman bazlı split
    cut = int(len(df) * 0.80)
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    # Dengesizlik bilgisi
    y_train = train[args.target].astype(np.int8).values
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = float(neg / max(1, pos)) if pos > 0 else 1.0

    # İsteğe bağlı negatif undersample
    if args.undersample and 0.0 < args.undersample < 0.95:
        neg_idx = train.index[train[args.target] == 0].to_numpy()
        pos_idx = train.index[train[args.target] == 1].to_numpy()
        keep_neg = int(len(neg_idx) * (1.0 - float(args.undersample)))
        if keep_neg < len(neg_idx):
            rng = np.random.default_rng(42)
            keep_neg_idx = rng.choice(neg_idx, size=keep_neg, replace=False)
            keep_idx = np.sort(np.concatenate([keep_neg_idx, pos_idx]))
            train = train.loc[keep_idx].copy()
            y_train = train[args.target].astype(np.int8).values
            pos = int((y_train == 1).sum()); neg = int((y_train == 0).sum())
            spw = float(neg / max(1, pos)) if pos > 0 else 1.0

    X_train = train[feat_cols].copy()
    X_test  = test[feat_cols].copy()
    y_test  = test[args.target].astype(np.int8).values

    print(f"[INFO] Input: {src.name} | freq={args.freq} | tz={args.tz or 'N/A'}")
    print(f"[INFO] Numerik={len(num_cols)} | Kategorik={len(cat_cols)} | "
          f"Train pos={pos:,} neg={neg:,} spw={spw:.2f}")

    # Pipeline
    num_tf = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
    ])

    transformers = []
    if num_cols: transformers.append(("num", num_tf, num_cols))
    if cat_cols: transformers.append(("cat", cat_tf, cat_cols))

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )

    clf = XGBClassifier(
        n_estimators=700 if args.freq != "hourly" else 600,
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
    pipe.fit(X_train, y_train)

    # Değerlendirme
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

    # Aşırı iyi metrik uyarısı (sızıntı alarmı)
    if ap > 0.95:
        print("⚠️  UYARI: PR-AUC > 0.95 — tipik olarak sızıntı göstergesidir. "
              "Feature set ve zaman splitini yeniden kontrol edin.")

    # Çıktılar
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.model_out)

    metrics = {
        "data_file": str(src),
        "freq": args.freq,
        "time_col": time_col,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "n_features_num": len(num_cols), "n_features_cat": len(cat_cols),
        "class_balance_train": {"pos": int(pos), "neg": int(neg), "scale_pos_weight": spw},
        "pr_auc": ap, "f1_best": f1_best, "best_threshold": best_th,
        "features_used": feat_cols[:50] + (["..."] if len(feat_cols) > 50 else []),
    }
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
