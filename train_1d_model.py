#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_1d_model.py  — Günlük (1D) odaklı, sızıntı-güvenli eğitim

Girdi  : 1D aggregate parquet (örn. sf_crime_grid_1d.parquet)
Hedef  : Y_label (>=1 olay → 1)
Split  : Zaman bazlı (ilk %80 train, son %20 test) — sızıntı yok
Denge  : scale_pos_weight (otomatik) + opsiyonel negatif undersample

Çıktılar:
- models/sutam_1d.joblib                (varsayılan model dosyası)
- reports/metrics_1d.json               (özet metrikler)
- reports/classification_report_1d.txt  (metin rapor; opsiyonel)
- reports/pr_curve_1d.csv               (PR eğrisi; opsiyonel)
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score, f1_score,
    precision_recall_curve, classification_report
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier
import joblib


# ===== CLI =====
def parse_args():
    p = argparse.ArgumentParser(description="SUTAM — 1D model eğitimi (sızıntı-güvenli)")
    p.add_argument("--input",       type=Path, default=Path("sf_crime_grid_1d.parquet"))
    p.add_argument("--target",      type=str,  default="Y_label")
    p.add_argument("--tz",          type=str,  default=None,
                   help="Sadece log amaçlı; verideki t0 zaten UTC varsayılır.")
    p.add_argument("--undersample", type=float, default=0.0,
                   help="Train set'te negatif sınıf KORUMA oranı (0=kapalı; 0.7 → negatiflerin %%70'i tutulur)")
    p.add_argument("--model-out",   type=Path, default=Path("models/sutam_1d.joblib"))
    p.add_argument("--report-out",  type=Path, default=Path("reports/metrics_1d.json"))
    p.add_argument("--clsrep-out",  type=Path, default=Path("reports/classification_report_1d.txt"))
    p.add_argument("--prcurve-out", type=Path, default=Path("reports/pr_curve_1d.csv"))
    return p.parse_args()


# ===== Sızıntı koruma =====
FORBIDDEN_SUBSTRINGS = ["y_label", "ylabel", "label", "crime_count", "hr_cnt", "daily_cnt"]
ALWAYS_EXCLUDE_PREFIXES = ("risk_", "metrics_")   # kesinlikle özellik dışı
PRIOR_OK_PREFIX = ("prior_cnt_", "prior_p_")      # sızıntısız üretildiyse güvenli


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


# ===== Yardımcılar =====
def _pick_time_col(df: pd.DataFrame) -> str:
    if "t0" in df.columns: return "t0"   # 1D özetlerde beklenen
    if "dt" in df.columns: return "dt"   # nadiren 1D'ye indirgenmiş ama dt korunmuş olabilir
    raise SystemExit("Zaman kolonu bulunamadı (t0 / dt).")


def main():
    args = parse_args()

    # Girdi
    src = args.input
    if not src.exists():
        raise SystemExit(f"Girdi yok: {src.resolve()}")
    df = pd.read_parquet(src)

    # Zaman kolonu ve kontroller
    time_col = _pick_time_col(df)
    need = {"GEOID", time_col, args.target}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Gerekli kolon(lar) eksik: {miss}")

    # Zaman sırası (leakage-safe split için)
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)

    # Özellik seçimi (1D)
    drop_cols = {"GEOID", time_col, args.target, "dt_local",
                 # güvenli olsun diye sayaç isimlerini de dışarıda tut
                 "crime_count", "hr_cnt", "daily_cnt"}
    cat_candidates = ["year", "month", "day_of_week", "block_id"]  # 1D’de block_id genelde yok; varsa alırız
    cat_cols = [c for c in cat_candidates if c in df.columns]

    # numerikleri seçerken risk_/metrics_ öneklerini tamamen hariç tut
    num_cols = []
    for c in df.columns:
        if c in drop_cols or c in cat_cols:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cl = c.lower()
        if cl.startswith(ALWAYS_EXCLUDE_PREFIXES):
            continue
        num_cols.append(c)

    feat_cols = cat_cols + num_cols
    if not feat_cols:
        raise SystemExit("Kullanılabilir feature bulunamadı (sızıntı dışı).")

    # Son kontrol: sızıntı yasaklı ifadeler var mı?
    assert_no_leak(feat_cols)

    # Zaman bazlı split
    cut = int(len(df) * 0.80)
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    # Dengesizlik & scale_pos_weight
    y_train = train[args.target].astype(np.int8).values
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = float(neg / max(1, pos)) if pos > 0 else 1.0

    # Opsiyonel negatif undersample (train)
    if args.undersample and 0.0 < args.undersample < 1.0:
        neg_idx = train.index[train[args.target] == 0].to_numpy()
        pos_idx = train.index[train[args.target] == 1].to_numpy()
        keep_neg = int(len(neg_idx) * float(args.undersample))  # oran: tutulacak NEGATİF yüzdesi
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

    print(f"[INFO] Input: {src.name} | freq=1D | tz={args.tz or 'N/A'}")
    print(f"[INFO] Numerik={len(num_cols)} | Kategorik={len(cat_cols)} | "
          f"Train pos={pos:,} neg={neg:,} spw={spw:.2f}")

    # === Pipeline ===
    num_tf = Pipeline([("impute", SimpleImputer(strategy="median"))])

    # sklearn sürüm uyumu (sparse_output vs sparse)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)

    cat_tf = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
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
    pipe.fit(X_train, y_train)

    # === Değerlendirme ===
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
    clsrep = classification_report(y_test, y_pred_best, digits=4)
    print(clsrep)

    # Aşırı iyi metrik uyarısı
    if ap > 0.95:
        print("⚠️  UYARI: PR-AUC > 0.95 — tipik sızıntı göstergesi. "
              "Feature set ve zaman splitini yeniden kontrol edin.")

    # === Çıktılar ===
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.clsrep_out.parent.mkdir(parents=True, exist_ok=True)
    args.prcurve_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, args.model_out)

    metrics = {
        "data_file": str(src),
        "freq": "1D",
        "time_col": time_col,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "n_features_num": len(num_cols), "n_features_cat": len(cat_cols),
        "class_balance_train": {"pos": int(pos), "neg": int(neg), "scale_pos_weight": spw},
        "pr_auc": ap, "f1_best": f1_best, "best_threshold": best_th,
        "features_used": feat_cols[:200] + (["..."] if len(feat_cols) > 200 else []),
    }
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(args.clsrep_out, "w", encoding="utf-8") as f:
        f.write(clsrep)

    pr_curve = pd.DataFrame({"precision": prec, "recall": rec})
    pr_curve.to_csv(args.prcurve_out, index=False)

    print(f"[OK] Model   → {args.model_out}")
    print(f"[OK] Metrik  → {args.report_out}")
    print(f"[OK] Rapor   → {args.clsrep_out}")
    print(f"[OK] PR-curve→ {args.prcurve_out}")


if __name__ == "__main__":
    main()
