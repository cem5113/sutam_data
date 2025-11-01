#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_stacking_multi_windows.py  (REVIZE — sızıntı kapalı, zaman-bazlı split, OHE uyumlu)

- Frekanslar: 1D, 1W, 1M
- Split: son %20 test (zaman bazlı) — leakage yok
- Özellikler:
    * Kategorik: year, month, day_of_week, (varsa) block_id
    * Numerik: priors (prior_cnt_*, prior_p_*) + sızıntı içermeyen diğer numerikler
      (risk_* / metrics_* / crime_count / Y_label türevleri hariç)
- Base learners: (opsiyonel) XGB, (opsiyonel) LGBM, RF, LR  →  Meta: LR
- Çıktılar:
    models/sutam_stack_{freq}.joblib
    models/sutam_{freq}.joblib              (uyumluluk için aynı kopya)
    reports/metrics_{freq}.json
    reports/pr_curve_{freq}.csv             (precision, recall)
    reports/classification_report_{freq}.txt
    reports/metrics_summary_multi_windows.json (toplu özet)

Kullanım örn.:
    python train_stacking_multi_windows.py --freqs 1D,1W --dir . --prefix sf_crime_grid_
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report,
    precision_recall_curve, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, StackingClassifier

# İsteğe bağlı base learners
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


# ===== Parametreler / Yollar =====
RANDOM_STATE = 42

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=Path("."), help="Parquet dizini")
    p.add_argument("--prefix", type=str, default="sf_crime_grid_", help="Dosya öneki")
    p.add_argument("--freqs", type=str, default="1D", help="Virgüllü: 1D,1W,1M")
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--reports", type=Path, default=Path("reports"))
    return p.parse_args()


# ===== Yardımcılar =====
LEAK_PREFIXES = ("risk_", "metrics_")
LEAK_FORBIDDEN_SUBSTR = ("y_label", "ylabel", "label", "crime_count", "hr_cnt", "daily_cnt")

def _ohe():
    from sklearn.preprocessing import OneHotEncoder
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)

def ensure_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, utc=True, errors="coerce")
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    return s

def _valid_cols(df_train: pd.DataFrame, cols: List[str]) -> List[str]:
    ok = []
    for c in cols:
        s = df_train[c]
        if s.notna().any() and (s.nunique(dropna=True) > 1):
            ok.append(c)
    return ok

def _filter_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Sızıntı güvenli numerik & kategorik listeleri döndür."""
    # Kategorikler
    cat_candidates = ["year", "month", "day_of_week"]
    if "block_id" in df.columns:
        cat_candidates.append("block_id")
    cat_cols = [c for c in cat_candidates if c in df.columns]

    # Numerikler (yasak önek/alt-string içermeyenler)
    drop_set = {"GEOID", "t0", "Y_label", "dt_local", "crime_count", "hr_cnt", "daily_cnt"}
    num_cols = []
    for c in df.columns:
        cl = c.lower()
        if c in drop_set or c in cat_cols:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if any(cl.startswith(p) for p in LEAK_PREFIXES):
            continue
        if any(s in cl for s in LEAK_FORBIDDEN_SUBSTR):
            continue
        num_cols.append(c)

    return num_cols, cat_cols

def _build_stacking(num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    num_tr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        # StandardScaler: sparse uyumlu (with_mean=False)
        ("scaler", StandardScaler(with_mean=False))
    ])
    cat_tr = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", _ohe())
    ])
    pre = ColumnTransformer(
        transformers=[("num", num_tr, num_cols), ("cat", cat_tr, cat_cols)],
        sparse_threshold=0.3,
        remainder="drop"
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
            n_estimators=500, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, objective="binary"
        )))
    estimators.append(("rf", RandomForestClassifier(
        n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1
    )))
    estimators.append(("lr", LogisticRegression(
        max_iter=2000, solver="saga", penalty="l2", n_jobs=-1
    )))

    meta = LogisticRegression(max_iter=2000, solver="liblinear")

    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=-1
    )

    return Pipeline([("pre", pre), ("mdl", clf)])

def _safe_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

def _best_f1(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float, float, float]:
    p, r, th = precision_recall_curve(y_true, proba)
    if len(p) <= 1 or len(r) <= 1:
        return 0.5, 0.0, float(p[-1] if len(p) else 0), float(r[-1] if len(r) else 0)
    f1s = (2 * p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-12)
    i = int(np.nanargmax(f1s))
    return float(th[i]), float(f1s[i]), float(p[i]), float(r[i])


# ===== Ana eğitim =====
def train_one(freq: str, dir_path: Path, prefix: str, models_dir: Path, reports_dir: Path) -> Dict:
    src = dir_path / f"{prefix}{freq.lower()}.parquet"
    if not src.exists():
        raise FileNotFoundError(f"[{freq}] Girdi yok: {src}")

    df = pd.read_parquet(src)
    need = {"GEOID", "t0", "Y_label"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"[{freq}] Eksik kolon(lar): {miss}")

    df["t0"] = ensure_utc(df["t0"])
    df = df.sort_values("t0").reset_index(drop=True)

    # Zaman bazlı split
    cut = int(len(df) * 0.80)
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    # Takvim kolonları yoksa üret
    for c, fn in (("year", lambda s: s.dt.year.astype("int16")),
                  ("month", lambda s: s.dt.month.astype("int8")),
                  ("day_of_week", lambda s: s.dt.dayofweek.astype("int8"))):
        if c not in df.columns:
            df[c] = fn(df["t0"])
            train[c] = df.loc[train.index, c]
            test[c]  = df.loc[test.index, c]

    num_cols, cat_cols = _filter_features(df)
    # Train bazlı sabit/boş kolonları ele
    num_cols = _valid_cols(train, num_cols)
    cat_cols = [c for c in cat_cols if c in train.columns]

    if not (num_cols or cat_cols):
        raise SystemExit(f"[{freq}] Kullanılabilir feature yok (sızıntı/temizlik sonrası).")

    X_train = train[cat_cols + num_cols]
    y_train = train["Y_label"].astype(np.int8).values
    X_test  = test[cat_cols + num_cols]
    y_test  = test["Y_label"].astype(np.int8).values

    pipe = _build_stacking(num_cols=num_cols, cat_cols=cat_cols)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1].astype(np.float32)
    ap = float(average_precision_score(y_test, proba))
    th, f1, p_at, r_at = _best_f1(y_test, proba)
    y_pred = (proba >= th).astype(np.int8)

    # Raporlar
    cls_rep = classification_report(y_test, y_pred, digits=4)
    cm = _safe_confusion(y_test, y_pred)

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # MODELLER: stacking adı + uyumluluk adı
    import joblib
    joblib.dump(pipe, models_dir / f"sutam_stack_{freq.lower()}.joblib")
    joblib.dump(pipe, models_dir / f"sutam_{freq.lower()}.joblib")  # risk_forecast.py ile uyum

    # PR eğrisi (precision/recall)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    pd.DataFrame({"precision": prec, "recall": rec}).to_csv(
        reports_dir / f"pr_curve_{freq.lower()}.csv", index=False
    )

    (reports_dir / f"classification_report_{freq.lower()}.txt").write_text(cls_rep, encoding="utf-8")

    metrics = {
        "freq": freq,
        "data_file": str(src),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features_num": len(num_cols),
        "n_features_cat": len(cat_cols),
        "features_num": num_cols,
        "features_cat": cat_cols,
        "class_balance_test": {
            "y1_rate": float(y_test.mean()),
            "y1_count": int((y_test == 1).sum()),
            "y0_count": int((y_test == 0).sum())
        },
        "pr_auc": ap,
        "best_threshold": th,
        "f1_best": float(f1),
        "precision_at_best": float(p_at),
        "recall_at_best": float(r_at),
        "confusion_best": cm,
        "has_xgb": HAS_XGB,
        "has_lgbm": HAS_LGBM,
    }
    (reports_dir / f"metrics_{freq.lower()}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                                             encoding="utf-8")

    print(f"[RESULT][{freq}] PR-AUC={ap:.4f} | Best-F1@{th:.3f}={f1:.4f} | test y=1 %{100*y_test.mean():.3f}")
    if ap > 0.95:
        print(f"[WARN][{freq}] PR-AUC aşırı yüksek — sızıntı şüphesi (feature set & split’i tekrar kontrol edin).")

    return metrics


def main():
    args = parse_args()
    freqs = [f.strip().upper() for f in (args.freqs or "1D").split(",") if f.strip()]
    allowed = {"1D", "1W", "1M"}
    freqs = [f for f in freqs if f in allowed]
    if not freqs:
        raise SystemExit("Geçerli frekans verilmedi. Örn: --freqs 1D veya --freqs 1D,1W")

    summary = {}
    for f in freqs:
        try:
            print(f"[TRAIN] {f}")
            m = train_one(f, args.dir, args.prefix, args.models, args.reports)
            summary[f] = m
        except Exception as e:
            print(f"[WARN] {f} eğitiminde hata: {e}")

    (args.reports / "metrics_summary_multi_windows.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[DONE] metrics_summary_multi_windows.json yazıldı.")


if __name__ == "__main__":
    main()
