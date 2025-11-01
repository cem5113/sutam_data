#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aggregate_all.py — daily-aware aggregation (supports 1D / 1W / 1M from daily input)
Girdi  : Daily grid (GEOID, t0, crime_count, ... ) veya saatlik grid (GEOID, dt, ...)
Çıktı  : sf_crime_grid_{1d,1w,1m}.parquet
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

ID_COLS = {"GEOID"}
TIME_COLS = {"dt", "t0", "t0_local"}
CAL_COLS = {"year", "month", "day_of_week", "hour_start", "block_id"}
LABEL_COLS = {"Y_label"}
PROTECTED = ID_COLS | TIME_COLS | CAL_COLS | LABEL_COLS | {"crime_count"}

# ------------ helpers ------------
def _ensure_geoid(df: pd.DataFrame) -> pd.DataFrame:
    if "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str)
        return df
    if "geoid" in df.columns:
        df = df.rename(columns={"geoid": "GEOID"})
        df["GEOID"] = df["GEOID"].astype(str)
        return df
    raise KeyError(f"GEOID kolonu yok. Kolonlar: {list(df.columns)[:20]}")

def add_calendar_cols(df: pd.DataFrame, tcol: str) -> pd.DataFrame:
    s = pd.to_datetime(df[tcol], utc=True)
    df["year"]        = s.dt.year.astype("int16")
    df["month"]       = s.dt.month.astype("int8")
    df["day_of_week"] = s.dt.dayofweek.astype("int8")
    df["hour_start"]  = np.int8(0)
    return df

def prior_rolling(df: pd.DataFrame, window: str, suffix: str, keys: list[str], time_col: str = "t0") -> pd.DataFrame:
    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' yok.")
    if "Y_label" not in df.columns:
        raise KeyError("prior_rolling için 'Y_label' zorunlu.")
    df = _ensure_geoid(df).sort_values(["GEOID", time_col]).copy()
    grp_cols = ["GEOID"] + [k for k in keys if k in df.columns]

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(time_col).set_index(time_col)
        cnt = g["Y_label"].rolling(window=window).sum().shift(1).fillna(0.0)
        g = g.reset_index()
        g[f"prior_cnt_{suffix}"] = cnt.to_numpy().astype("float32")
        return g

    try:
        out = df.groupby(grp_cols, group_keys=False).apply(_roll, include_groups=False)
    except TypeError:
        out = df.groupby(grp_cols, group_keys=False).apply(_roll)

    hours_in_window = float(pd.Timedelta(window) / pd.Timedelta("1h"))
    out[f"prior_p_{suffix}"] = (out[f"prior_cnt_{suffix}"] / hours_in_window).astype("float32")
    return _ensure_geoid(out)

# ------------ daily input path ------------
def _normalize_1d(df1d: pd.DataFrame) -> pd.DataFrame:
    """Daily girdiyi (GEOID,t0,crime_count,...) 1D şemasına sabitler."""
    df = _ensure_geoid(df1d.copy())
    if "t0" not in df.columns:
        raise SystemExit("1D normalize: 't0' kolonu yok.")
    df["t0"] = pd.to_datetime(df["t0"], utc=True, errors="coerce")
    if "crime_count" not in df.columns:
        # varsa binary Y_label’dan üret: ama tavsiye edilen crime_count’un olması
        if "Y_label" in df.columns:
            df["crime_count"] = df["Y_label"].astype("int16")
        else:
            raise SystemExit("1D normalize: 'crime_count' veya 'Y_label' bekleniyordu.")
    df["Y_label"] = (df["crime_count"] > 0).astype("int8")
    df = add_calendar_cols(df, "t0")
    keep = ["GEOID","t0","crime_count","Y_label","year","month","day_of_week","hour_start"]
    # diğer numerikler korunur
    num_others = [c for c in df.columns if c not in PROTECTED and pd.api.types.is_numeric_dtype(df[c])]
    keep += num_others
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df[ [c for c in keep if c in df.columns] ].copy()
    df = df.sort_values(["GEOID","t0"]).reset_index(drop=True)
    # priors
    df = prior_rolling(df, "28D",  "28d",  keys=["day_of_week"], time_col="t0")
    df = prior_rolling(df, "180D", "180d", keys=["day_of_week"], time_col="t0")
    return df

def _up_agg(df1d: pd.DataFrame, target: str) -> pd.DataFrame:
    """1D → 1W ya da 1M."""
    df = _normalize_1d(df1d)
    if target == "1W":
        # 7 günlük blok (UTC tabanlı)
        t_floor = pd.to_datetime(df["t0"], utc=True).dt.floor("7D")
    elif target == "1M":
        # ay başı
        t_floor = pd.to_datetime(df["t0"], utc=True).dt.to_period("M").dt.to_timestamp("MS", tz="UTC")
    else:
        raise ValueError("target must be 1W or 1M")

    sdf = df.copy()
    sdf["t0"] = t_floor

    # numerikler: crime_count → sum, diğer numerikler → mean
    num_cols = [c for c in sdf.columns if c not in PROTECTED and pd.api.types.is_numeric_dtype(sdf[c])]
    agg_map = {"crime_count": "sum"}
    for c in num_cols:
        agg_map[c] = "mean"

    out = (sdf.groupby(["GEOID", "t0"], as_index=False)
               .agg(agg_map))

    out["Y_label"] = (out["crime_count"] > 0).astype("int8")
    out = add_calendar_cols(out, "t0")

    # priors (weekly/monthly’de de leakage-safe)
    out = prior_rolling(out, "28D",  "28d",  keys=["day_of_week"], time_col="t0")
    out = prior_rolling(out, "180D", "180d", keys=["day_of_week"], time_col="t0")
    out = out.sort_values(["GEOID","t0"]).reset_index(drop=True)
    return out

# ------------ CLI ------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Daily grid (GEOID×t0, crime_count...) veya hourly (GEOID×dt)")
    p.add_argument("--freqs", type=str, default="1D,1W,1M", help="Sadece 1D,1W,1M desteklenir")
    return p.parse_args()

def main():
    args = parse_args()
    src = args.input
    if not src.exists():
        raise SystemExit(f"Girdi yok: {src}")

    freqs: List[str] = [f.strip().upper() for f in args.freqs.split(",") if f.strip()]
    allowed = {"1D","1W","1M"}
    bad = [f for f in freqs if f not in allowed]
    if bad:
        raise SystemExit(f"Geçersiz frekans(lar): {bad} | izinli: {sorted(allowed)}")

    df0 = pd.read_parquet(src)
    df0 = _ensure_geoid(df0)

    if "t0" in df0.columns and "dt" not in df0.columns:
        # DAILY PATH
        if "1D" in freqs:
            out_1d = _normalize_1d(df0)
            out_1d.to_parquet("sf_crime_grid_1d.parquet", index=False)
            print(f"[OK] sf_crime_grid_1d.parquet  rows={len(out_1d):,}  GEOID={out_1d['GEOID'].nunique():,}")
        if "1W" in freqs:
            out_1w = _up_agg(df0, "1W")
            out_1w.to_parquet("sf_crime_grid_1w.parquet", index=False)
            print(f"[OK] sf_crime_grid_1w.parquet  rows={len(out_1w):,}  GEOID={out_1w['GEOID'].nunique():,}")
        if "1M" in freqs:
            out_1m = _up_agg(df0, "1M")
            out_1m.to_parquet("sf_crime_grid_1m.parquet", index=False)
            print(f"[OK] sf_crime_grid_1m.parquet  rows={len(out_1m):,}  GEOID={out_1m['GEOID'].nunique():,}")
        return

    # Eğer saatlik grid gelseydi burada işlenirdi (3H/8H kaldırıldı).
    raise SystemExit("Girdi daily (GEOID×t0) olmalı. Saatlik (dt) destek dışı bırakıldı.")

if __name__ == "__main__":
    main()
