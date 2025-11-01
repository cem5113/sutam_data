#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aggregate_all.py — 1D tabanlı 1D/1W/1M üretimi (UTC)
Girdi : Günlük grid parquet (GEOID, t0 [UTC], crime_count, ... numerikler)
Çıktı : sf_crime_grid_{1d,1w,1m}.parquet

Notlar
- 1D girdiniz yoksa (saatlik 'dt' ile) bu dosya yerine 1D üreten sürümü kullanın.
- Priors: 28D ve 180D, leakage-safe (shift(1)), anahtarlar:
    * 1D/1W/1M → GEOID × day_of_week
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

# ——— Sabitler
PROTECTED = {
    "GEOID", "t0", "t0_local", "Y_label",
    "year", "month", "day_of_week", "hour_start", "block_id",
    "crime_count"
}

# ——— Yardımcılar
def _ensure_geoid(df: pd.DataFrame) -> pd.DataFrame:
    if "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str)
        return df
    if "geoid" in df.columns:
        df = df.rename(columns={"geoid": "GEOID"})
        df["GEOID"] = df["GEOID"].astype(str)
        return df
    raise SystemExit("GEOID/geoid kolonu bulunamadı.")

def _add_calendar_from_t0(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.to_datetime(df["t0"], utc=True)
    df["year"]        = s.dt.year.astype("int16")
    df["month"]       = s.dt.month.astype("int8")
    df["day_of_week"] = s.dt.dayofweek.astype("int8")
    # 1D/1W/1M için hour_start = 0 tutulur
    df["hour_start"]  = np.int8(0)
    return df

def prior_rolling(
    df: pd.DataFrame,
    window: str,
    suffix: str,
    keys: Sequence[str],
    time_col: str = "t0",
) -> pd.DataFrame:
    """
    df: GEOID, time_col (UTC), Y_label içerir.
    keys: ["day_of_week", ...]
    """
    if time_col not in df.columns:
        raise SystemExit(f"prior_rolling: '{time_col}' kolonu yok.")
    if "Y_label" not in df.columns:
        raise SystemExit("prior_rolling: 'Y_label' kolonu yok.")
    df = _ensure_geoid(df).copy()
    df = df.sort_values(["GEOID", time_col])

    keys = [k for k in keys if k in df.columns]
    grp_cols = ["GEOID"] + keys

    hours_in_window = float(pd.Timedelta(window) / pd.Timedelta("1h"))

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(time_col).copy()
        s = g.set_index(time_col)["Y_label"].rolling(window=window).sum().shift(1).fillna(0.0)
        g[f"prior_cnt_{suffix}"] = s.to_numpy().astype("float32")
        g[f"prior_p_{suffix}"]   = (g[f"prior_cnt_{suffix}"] / hours_in_window).astype("float32")
        return g

    # group_keys=False → gruplayıp geri birleştir, kolonları koru
    try:
        out = df.groupby(grp_cols, group_keys=False).apply(_roll, include_groups=False)
    except TypeError:
        out = df.groupby(grp_cols, group_keys=False).apply(_roll)

    return out

def _normalize_1d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Girdiyi doğrular, 1D yapıyı standartlaştırır, etiket ve priors ekler.
    Beklenen minimum kolonlar: GEOID, t0, crime_count
    """
    df = _ensure_geoid(df).copy()

    need = {"GEOID", "t0", "crime_count"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"1D normalize: eksik kolon(lar): {miss}")

    # t0 UTC normalize
    df["t0"] = pd.to_datetime(df["t0"], utc=True, errors="coerce")
    if df["t0"].isna().any():
        raise SystemExit("1D normalize: t0 parse edilemedi (NaT var).")

    # Etiket: günlük pencerede >=1 olay?
    df["Y_label"] = (pd.to_numeric(df["crime_count"], errors="coerce").fillna(0) > 0).astype("int8")

    # Takvim kolonları
    df = _add_calendar_from_t0(df)

    # Tip güvenliği
    for c in ("day_of_week", "hour_start", "month"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype("int8")
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(-1).astype("int16")

    # Priors (leakage-safe)
    df = prior_rolling(df, "28D",  "28d",  keys=["day_of_week"], time_col="t0")
    df = prior_rolling(df, "180D", "180d", keys=["day_of_week"], time_col="t0")

    # Sırala & kolonları düzenle
    keep = [c for c in df.columns if c not in {"t0_local"}]
    out = df[keep].sort_values(["GEOID", "t0"]).reset_index(drop=True)
    return out

def _up_agg(df_1d: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    1D → 1W veya 1M toplama.
    - crime_count: SUM
    - diğer numerikler: MEAN
    """
    if target not in {"1W", "1M"}:
        raise ValueError("target 1W veya 1M olmalı.")

    df = df_1d.copy()

    # Zamanı hedef periyoda yuvarla (tz-aware, UTC)
    if target == "1W":
        t_floor = pd.to_datetime(df["t0"], utc=True).dt.floor("7D")
    else:  # "1M"
        # Aylık başa güvenli floor (Period → tz hatasını engeller)
        t_floor = pd.to_datetime(df["t0"], utc=True).dt.floor("MS")

    df["__t_floor"] = t_floor

    # Toplanacak numeric kolonlar (korumalı olanlar hariç)
    num_cols = [c for c in df.columns if c not in PROTECTED and pd.api.types.is_numeric_dtype(df[c])]
    agg_map = {"crime_count": "sum"}
    agg_map.update({c: "mean" for c in num_cols})

    gkeys = ["GEOID", "__t_floor"]
    agg = df.groupby(gkeys, as_index=False).agg(agg_map)

    # Etiket (>=1 olay?)
    agg["Y_label"] = (agg["crime_count"] > 0).astype("int8")

    # t0 olarak yazalım
    agg = agg.rename(columns={"__t_floor": "t0"})
    agg["t0"] = pd.to_datetime(agg["t0"], utc=True)

    # Takvim kolonları
    agg = _add_calendar_from_t0(agg)

    # Priors (day_of_week ile)
    agg = prior_rolling(agg, "28D",  "28d",  keys=["day_of_week"], time_col="t0")
    agg = prior_rolling(agg, "180D", "180d", keys=["day_of_week"], time_col="t0")

    # Çıkış
    out = agg.sort_values(["GEOID", "t0"]).reset_index(drop=True)
    return out

def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    y = float(df["Y_label"].mean())
    print(f"[OK] {path.name:>24}  rows={len(df):,}  GEOID={df['GEOID'].nunique():,}  y1%≈{100*y:.4f}")
    print(f"[INFO] t0 UTC range: {df['t0'].min()} → {df['t0'].max()}")

# ——— CLI
def _parse_args():
    p = argparse.ArgumentParser(description="SUTAM 1D tabanlı 1D/1W/1M üretimi")
    p.add_argument("--input", type=Path, required=True, help="Günlük parquet (GEOID, t0, crime_count, ...)")
    p.add_argument("--freqs", type=str, default="1D,1W,1M", help="Virgüllü liste: 1D,1W,1M")
    return p.parse_args()

def main():
    args = _parse_args()
    src: Path = args.input
    if not src.exists():
        raise SystemExit(f"Girdi yok: {src.resolve()}")

    # Oku
    df0 = pd.read_parquet(src)
    df0 = _ensure_geoid(df0)

    if "t0" not in df0.columns:
        raise SystemExit("Bu script günlük (1D) girdiler içindir; 't0' kolonu zorunlu.")

    req = [f.strip().upper() for f in args.freqs.split(",") if f.strip()]
    allowed = {"1D", "1W", "1M"}
    bad = [f for f in req if f not in allowed]
    if bad:
        raise SystemExit(f"Geçersiz frekans(lar): {bad} | izinli: {sorted(allowed)}")

    print(f"▶️  aggregate_all.py ({','.join(req)})")

    # 1D normalize
    out_1d = _normalize_1d(df0)
    if "1D" in req:
        _save(out_1d, Path("sf_crime_grid_1d.parquet"))

    # 1W / 1M yukarı toplama
    if "1W" in req:
        out_1w = _up_agg(out_1d, "1W")
        _save(out_1w, Path("sf_crime_grid_1w.parquet"))
    if "1M" in req:
        out_1m = _up_agg(out_1d, "1M")
        _save(out_1m, Path("sf_crime_grid_1m.parquet"))

if __name__ == "__main__":
    main()
