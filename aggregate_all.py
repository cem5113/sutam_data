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
from typing import List, Sequence, Dict

import numpy as np
import pandas as pd

# ——— Sabitler
PROTECTED = {
    "GEOID", "t0", "t0_local", "Y_label",
    "year", "month", "day_of_week", "hour_start", "block_id",
    "crime_count"
}
SUM_COLS = {"crime_count"}  # pencerede SUM; diğer numerikler MEAN

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

def _as_utc_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")

def _add_calendar_from_t0(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.to_datetime(df["t0"], utc=True)
    df["year"]        = s.dt.year.astype("int16")
    df["month"]       = s.dt.month.astype("int8")
    df["day_of_week"] = s.dt.dayofweek.astype("int8")
    df["hour_start"]  = np.int8(0)  # 1D/1W/1M için 0
    return df

def week_start_utc(t0: pd.Series) -> pd.Series:
    # 7 günlük bloklar (fixed) — floor('7D') güvenli
    s = _as_utc_ts(t0)
    return s.dt.floor("7D")

def month_start_utc(t0: pd.Series) -> pd.Series:
    """
    Aylık başlangıcı güvenle üret (floor('MS')/Period → tz problemlerini önler).
    """
    s = _as_utc_ts(t0)
    ms = pd.to_datetime({"year": s.dt.year, "month": s.dt.month, "day": 1}, utc=True)
    return ms

def prior_rolling(df: pd.DataFrame, window: str, suffix: str, keys: List[str], time_col: str = "t0") -> pd.DataFrame:
    """
    df: GEOID, time_col (UTC), Y_label ... içerir.
    keys: ["day_of_week", ...]
    Sızıntısız: shift(1)
    """
    need = {"GEOID", time_col, "Y_label"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"prior_rolling için eksik kolon(lar): {miss}")

    df = df.copy()
    df["GEOID"] = df["GEOID"].astype(str)
    keys = [k for k in (keys or []) if k in df.columns]

    df = df.sort_values(["GEOID", time_col])
    grp_cols = ["GEOID", *keys]

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(time_col).set_index(time_col)
        cnt = g["Y_label"].rolling(window=window).sum().shift(1).fillna(0.0)
        out = g.copy()
        out[f"prior_cnt_{suffix}"] = cnt.astype("float32").to_numpy()
        return out

    try:
        tmp = df.groupby(grp_cols, group_keys=True).apply(_roll, include_groups=False)
    except TypeError:
        tmp = df.groupby(grp_cols, group_keys=True).apply(_roll)

    tmp = tmp.reset_index()
    if time_col not in tmp.columns:
        tmp.rename(columns={"index": time_col}, inplace=True)

    hours_in_window = float(pd.Timedelta(window) / pd.Timedelta("1h"))
    tmp[f"prior_p_{suffix}"] = (tmp[f"prior_cnt_{suffix}"] / hours_in_window).astype("float32")

    tmp["GEOID"] = tmp["GEOID"].astype(str)
    tmp[time_col] = pd.to_datetime(tmp[time_col], utc=True)
    return tmp

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

    df["t0"] = _as_utc_ts(df["t0"])
    if df["t0"].isna().any():
        raise SystemExit("1D normalize: t0 parse edilemedi (NaT var).")

    # Etiket: pencerede ≥1 olay?
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

    out = df[[c for c in df.columns if c != "t0_local"]].sort_values(["GEOID", "t0"]).reset_index(drop=True)
    return out

def _pick_numeric_cols(df: pd.DataFrame) -> List[str]:
    # PROTECTED hariç tüm numerikler
    return [c for c in df.columns if c not in PROTECTED and pd.api.types.is_numeric_dtype(df[c])]

def _series_or_zero(df: pd.DataFrame, col: str) -> pd.Series:
    """Kolon yoksa index uzunluğunda 0 serisi döndür."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series(0, index=df.index, dtype="float32")

def _agg_block(df: pd.DataFrame, key_time: str) -> pd.DataFrame:
    """
    GEOID × key_time bazında agregasyon (SUM/MEAN) + etiket + takvim + priors
    """
    num_cols = _pick_numeric_cols(df)

    # 1) crime_count'ı açıkça SUM olarak ekle (PROTECTED'ta olduğu için num_cols'a girmez)
    agg_map: Dict[str, str] = {}
    if "crime_count" in df.columns and pd.api.types.is_numeric_dtype(df["crime_count"]):
        agg_map["crime_count"] = "sum"

    # 2) Diğer numerikler: MEAN
    for c in num_cols:
        if c not in agg_map:
            agg_map[c] = "mean"

    gcols = ["GEOID", key_time]
    out = df.groupby(gcols, as_index=False).agg(agg_map) if agg_map else df.groupby(gcols, as_index=False).size()
    if "size" in out.columns and "crime_count" not in out.columns:
        # size kolonu istemiyoruz; sadece zaman damgasını taşıyacağız
        out = out.drop(columns=["size"])

    # Etiket (>=1 olay?)
    cc = _series_or_zero(out, "crime_count")
    out["Y_label"] = (cc > 0).astype("int8")

    # t0 olarak yazalım
    out = out.rename(columns={key_time: "t0"})
    out["t0"] = _as_utc_ts(out["t0"])

    # Takvim alanları
    out = _add_calendar_from_t0(out)

    # Priors (day_of_week ile)
    out = prior_rolling(out, "28D",  "28d",  keys=["day_of_week"], time_col="t0")
    out = prior_rolling(out, "180D", "180d", keys=["day_of_week"], time_col="t0")

    out = out.sort_values(["GEOID", "t0"]).reset_index(drop=True)
    return out

def _up_agg_1w(df1d: pd.DataFrame) -> pd.DataFrame:
    df = df1d.copy()
    df["__t_week"] = week_start_utc(df["t0"])
    return _agg_block(df, "__t_week")

def _up_agg_1m(df1d: pd.DataFrame) -> pd.DataFrame:
    df = df1d.copy()
    df["__t_month"] = month_start_utc(df["t0"])  # güvenli ay başlangıcı
    return _agg_block(df, "__t_month")

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

    req = [f.strip().upper() for f in args.freqs.split(",") if f.strip()]
    allowed = {"1D", "1W", "1M"}
    bad = [f for f in req if f not in allowed]
    if bad:
        raise SystemExit(f"Geçersiz frekans(lar): {bad} | izinli: {sorted(allowed)}")

    print(f"▶️  aggregate_all.py ({','.join(req)})")

    # Oku & normalize 1D
    df0 = pd.read_parquet(src)
    df0 = _ensure_geoid(df0)
    if "t0" not in df0.columns:
        raise SystemExit("Bu script günlük (1D) girdiler içindir; 't0' kolonu zorunlu.")
    out_1d = _normalize_1d(df0)

    if "1D" in req:
        _save(out_1d, Path("sf_crime_grid_1d.parquet"))
    if "1W" in req:
        out_1w = _up_agg_1w(out_1d)
        _save(out_1w, Path("sf_crime_grid_1w.parquet"))
    if "1M" in req:
        out_1m = _up_agg_1m(out_1d)
        _save(out_1m, Path("sf_crime_grid_1m.parquet"))

if __name__ == "__main__":
    main()
