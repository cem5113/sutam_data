#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aggregate_all.py — runtime-anchored multi-window aggregation
Girdi  : Saatlik full-grid parquet (GEOID, dt, crime_count?, Y_label?, diğer numerikler)
Çıktı  : sf_crime_grid_{3h,8h,1d,1w,1m}.parquet

Pencereler:
  3H, 8H: run-time gün başlangıcına göre hizalı bloklar (block_id üretir)
  1D: 1 günlük
  1W: 7 günlük
  1M: 30 günlük (takvim ayı değil)

Toplama:
  crime_count → SUM, diğer numerikler → MEAN
Etiket:
  Y_label = (pencere içinde crime_count > 0)
Priors (leakage-safe):
  28D ve 180D rolling
    - 3H/8H: keys = GEOID, day_of_week, block_id
    - 1D/1W/1M: keys = GEOID, day_of_week
Kullanım:
  python -u aggregate_all.py --input sf_crime_grid_full_labeled.parquet --freqs "3H,8H,1D,1W,1M" --tz America/Los_Angeles
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

CAL_DROP = {"dt_local"}  # varsa atarız


# -------------------------
# Zaman yardımcıları
# -------------------------
def _now_tz(tz: Optional[str]) -> pd.Timestamp:
    return pd.Timestamp.now(tz) if tz else pd.Timestamp.utcnow().tz_localize("UTC")

def _anchored_floor(series: pd.Series, freq: str, tz: Optional[str]) -> pd.Series:
    """
    Pencereleri takvim başlangıcına değil, çalıştırma anının gün başına (origin) hizalar.
    """
    anchor = _now_tz(tz)
    origin = anchor.floor("D")
    s = pd.to_datetime(series)
    try:
        # pandas >= 2.1
        return s.dt.floor(freq=freq, origin=origin)
    except TypeError:
        # Eski pandas için yaklaşık çözüm
        delta = (s - origin).dt.total_seconds()
        step = pd.Timedelta(freq).total_seconds()
        bins = np.floor(delta / step) * step
        return (origin + pd.to_timedelta(bins, unit="s")).dt.tz_convert("UTC") if s.dt.tz is not None \
               else (origin + pd.to_timedelta(bins, unit="s")).dt.tz_localize("UTC")

def _anchored_day0(dt_local: pd.Series, tz: Optional[str]) -> pd.Series:
    return _anchored_floor(dt_local, "1D", tz)

def to_local(s_utc: pd.Series, tz: Optional[str]) -> pd.Series:
    s_utc = pd.to_datetime(s_utc, utc=True)
    return s_utc.dt.tz_convert(tz) if tz else s_utc

def _ns64(x: pd.Series) -> pd.Series:
    """tz-aware datetime64'ü int64 ns'e güvenli çevir."""
    x = pd.to_datetime(x)
    # .view('int64') bazı sürümlerde uyarı veriyor; astype daha güvenli
    return x.astype("int64")


def make_t0_and_block(dt_local: pd.Series, freq: str, tz: Optional[str]) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    3H/8H: block_id üretir (3H→0..7, 8H→0..2) — run-time day0'a göre hizalanır
    1D/1W/1M: block_id yok
    """
    if freq == "3H":
        t0_local = _anchored_floor(dt_local, "3H", tz)
        day0 = _anchored_day0(dt_local, tz)
        hrs_since = (_ns64(dt_local) - _ns64(day0)) / 3_600_000_000_000
        block_id = np.floor(hrs_since / 3.0).astype("int8")
    elif freq == "8H":
        t0_local = _anchored_floor(dt_local, "8H", tz)
        day0 = _anchored_day0(dt_local, tz)
        hrs_since = (_ns64(dt_local) - _ns64(day0)) / 3_600_000_000_000
        block_id = np.floor(hrs_since / 8.0).astype("int8")
    elif freq == "1D":
        t0_local = _anchored_floor(dt_local, "1D", tz)
        block_id = None
    elif freq == "1W":
        t0_local = _anchored_floor(dt_local, "7D", tz)
        block_id = None
    elif freq == "1M":
        t0_local = _anchored_floor(dt_local, "30D", tz)
        block_id = None
    else:
        raise ValueError(f"Desteklenmeyen freq: {freq}")
    return t0_local, block_id


# -------------------------
# Özellik & prior yardımcıları
# -------------------------
def add_calendar_cols(df: pd.DataFrame, t0_local_col: str) -> pd.DataFrame:
    df["year"]        = df[t0_local_col].dt.year.astype("int16")
    df["month"]       = df[t0_local_col].dt.month.astype("int8")
    df["day_of_week"] = df[t0_local_col].dt.dayofweek.astype("int8")
    df["hour_start"]  = df[t0_local_col].dt.hour.astype("int8")
    return df

def prior_rolling(df: pd.DataFrame, window: str, suffix: str, keys: List[str]) -> pd.DataFrame:
    """
    Leakage-safe prior: Y_label rolling SUM, shift(1)
    """
    df = df.sort_values(["GEOID", "t0"]).copy()
    keys = [k for k in keys if k in df.columns]
    grp_cols = ["GEOID"] + keys

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("t0").set_index("t0")
        cnt = g["Y_label"].rolling(window=window).sum().shift(1).fillna(0.0)
        out = g.copy()
        out[f"prior_cnt_{suffix}"] = cnt.to_numpy().astype("float32")
        return out.reset_index()

    out = df.groupby(grp_cols, group_keys=False).apply(_roll)
    hours_in_window = float(pd.Timedelta(window) / pd.Timedelta("1h"))
    out[f"prior_p_{suffix}"] = (out[f"prior_cnt_{suffix}"] / hours_in_window).astype("float32")
    return out

def aggregate_one(df_hourly: pd.DataFrame, freq: str, tz: Optional[str]) -> pd.DataFrame:
    """
    Tek bir frekans için toplama + etiket + priors üretir.
    """
    # 1) Zaman hazırlığı (UTC→opsiyonel yerel)
    df = df_hourly.copy()
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    dt_local = to_local(df["dt"], tz)
    t0_local, block_id = make_t0_and_block(dt_local, freq, tz)

    # 2) Toplama
    sdf = df.copy()
    sdf["t0_local"] = t0_local
    if block_id is not None:
        sdf["block_id"] = block_id

    base_exclude = {"GEOID", "dt", "Y_label"}
    num_cols = [c for c in sdf.columns if c not in base_exclude and pd.api.types.is_numeric_dtype(sdf[c])]

    agg_map = {}
    if "crime_count" in sdf.columns:
        agg_map["crime_count"] = "sum"
    other_nums = [c for c in num_cols if c != "crime_count"]
    agg_map.update({c: "mean" for c in other_nums})

    group_keys = ["GEOID", "t0_local"] + (["block_id"] if "block_id" in sdf.columns else [])
    agg = sdf.groupby(group_keys, as_index=False).agg(agg_map)

    # 3) Etiket
    if "crime_count" not in agg.columns:
        raise SystemExit("Beklenen kolon 'crime_count' yok (input saatlik gridde üretmiş olmalı).")
    agg["Y_label"] = (agg["crime_count"] > 0).astype("int8")

    # 4) t0 (UTC) üret
    t0_loc = pd.to_datetime(agg["t0_local"], errors="coerce")
    if getattr(t0_loc.dt, "tz", None) is None:
        agg["t0"] = t0_loc.dt.tz_localize(tz or "UTC").dt.tz_convert("UTC")
    else:
        agg["t0"] = t0_loc.dt.tz_convert("UTC")

    # 5) Takvim kolonları
    agg = add_calendar_cols(agg, "t0_local")

    # 6) Çıktıda kalacak kolonları seç
    keep_cols = ["GEOID", "t0", "t0_local", "Y_label", "crime_count",
                 "year", "month", "day_of_week", "hour_start"]
    if "block_id" in agg.columns:
        keep_cols.append("block_id")
    keep_cols += other_nums
    keep_cols = [c for c in keep_cols if c in agg.columns]
    agg = agg.loc[:, ~agg.columns.duplicated()]
    agg = agg[keep_cols].copy()

    # Tip güvenliği
    for col in ("day_of_week", "block_id"):
        if col in agg.columns:
            agg[col] = pd.to_numeric(agg[col], errors="coerce").fillna(-1).astype("int8")

    # 7) Priors
    if "block_id" in agg.columns:
        season_keys = ["day_of_week", "block_id"]  # 3H/8H
    else:
        season_keys = ["day_of_week"]              # 1D/1W/1M

    agg = prior_rolling(agg, window="28D",  suffix="28d",  keys=season_keys)
    agg = prior_rolling(agg, window="180D", suffix="180d", keys=season_keys)

    # 8) Sırala & temizle
    order_cols = ["GEOID", "t0"] + (["block_id"] if "block_id" in agg.columns else [])
    agg = agg.sort_values(order_cols).reset_index(drop=True)
    if "t0_local" in agg.columns:
        agg = agg.drop(columns=["t0_local"])

    return agg


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Saatlik full-grid parquet (GEOID×dt)")
    p.add_argument("--freqs", type=str, default="3H,8H,1D,1W,1M", help="Virgüllü liste")
    p.add_argument("--tz",    type=str, default=None, help="Yerel zaman örn. America/Los_Angeles")
    return p.parse_args()


def main():
    args = parse_args()
    src: Path = args.input
    if not src.exists():
        raise SystemExit(f"Girdi yok: {src.resolve()}")

    # 0) Oku + şema doğrula
    df_h = pd.read_parquet(src)
    need = {"dt", "GEOID"}
    miss = need - set(df_h.columns)
    if miss:
        raise SystemExit(f"Eksik kolon(lar): {miss}")

    # Eski yardımcı kolonu at
    drop_candidates = [c for c in CAL_DROP if c in df_h.columns]
    if drop_candidates:
        df_h = df_h.drop(columns=drop_candidates)

    # Frekansları hazırla
    freqs: List[str] = [f.strip().upper() for f in args.freqs.split(",") if f.strip()]
    valid = {"3H", "8H", "1D", "1W", "1M"}
    bad = [f for f in freqs if f not in valid]
    if bad:
        raise SystemExit(f"Geçersiz frekans(lar): {bad} | izinli: {sorted(valid)}")

    for f in freqs:
        print(f"▶️  Aggregate {f}")
        out_path = Path(f"sf_crime_grid_{f.lower()}.parquet")
        agg = aggregate_one(df_h, f, args.tz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(out_path, index=False)
        y_rate = float(agg["Y_label"].mean())
        print(f"[OK] {out_path}  rows={len(agg):,}  GEOID={agg['GEOID'].nunique():,}  y1%≈{100*y_rate:.4f}")
        print(f"[INFO] t0 UTC range: {agg['t0'].min()} → {agg['t0'].max()}")

if __name__ == "__main__":
    main()
