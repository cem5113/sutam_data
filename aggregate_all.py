#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aggregate_all.py — multi-window aggregation (supports hourly *or* daily input)
Girdi  :
  - Saatlik full-grid parquet (GEOID, dt, crime_count?, Y_label?, diğer numerikler) ➊
  - Günlük grid parquet (GEOID, t0, crime_count?, Y_label?, diğer numerikler)       ➋
Çıktı  : sf_crime_grid_{3h,8h,1d,1w,1m}.parquet

Notlar:
- ➋ Günlük girdiyle **sadece 1D** üretilebilir (3H/8H/1W/1M runtime-anchored olduğundan hourly gerekir).
- ➊ Saatlik girdide tüm frekanslar desteklenir.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

CAL_DROP = {"dt_local"}  # varsa atarız
ID_COLS = {"GEOID"}
TIME_COLS = {"dt", "t0", "t0_local"}
CAL_COLS = {"year", "month", "day_of_week", "hour", "hour_start", "season", "block_id"}
LABEL_COLS = {"Y_label"}
PROTECTED = ID_COLS | TIME_COLS | CAL_COLS | LABEL_COLS | {"crime_count"}


# -------------------------
# Şema algılama
# -------------------------
def _detect_granularity(cols: set[str]) -> str:
    if "dt" in cols:
        return "hourly"
    if "t0" in cols:
        return "daily"
    return "unknown"


# -------------------------
# Zaman yardımcıları
# -------------------------
def _now_tz(tz: Optional[str]) -> pd.Timestamp:
    return pd.Timestamp.now(tz) if tz else pd.Timestamp.utcnow().tz_localize("UTC")

def _anchored_floor(series: pd.Series, freq: str, tz: Optional[str]) -> pd.Series:
    """Pencereleri çalıştırma anının gün başına (origin) hizalar (3H/8H/1W/1M)."""
    anchor = _now_tz(tz)
    origin = anchor.floor("D")
    s = pd.to_datetime(series)
    try:
        # pandas >= 2.1
        return s.dt.floor(freq=freq, origin=origin)
    except TypeError:
        # Eski pandas için yaklaşık çözüm
        delta = (s - origin).dt.total_seconds()
        step = pd.Timedelta(freq.lower()).total_seconds()
        bins = np.floor(delta / step) * step
        out = origin + pd.to_timedelta(bins, unit="s")
        if getattr(s.dt, "tz", None) is None:
            return out.dt.tz_localize("UTC")
        return out.dt.tz_convert("UTC")

def _anchored_day0(dt_local: pd.Series, tz: Optional[str]) -> pd.Series:
    return _anchored_floor(dt_local, "1D", tz)

def to_local(s_utc: pd.Series, tz: Optional[str]) -> pd.Series:
    s_utc = pd.to_datetime(s_utc, utc=True)
    return s_utc.dt.tz_convert(tz) if tz else s_utc

def _ns64(x: pd.Series) -> pd.Series:
    x = pd.to_datetime(x)
    return x.astype("int64")

def make_t0_and_block(dt_local: pd.Series, freq: str, tz: Optional[str]) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    1D: TAKVİM GÜNÜ (yerel TZ'de 00:00'a sabitle)
    Diğerleri: runtime-anchored
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
        # TAKVİM GÜNÜ
        t0_local = pd.to_datetime(dt_local).dt.floor("1D")
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

def add_calendar_cols_from_t0(df: pd.DataFrame, t0_col: str = "t0") -> pd.DataFrame:
    s = pd.to_datetime(df[t0_col], utc=True)
    df["year"]        = s.dt.year.astype("int16")
    df["month"]       = s.dt.month.astype("int8")
    df["day_of_week"] = s.dt.dayofweek.astype("int8")
    df["hour_start"]  = 0  # günlükte sabit; training için tutarlı kolon
    return df

def _ensure_geoid(df: pd.DataFrame) -> pd.DataFrame:
    """GEOID kolonunu garanti et (gerekirse 'geoid' -> 'GEOID' rename)."""
    cols = set(df.columns)
    if "GEOID" in cols:
        df["GEOID"] = df["GEOID"].astype(str)
        return df
    if "geoid" in cols:
        df = df.rename(columns={"geoid": "GEOID"})
        df["GEOID"] = df["GEOID"].astype(str)
        return df
    raise KeyError(f"GEOID kolonu yok. Mevcut kolonlar: {list(df.columns)[:20]}...")

def prior_rolling(df: pd.DataFrame, window: str, suffix: str, keys: list[str], time_col: str = "t0") -> pd.DataFrame:
    """
    df: GEOID, time_col (UTC), Y_label ... içerir.
    keys: ["day_of_week", ... (opsiyonel "block_id")]
    """
    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' bulunamadı. Kolonlar: {list(df.columns)[:20]}...")
    if "Y_label" not in df.columns:
        raise KeyError("prior_rolling için 'Y_label' zorunlu.")
    df = _ensure_geoid(df).copy()

    df = df.sort_values(["GEOID", time_col]).copy()
    keys = [k for k in keys if k in df.columns]
    grp_cols = ["GEOID"] + keys

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        # Grup anahtarlarını güvenceye al (pandas sürüm farklılıklarına karşı)
        # g.name -> ('GEOID', *keys) veya tekil değer olabilir
        if hasattr(g, "name"):
            name = g.name
            if not isinstance(name, tuple):
                name = (name,)
        else:
            name = tuple()

        # Beklenen sırada değerleri çöz
        key_vals = {}
        if len(name) == len(grp_cols):
            key_vals = dict(zip(grp_cols, name))
        else:
            # Yedek: g içinden oku (çoğu sürümde kolonlar mevcut)
            for k in grp_cols:
                if k in g.columns:
                    key_vals[k] = g[k].iloc[0]

        g = g.sort_values(time_col).set_index(time_col)
        cnt = g["Y_label"].rolling(window=window).sum().shift(1).fillna(0.0)

        out = g.copy()
        out[f"prior_cnt_{suffix}"] = cnt.to_numpy().astype("float32")
        out = out.reset_index()

        # Grup anahtarlarını açıkça ekle
        for k, v in key_vals.items():
            if k not in out.columns:
                out[k] = v
        return out

    try:
        out = df.groupby(grp_cols, group_keys=False).apply(_roll, include_groups=False)
    except TypeError:
        out = df.groupby(grp_cols, group_keys=False).apply(_roll)

    hours_in_window = float(pd.Timedelta(window) / pd.Timedelta("1h"))
    out[f"prior_p_{suffix}"] = (out[f"prior_cnt_{suffix}"] / hours_in_window).astype("float32")

    # Sütunları standartla ve güvenceye al
    out = _ensure_geoid(out)
    out = out.sort_values(["GEOID", time_col]).reset_index(drop=True)
    return out
  
# -------------------------
# Saatlik girdiden tek-frekans agregasyon
# -------------------------
def aggregate_one_hourly(df_hourly: pd.DataFrame, freq: str, tz: Optional[str]) -> pd.DataFrame:
    """Saatlik (dt) girdi → freq çıktısı (3H/8H/1D/1W/1M)."""
    df = df_hourly.copy()
    df = _ensure_geoid(df)

    if "dt" not in df.columns:
        raise SystemExit("Girdi saatlik grid 'dt' kolonunu içermiyor.")
    df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce")
    if df["dt"].isna().all():
        raise SystemExit("'dt' çözümlenemedi (hepsi NaT).")

    dt_local = to_local(df["dt"], tz)
    t0_local, block_id = make_t0_and_block(dt_local, freq, tz)

    sdf = df.copy()
    sdf["t0_local"] = t0_local
    if block_id is not None:
        sdf["block_id"] = block_id

    candidates = [c for c in sdf.columns if c not in PROTECTED]
    num_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(sdf[c])]

    agg_map = {}
    if "crime_count" in sdf.columns:
        agg_map["crime_count"] = "sum"
    other_nums = [c for c in num_cols if c != "crime_count"]
    if other_nums:
        agg_map.update({c: "mean" for c in other_nums})

    group_keys = ["GEOID", "t0_local"] + (["block_id"] if "block_id" in sdf.columns else [])
    agg = sdf.groupby(group_keys, as_index=False).agg(agg_map)

    if "crime_count" not in agg.columns:
        raise SystemExit("Beklenen kolon 'crime_count' yok (input saatlik gridde üretmiş olmalı).")
    agg["Y_label"] = (agg["crime_count"] > 0).astype("int8")

    t0_loc = pd.to_datetime(agg["t0_local"], errors="coerce")
    if getattr(t0_loc.dt, "tz", None) is None:
        agg["t0"] = t0_loc.dt.tz_localize(tz or "UTC").dt.tz_convert("UTC")
    else:
        agg["t0"] = t0_loc.dt.tz_convert("UTC")

    agg = add_calendar_cols(agg, "t0_local")

    keep_base = ["GEOID", "t0", "t0_local", "Y_label", "crime_count",
                 "year", "month", "day_of_week", "hour_start"]
    if "block_id" in agg.columns:
        keep_base.append("block_id")

    other_nums_clean = [c for c in other_nums if c not in PROTECTED]
    keep_cols = keep_base + other_nums_clean

    seen, ordered = set(), []
    for c in keep_cols:
        if c in agg.columns and c not in seen:
            seen.add(c)
            ordered.append(c)

    agg = agg.loc[:, ~agg.columns.duplicated()]
    agg = agg[ordered].copy()
    agg = _ensure_geoid(agg)

    for col in ("day_of_week", "block_id"):
        if col in agg.columns:
            s = agg[col]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            agg[col] = pd.to_numeric(s, errors="coerce").fillna(-1).astype("int8")

    if "block_id" in agg.columns:
        season_keys = ["day_of_week", "block_id"]  # 3H/8H
    else:
        season_keys = ["day_of_week"]              # 1D/1W/1M

    agg = prior_rolling(agg, window="28D",  suffix="28d",  keys=season_keys, time_col="t0")
    agg = prior_rolling(agg, window="180D", suffix="180d", keys=season_keys, time_col="t0")

    order_cols = ["GEOID", "t0"] + (["block_id"] if "block_id" in agg.columns else [])
    agg = agg.sort_values(order_cols).reset_index(drop=True)
    if "t0_local" in agg.columns:
        agg = agg.drop(columns=["t0_local"])

    return agg


# -------------------------
# Günlük girdiden 1D agregasyon
# -------------------------
def aggregate_from_daily(df_daily: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    """
    Günlük (GEOID×t0) girdi → 1D çıktı (takvim + 28D/180D priors).
    df_daily beklenen kolonlar: GEOID, t0, (crime_count?), (Y_label?)
    """
    df = df_daily.copy()
    df = _ensure_geoid(df)

    need_any = {"GEOID", "t0"}
    miss = need_any - set(df.columns)
    if miss:
        raise SystemExit(f"Günlük girdi eksik kolon(lar): {miss}")

    df["t0"] = pd.to_datetime(df["t0"], utc=True, errors="coerce")
    if df["t0"].isna().all():
        raise SystemExit("'t0' çözümlenemedi (hepsi NaT).")

    if "Y_label" not in df.columns:
        if "crime_count" in df.columns:
            df["Y_label"] = (pd.to_numeric(df["crime_count"], errors="coerce").fillna(0) > 0).astype("int8")
        else:
            raise SystemExit("Günlük girdide Y_label yoksa, crime_count da gerekli.")

    df = add_calendar_cols_from_t0(df, "t0")

    season_keys = ["day_of_week"]
    df = prior_rolling(df, window="28D",  suffix="28d",  keys=season_keys, time_col="t0")
    df = prior_rolling(df, window="180D", suffix="180d", keys=season_keys, time_col="t0")

    keep = ["GEOID","t0","Y_label","crime_count",
            "year","month","day_of_week","hour_start",
            "prior_cnt_28d","prior_p_28d","prior_cnt_180d","prior_p_180d"]
    keep = [c for c in keep if c in df.columns]
    df = df.sort_values(["GEOID","t0"]).reset_index(drop=True)
    return df[keep].copy()


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="Saatlik full-grid parquet (GEOID×dt) veya günlük grid (GEOID×t0)")
    p.add_argument("--freqs", type=str, default="3H,8H,1D,1W,1M", help="Virgüllü liste")
    p.add_argument("--tz",    type=str, default=None, help="Yerel zaman örn. America/Los_Angeles")
    return p.parse_args()


def main():
    args = parse_args()
    src: Path = args.input
    if not src.exists():
        raise SystemExit(f"Girdi yok: {src.resolve()}")

    df0 = pd.read_parquet(src)
    df0 = _ensure_geoid(df0)

    gran = _detect_granularity(set(df0.columns))
    if gran == "unknown":
        raise SystemExit("Girdi şeması anlaşılamadı: 'dt' (hourly) ya da 't0' (daily) beklenir.")

    # Eski yardımcı kolonu at
    drop_candidates = [c for c in CAL_DROP if c in df0.columns]
    if drop_candidates:
        df0 = df0.drop(columns=drop_candidates)

    freqs: List[str] = [f.strip().upper() for f in args.freqs.split(",") if f.strip()]
    valid = {"3H", "8H", "1D", "1W", "1M"}
    bad = [f for f in freqs if f not in valid]
    if bad:
        raise SystemExit(f"Geçersiz frekans(lar): {bad} | izinli: {sorted(valid)}")

    if gran == "daily":
        # Günlük girdide yalnız 1D
        if any(f for f in freqs if f != "1D"):
            others = [f for f in freqs if f != "1D"]
            raise SystemExit(f"Günlük (t0) girdiyle {others} üretilemez. Lütfen --freqs '1D' kullanın.")
        print("▶️  Aggregate 1D (daily input)")
        out_path = Path("sf_crime_grid_1d.parquet")
        agg = aggregate_from_daily(df0, args.tz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(out_path, index=False)
        y_rate = float(agg["Y_label"].mean())
        print(f"[OK] {out_path}  rows={len(agg):,}  GEOID={agg['GEOID'].nunique():,}  y1%≈{100*y_rate:.4f}")
        print(f"[INFO] t0 UTC range: {agg['t0'].min()} → {agg['t0'].max()}")
        return

    # Hourly: tüm frekanslar desteklenir
    need = {"dt", "GEOID"}
    miss = need - set(df0.columns)
    if miss:
        raise SystemExit(f"Eksik kolon(lar): {miss}")

    for f in freqs:
        print(f"▶️  Aggregate {f}")
        out_path = Path(f"sf_crime_grid_{f.lower()}.parquet")
        agg = aggregate_one_hourly(df0, f, args.tz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(out_path, index=False)
        y_rate = float(agg["Y_label"].mean())
        print(f"[OK] {out_path}  rows={len(agg):,}  GEOID={agg['GEOID'].nunique():,}  y1%≈{100*y_rate:.4f}")
        print(f"[INFO] t0 UTC range: {agg['t0'].min()} → {agg['t0'].max()}")


if __name__ == "__main__":
    main()
