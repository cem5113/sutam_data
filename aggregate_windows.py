#!/usr/bin/env python3
# -*- coding: utf-8 -*-

aggregate_windows.py — calendar-day 1D, runtime-anchored others

import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

CAL_DROP = {"dt_local"}  # eski yardımcı kolon olursa temizleriz
PROTECTED = {"GEOID", "dt", "t0", "t0_local", "Y_label", "year", "month",
             "day_of_week", "hour", "hour_start", "season", "block_id", "crime_count"}

# -------------------------
# Helpers
# -------------------------
def _now_tz(tz: Optional[str]) -> pd.Timestamp:
    return (pd.Timestamp.now(tz) if tz else pd.Timestamp.utcnow().tz_localize("UTC"))

def _ensure_geoid(df: pd.DataFrame) -> pd.DataFrame:
    """GEOID kolonunu garanti et (gerekirse 'geoid' -> 'GEOID')."""
    if "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str)
        return df
    if "geoid" in df.columns:
        df = df.rename(columns={"geoid": "GEOID"})
        df["GEOID"] = df["GEOID"].astype(str)
        return df
    raise SystemExit("GEOID/geoid kolonu bulunamadı.")

def _anchored_floor(series: pd.Series, freq: str, tz: Optional[str]) -> pd.Series:
    """1W/1M için origin=run-day, 1D özel (takvim)."""
    anchor = _now_tz(tz)
    origin = anchor.floor("D")
    s = pd.to_datetime(series)
    try:
        return s.dt.floor(freq=freq, origin=origin)
    except TypeError:
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

def make_t0_and_block(dt_local: pd.Series, freq: str, tz: Optional[str]) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    1D: **takvim günü** (yerel TZ’de 00:00), block_id yok
    1W: 7 günlük pencereler (runtime-anchored)
    1M: 30 günlük pencereler (runtime-anchored)
    """
    if freq == "1D":
        # TAKVİM GÜNÜ (origin kullanılmaz)
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

def add_calendar_cols(df: pd.DataFrame, t0_local_col: str) -> pd.DataFrame:
    df["year"]        = df[t0_local_col].dt.year.astype("int16")
    df["month"]       = df[t0_local_col].dt.month.astype("int8")
    df["day_of_week"] = df[t0_local_col].dt.dayofweek.astype("int8")
    df["hour_start"]  = df[t0_local_col].dt.hour.astype("int8")
    return df

def prior_rolling(df: pd.DataFrame, window: str, suffix: str, keys: List[str]) -> pd.DataFrame:
    """
    time_col = 't0' zorunlu; GEOID garanti edilir; shift(1) ile sızıntı engellenir.
    """
    if "t0" not in df.columns:
        raise SystemExit("prior_rolling: 't0' kolonu yok.")
    if "Y_label" not in df.columns:
        raise SystemExit("prior_rolling: 'Y_label' kolonu yok.")
    df = _ensure_geoid(df).sort_values(["GEOID", "t0"]).copy()

    keys = [k for k in keys if k in df.columns]
    grp_cols = ["GEOID"] + keys

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("t0").set_index("t0")
        cnt = g["Y_label"].rolling(window=window).sum().shift(1).fillna(0.0)
        out = g.copy()
        out[f"prior_cnt_{suffix}"] = cnt.to_numpy().astype("float32")
        return out.reset_index()

    try:
        out = df.groupby(grp_cols, group_keys=False).apply(_roll, include_groups=False)
    except TypeError:
        out = df.groupby(grp_cols, group_keys=False).apply(_roll)

    hours_in_window = float(pd.Timedelta(window) / pd.Timedelta("1h"))
    out[f"prior_p_{suffix}"] = (out[f"prior_cnt_{suffix}"] / hours_in_window).astype("float32")
    return out

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Saatlik full-grid (GEOID×dt) parquet")
    p.add_argument("--out",   type=Path, required=True, help="Çıktı parquet")
    p.add_argument("--freq",  type=str, choices=["1D","1W","1M"], required=True)
    p.add_argument("--tz",    type=str, default=None, help="Yerel zaman (örn. America/Los_Angeles)")
    return p.parse_args()

def main():
    args = parse_args()
    src: Path = args.input
    dst: Path = args.out
    freq = args.freq

    if not src.exists():
        raise SystemExit(f"Girdi yok: {src.resolve()}")

    # 1) Oku + şemayı doğrula
    df = pd.read_parquet(src)
    df = _ensure_geoid(df)
    need = {"dt", "GEOID"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Eksik kolon(lar): {miss}")

    # Eski yardımcı kolonu at
    drop_candidates = [c for c in CAL_DROP if c in df.columns]
    if drop_candidates:
        df = df.drop(columns=drop_candidates)

    # 2) Zaman hazırlığı (UTC→opsiyonel yerel)
    df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce")
    if df["dt"].isna().all():
        raise SystemExit("'dt' çözümlenemedi (hepsi NaT).")
    dt_local = to_local(df["dt"], args.tz)
    t0_local, block_id = make_t0_and_block(dt_local, freq, args.tz)

    # 3) Toplama için hazırlık
    sdf = df.copy()
    sdf["t0_local"] = t0_local
    if block_id is not None:
        sdf["block_id"] = block_id

    # Numerikler: crime_count SUM, diğer numeric kolonlar MEAN
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

    # 4) Pencere etiketi (≥1 olay?)
    if "crime_count" not in agg.columns:
        raise SystemExit("crime_count kolonu bekleniyordu.")
    agg["Y_label"] = (agg["crime_count"] > 0).astype("int8")

    # 5) t0 (UTC) üret
    t0_loc = pd.to_datetime(agg["t0_local"], errors="coerce")
    if getattr(t0_loc.dt, "tz", None) is None:
        agg["t0"] = t0_loc.dt.tz_localize(args.tz or "UTC").dt.tz_convert("UTC")
    else:
        agg["t0"] = t0_loc.dt.tz_convert("UTC")

    # 6) Takvim kolonları (yerel t0 üzerinden)
    agg = add_calendar_cols(agg, "t0_local")

    # 7) Çıktıda tutulacaklar
    keep_cols = ["GEOID", "t0", "t0_local", "Y_label", "crime_count",
                 "year", "month", "day_of_week", "hour_start"]
    if "block_id" in agg.columns:
        keep_cols.append("block_id")
    keep_cols += [c for c in other_nums if c not in PROTECTED]
    keep_cols = [c for c in keep_cols if c in agg.columns]

    agg = agg.loc[:, ~agg.columns.duplicated()]
    agg = agg[keep_cols].copy()

    # 8) Tip güvenliği
    for col in ("day_of_week", "block_id"):
        if col in agg.columns:
            agg[col] = pd.to_numeric(agg[col], errors="coerce").fillna(-1).astype("int8")

   # 9) Priors (t0 üzerinden, sızıntısız; 3H/8H yok → block_id yok)
   if "day_of_week" not in agg.columns:
       agg["day_of_week"] = pd.to_datetime(agg["t0"], utc=True).dt.dayofweek.astype("int8")
   
   season_keys = ["day_of_week"]  # 1D/1W/1M
   
   agg = prior_rolling(agg, window="28D",  suffix="28d",  keys=season_keys)
   agg = prior_rolling(agg, window="180D", suffix="180d", keys=season_keys)

    # 10) Sırala + t0_local'ı düşür + yaz
    order_cols = ["GEOID", "t0"] + (["block_id"] if "block_id" in agg.columns else [])
    agg = agg.sort_values(order_cols).reset_index(drop=True)
    if "t0_local" in agg.columns:
        agg = agg.drop(columns=["t0_local"])  # çıktı sade

    dst.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(dst, index=False)

    # 11) Bilgi
    y_rate = float(agg["Y_label"].mean())
    print(f"[OK] Yazıldı: {dst}  satır={len(agg):,}  GEOID={agg['GEOID'].nunique():,}")
    print(f"[INFO] Y_label(1) % ≈ {100*y_rate:.4f} | 0 % ≈ {100*(1-y_rate):.4f}")
    print(f"[INFO] Zaman aralığı (t0 UTC): {agg['t0'].min()} → {agg['t0'].max()}")
    if "block_id" in agg.columns:
        print("[INFO] block_id dağılımı:")
        print(agg["block_id"].value_counts().sort_index())

if __name__ == "__main__":
    main()
