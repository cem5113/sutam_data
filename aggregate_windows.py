#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aggregate_windows.py  — runtime-anchored windowing
- Saatlik full-grid verisini (GEOID × dt) alır (dt: UTC-aware, saatlik)
- Pencereler: 3H, 8H, 1D, 1W(=7D), 1M(=30D) — tümü "çalıştırma anı"na göre hizalanır
- Y_label: pencere içinde >=1 olay varsa 1 (crime_count>0 toplamsal)
- Numerikler: crime_count SUM, diğer numeric kolonlar MEAN
- Priors (sızıntısız): 28D ve 180D rolling, anahtarlar:
    * 3H/8H: GEOID × day_of_week × block_id
    * 1D/1W/1M: GEOID × day_of_week
- Çıktı parquet (tüm zaman damgaları UTC). t0: pencere başlangıcı (UTC)
"""

import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

CAL_DROP = {"dt_local"}  # eski yardımcı kolon olursa temizleriz

# -------------------------
# Helpers (runtime anchor)
# -------------------------
def _now_tz(tz: Optional[str]) -> pd.Timestamp:
    return (pd.Timestamp.now(tz) if tz else pd.Timestamp.utcnow().tz_localize("UTC"))

def _anchored_floor(series: pd.Series, freq: str, tz: Optional[str]) -> pd.Series:
    """
    Pencereleri haftabaşı/pazartesi gibi sabitlere değil, "şimdi"nin gün başına göre hizalar.
    origin = now(tz).floor('D')
    """
    anchor = _now_tz(tz)
    origin = anchor.floor("D")
    return pd.to_datetime(series).dt.floor(freq, origin=origin)

def _anchored_day0(dt_local: pd.Series, tz: Optional[str]) -> pd.Series:
    return _anchored_floor(dt_local, "1D", tz)

def to_local(s_utc: pd.Series, tz: Optional[str]) -> pd.Series:
    s_utc = pd.to_datetime(s_utc, utc=True)
    return s_utc.dt.tz_convert(tz) if tz else s_utc

def make_t0_and_block(dt_local: pd.Series, freq: str, tz: Optional[str]) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    3H/8H: block_id üretir (gün içi blok: 3H→0..7, 8H→0..2), anchor = run-time day0
    1D: 1 günlük pencereler (anchor’lı)
    1W: sabit 7 günlük pencereler (anchor’lı)
    1M: sabit 30 günlük pencereler (anchor’lı)
    """
    if freq == "3H":
        t0_local = _anchored_floor(dt_local, "3H", tz)
        day0 = _anchored_day0(dt_local, tz)
        hrs_since = (dt_local.view("int64") - day0.view("int64")) / 3_600_000_000_000
        block_id = (np.floor(hrs_since / 3.0)).astype("int8")  # 0..7
    elif freq == "8H":
        t0_local = _anchored_floor(dt_local, "8H", tz)
        day0 = _anchored_day0(dt_local, tz)
        hrs_since = (dt_local.view("int64") - day0.view("int64")) / 3_600_000_000_000
        block_id = (np.floor(hrs_since / 8.0)).astype("int8")  # 0..2
    elif freq == "1D":
        t0_local = _anchored_floor(dt_local, "1D", tz)
        block_id = None
    elif freq == "1W":
        # 7 günlük sabit pencereler
        t0_local = _anchored_floor(dt_local, "7D", tz)
        block_id = None
    elif freq == "1M":
        # 30 günlük sabit pencereler (takvim ayı değil)
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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Saatlik full-grid (GEOID×dt) parquet")
    p.add_argument("--out",   type=Path, required=True, help="Çıktı parquet")
    p.add_argument("--freq",  type=str, choices=["3H","8H","1D","1W","1M"], required=True)
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
    need = {"dt", "GEOID"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Eksik kolon(lar): {miss}")

    # Eski yardımcı kolonu at
    drop_candidates = [c for c in CAL_DROP if c in df.columns]
    if drop_candidates:
        df = df.drop(columns=drop_candidates)

    # 2) Zaman hazırlığı (UTC→opsiyonel yerel)
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    dt_local = to_local(df["dt"], args.tz)
    t0_local, block_id = make_t0_and_block(dt_local, freq, args.tz)

    # 3) Toplama için hazırlık
    sdf = df.copy()
    sdf["t0_local"] = t0_local
    if block_id is not None:
        sdf["block_id"] = block_id

    # Numerikler: crime_count SUM, diğer numeric kolonlar MEAN
    base_exclude = {"GEOID", "dt", "Y_label"}
    num_cols = [c for c in sdf.columns
                if c not in base_exclude and pd.api.types.is_numeric_dtype(sdf[c])]

    agg_map = {}
    if "crime_count" in sdf.columns:
        agg_map["crime_count"] = "sum"
    other_nums = [c for c in num_cols if c != "crime_count"]
    agg_map.update({c: "mean" for c in other_nums})

    group_keys = ["GEOID", "t0_local"] + (["block_id"] if "block_id" in sdf.columns else [])
    agg = sdf.groupby(group_keys, as_index=False).agg(agg_map)

    # 4) Pencere etiketi (≥1 olay?)
    if "crime_count" not in agg.columns:
        # Teorik olarak olmamalı; yine de koruyalım
        raise SystemExit("crime_count kolonu bekleniyordu.")
    agg["Y_label"] = (agg["crime_count"] > 0).astype("int8")

    # 5) t0 (UTC) üret (t0_local tz-aware olduğundan sadece tz_convert)
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
    keep_cols += other_nums
    keep_cols = [c for c in keep_cols if c in agg.columns]
    agg = agg[keep_cols].copy()

    # 8) Güvenlik bloğu: kopya başlıkları at + anahtar tipleri sabitle
    agg = agg.loc[:, ~agg.columns.duplicated()]
    for col in ("day_of_week", "block_id"):
        if col in agg.columns:
            agg[col] = pd.to_numeric(agg[col], errors="coerce").fillna(-1).astype("int8")

    # 9) Priors
    if "block_id" in agg.columns:
        season_keys = ["day_of_week", "block_id"]  # 3H/8H
    else:
        season_keys = ["day_of_week"]              # 1D/1W/1M

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
