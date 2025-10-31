#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aggregate_windows.py
- Saatlik full-grid verisini (GEOID × dt) alır
- Günlük (1D) veya 8 saatlik (8H) pencerelere toplar (yerel zamanla hizalı)
- Y_label: pencere içinde >=1 olay varsa 1
- Leakage-safe rolling priors (28g / 180g): GEOID × {day_of_week, block_id?}
- Çıktıyı parquet yazar (tüm zaman damgaları UTC)

Örnek:
  python aggregate_windows.py \
    --input sf_crime_grid_full_labeled.parquet \
    --out   sf_crime_grid_8h.parquet \
    --freq  8H \
    --tz    America/Los_Angeles
"""

import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

CAL_DROP = {"dt_local"}  # saatlikten kalmış olabilecek yardımcı kolon

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="Saatlik full-grid parquet (sf_crime_grid_full_labeled.parquet)")
    p.add_argument("--out", type=Path, required=True,
                   help="Çıktı parquet (örn. sf_crime_grid_8h.parquet / sf_crime_grid_1d.parquet)")
    p.add_argument("--freq", type=str, choices=["1D", "8H"], default="1D",
                   help="Pencere frekansı: 1D (günlük) veya 8H (8 saatlik bloklar)")
    p.add_argument("--tz", type=str, default=None,
                   help="Yerel TZ (örn: America/Los_Angeles). Verilirse pencereler yerel saate göre oluşur.")
    return p.parse_args()

def to_local(s_utc: pd.Series, tz: Optional[str]) -> pd.Series:
    """UTC-zoned datetime serisini, tz verilmişse yerel saat dilimine çevirir."""
    s_utc = pd.to_datetime(s_utc, utc=True)
    if tz:
        return s_utc.dt.tz_convert(tz)
    return s_utc  # UTC kalsın

def make_t0_and_block(dt_local: pd.Series, freq: str) -> Tuple[pd.Series, Optional[pd.Series]]:
    """Yerel saat serisinden (tz-aware) 1D veya 8H pencere başlangıcı (t0_local) ve block_id üret."""
    if freq == "1D":
        t0_local = dt_local.dt.floor("D")
        block_id = None
    else:
        # 0–7, 8–15, 16–23
        block_id = (dt_local.dt.hour // 8).astype("int8")
        t0_local = dt_local.dt.floor("D") + pd.to_timedelta(block_id * 8, unit="h")
    return t0_local, block_id

def add_calendar_cols(df: pd.DataFrame, t0_local_col: str) -> pd.DataFrame:
    """Takvim kolonları (yerel t0 üzerinden)."""
    df["year"]        = df[t0_local_col].dt.year.astype("int16")
    df["month"]       = df[t0_local_col].dt.month.astype("int8")
    df["day_of_week"] = df[t0_local_col].dt.dayofweek.astype("int8")
    df["hour_start"]  = df[t0_local_col].dt.hour.astype("int8")
    return df

def prior_rolling(df: pd.DataFrame, window: str, suffix: str, keys: List[str]) -> pd.DataFrame:
    """
    df: GEOID, t0(UTC), Y_label ... içerir; t0 UTC'dir fakat mevsimsellik keys yerel tabanlı kolonlardır
    keys: ["day_of_week"] (+ opsiyonel "block_id")
    """
    df = df.sort_values(["GEOID", "t0"]).copy()

    grp_cols = ["GEOID"] + [k for k in keys if k in df.columns]

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("t0").set_index("t0")
        # Rolling window zaman-ofsetli; shift(1) ile sızıntıyı kapat
        cnt = g["Y_label"].rolling(window=window).sum().shift(1).fillna(0.0)
        out = g.copy()
        out[f"prior_cnt_{suffix}"] = cnt.to_numpy().astype("float32")
        return out.reset_index()

    out = df.groupby(grp_cols, group_keys=False).apply(_roll)

    hours_in_window = float(pd.Timedelta(window) / pd.Timedelta("1h"))
    out[f"prior_p_{suffix}"] = (out[f"prior_cnt_{suffix}"] / hours_in_window).astype("float32")
    return out

def main():
    args = parse_args()
    src: Path = args.input
    dst: Path = args.out
    freq = args.freq

    if not src.exists():
        raise SystemExit(f"Girdi yok: {src.resolve()}")

    df = pd.read_parquet(src)

    # Gerekli kolonlar
    need = {"dt", "GEOID"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Eksik kolon(lar): {miss}")

    # Eski yardımcı kolonları temizle
    drop_candidates = [c for c in CAL_DROP if c in df.columns]
    if drop_candidates:
        df = df.drop(columns=drop_candidates)

    # Zaman hazırlığı
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    dt_local = to_local(df["dt"], args.tz)
    t0_local, block_id = make_t0_and_block(dt_local, freq)

    # Toplama anahtarları ve veri
    sdf = df.copy()
    sdf["t0_local"] = t0_local
    if block_id is not None:
        sdf["block_id"] = block_id

    # Numerik kolonlar: crime_count SUM, diğer tüm numerikler MEAN
    base_exclude = {"GEOID", "dt", "Y_label"}  # Y_label pencere sonunda yeniden kurulacak
    num_cols = [c for c in sdf.columns if c not in base_exclude and pd.api.types.is_numeric_dtype(sdf[c])]

    agg_map = {}
    if "crime_count" in sdf.columns:
        agg_map["crime_count"] = "sum"
    other_nums = [c for c in num_cols if c != "crime_count"]
    agg_map |= {c: "mean" for c in other_nums}

    group_keys = ["GEOID", "t0_local"] + (["block_id"] if "block_id" in sdf.columns else [])

    agg = sdf.groupby(group_keys, as_index=False).agg(agg_map)

    # Y_label: pencerede >=1 olay?
    if "crime_count" not in agg.columns:
        raise SystemExit("crime_count kolonu bekleniyordu (pencere Y_label'ı için).")
    agg["Y_label"] = (agg["crime_count"] > 0).astype("int8")

    # t0 (UTC) ve takvim kolonları
    agg["t0"] = pd.to_datetime(agg["t0_local"]).dt.tz_localize(None).dt.tz_localize(args.tz or "UTC").dt.tz_convert("UTC")
    agg = add_calendar_cols(agg, "t0_local")

    # Priors (28g ve 180g), mevsimsellik anahtarları
    season_keys = ["day_of_week"]
    if "block_id" in agg.columns:
        season_keys.append("block_id")

    # Sadece gerekli kolonlar + priors
    keep_cols = ["GEOID", "t0", "Y_label", "crime_count", "year", "month", "day_of_week"]
    if "block_id" in agg.columns:
        keep_cols.append("block_id")
    # Ortalama alınan numerikleri de koru
    keep_cols += other_nums
    keep_cols = [c for c in keep_cols if c in agg.columns]

    agg = agg[keep_cols].copy()

    # Priors ekle
    agg = prior_rolling(agg, window="28D",  suffix="28d",  keys=season_keys)
    agg = prior_rolling(agg, window="180D", suffix="180d", keys=season_keys)

    # Sıralama + yazma
    order_cols = ["GEOID", "t0"] + (["block_id"] if "block_id" in agg.columns else [])
    agg = agg.sort_values(order_cols).reset_index(drop=True)

    dst.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(dst, index=False)

    # Bilgi
    y_rate = float(agg["Y_label"].mean())
    print(f"[OK] Yazıldı: {dst}  satır={len(agg):,}  GEOID={agg['GEOID'].nunique():,}")
    print(f"[INFO] Y_label(1) % ≈ {100*y_rate:.4f} | 0 % ≈ {100*(1-y_rate):.4f}")
    print(f"[INFO] Zaman aralığı (t0 UTC): {agg['t0'].min()} → {agg['t0'].max()}")
    if "block_id" in agg.columns:
        print("[INFO] 8H blok dağılımı (0=00–08, 1=08–16, 2=16–24 yerel):")
        print(agg["block_id"].value_counts().sort_index())

if __name__ == "__main__":
    main()
