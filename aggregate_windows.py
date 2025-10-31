#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aggregate_windows.py (RUN-TIME ANCHORED)
- Saatlik full-grid verisini (GEOID × dt[UTC]) alır
- 3H / 8H / 1D / 1W(=7D) / 1M(=30D sabit) pencerelere anchor'lı (run-time) toplar
- Y_label: pencere içinde >=1 olay varsa 1
- Leakage-safe rolling priors (28g / 180g): GEOID × {day_of_week, block_id?}
- Çıktıyı parquet yazar (tüm zaman damgaları UTC)
"""

import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

CAL_DROP = {"dt_local"}  # eski yardımcı kolonlar varsa düşür

# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Saatlik full-grid parquet (GEOID, dt, ...)")
    p.add_argument("--out", type=Path, required=True, help="Çıktı parquet")
    p.add_argument("--freq", type=str, choices=["3H","8H","1D","1W","1M"], required=True,
                   help="1W=7 gün; 1M=30 gün (sabit). Hepsi run-time anchored.")
    p.add_argument("--tz", type=str, default=None, help="Yerel timezone (örn. America/Los_Angeles)")
    return p.parse_args()

# -------------------- Zaman yardımcıları --------------------

def to_local(s_utc: pd.Series, tz: Optional[str]) -> pd.Series:
    s_utc = pd.to_datetime(s_utc, utc=True)
    return s_utc.dt.tz_convert(tz) if tz else s_utc

def _now_tz(tz: Optional[str]) -> pd.Timestamp:
    # Çalıştırma anı (anchor) — yerel tz verilmişse yerel, yoksa UTC
    return (pd.Timestamp.now(tz) if tz else pd.Timestamp.utcnow().tz_localize("UTC"))

def _anchored_floor(series: pd.Series, freq: str, tz: Optional[str]) -> pd.Series:
    """
    series: tz-aware (yerel veya UTC) datetime Series
    freq  : '3H','8H','1D','7D','30D' gibi sabit frekans
    origin: run-time anchor'ın gün başına indirilmiş hali
    """
    anchor = _now_tz(tz)
    origin = anchor.floor("D")
    return series.dt.floor(freq, origin=origin)

def _anchored_day0(dt_local: pd.Series, tz: Optional[str]) -> pd.Series:
    """Anchor'lı gün başlangıcı: dt_local'i, run-time gün başına göre floor et."""
    return _anchored_floor(dt_local, "1D", tz)

def make_t0_and_block(dt_local: pd.Series, freq: str, tz: Optional[str]) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    t0_local başlangıcını üretir ve gerekirse block_id döndürür (anchor'lı).
    - 3H, 8H: block_id (0..7 / 0..2) anchor'lı gün başlangıcına göre hesaplanır.
    - 1D: günlük anchor'lı; block_id yok.
    - 1W: 7D anchor'lı; block_id yok.
    - 1M: 30D anchor'lı (takvim ayı değil); block_id yok.
    """
    if freq == "3H":
        t0_local = _anchored_floor(dt_local, "3H", tz)
        # Gün başlangıcını anchor'lı al, kaçıncı 3 saatlik blok?
        day0 = _anchored_day0(dt_local, tz)
        hrs_since = (dt_local.view("int64") - day0.view("int64")) / 3_600_000_000_000  # nanos → saat
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
        t0_local = _anchored_floor(dt_local, "7D", tz)  # 1W yerine 7D (sabit) kullandık
        block_id = None
    elif freq == "1M":
        t0_local = _anchored_floor(dt_local, "30D", tz)  # takvim ayı yerine 30 günlük sabit pencere
        block_id = None
    else:
        raise ValueError(f"Bilinmeyen freq: {freq}")
    return t0_local, block_id

# -------------------- Calendar & Priors --------------------

def add_calendar_cols(df: pd.DataFrame, t0_local_col: str) -> pd.DataFrame:
    df["year"]        = df[t0_local_col].dt.year.astype("int16")
    df["month"]       = df[t0_local_col].dt.month.astype("int8")
    df["day_of_week"] = df[t0_local_col].dt.dayofweek.astype("int8")
    df["hour_start"]  = df[t0_local_col].dt.hour.astype("int8")
    # ISO hafta bilgisi (anchor'lı olduğundan klasik haftayla birebir değil; raporlama için yine de faydalı)
    try:
        df["week"] = df[t0_local_col].dt.isocalendar().week.astype("UInt32")
    except Exception:
        pass
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

# -------------------- Ana akış --------------------

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

    # 3) Anchor'lı t0 ve block
    t0_local, block_id = make_t0_and_block(dt_local, freq, args.tz)

    # 4) Toplama için hazırlık
    sdf = df.copy()
    sdf["t0_local"] = t0_local
    if block_id is not None:
        sdf["block_id"] = block_id

    # Numerikler: crime_count SUM, diğerleri MEAN
    base_exclude = {"GEOID", "dt", "Y_label"}
    num_cols = [c for c in sdf.columns if c not in base_exclude and pd.api.types.is_numeric_dtype(sdf[c])]

    agg_map = {}
    if "crime_count" in sdf.columns:
        agg_map["crime_count"] = "sum"
    other_nums = [c for c in num_cols if c != "crime_count"]
    agg_map.update({c: "mean" for c in other_nums})

    group_keys = ["GEOID", "t0_local"] + (["block_id"] if "block_id" in sdf.columns else [])
    agg = sdf.groupby(group_keys, as_index=False).agg(agg_map)

    # 5) Pencere etiketi (≥1 olay?)
    if "crime_count" not in agg.columns:
        raise SystemExit("crime_count kolonu bekleniyordu.")
    agg["Y_label"] = (agg["crime_count"] > 0).astype("int8")

    # 6) t0 (UTC) üret (t0_local tz-aware olduğundan sadece tz_convert)
    t0_loc = pd.to_datetime(agg["t0_local"], errors="coerce")
    if getattr(t0_loc.dt, "tz", None) is None:
        # nadiren localize edilmemiş olabilir: args.tz ya da UTC ile localize et
        agg["t0"] = t0_loc.dt.tz_localize(args.tz or "UTC").dt.tz_convert("UTC")
    else:
        agg["t0"] = t0_loc.dt.tz_convert("UTC")

    # 7) Takvim kolonları (yerel t0 üzerinden)
    agg = add_calendar_cols(agg, "t0_local")

    # 8) Çıktıda tutulacaklar
    keep_cols = ["GEOID", "t0", "t0_local", "Y_label", "crime_count",
                 "year", "month", "day_of_week", "hour_start"]
    if "block_id" in agg.columns:
        keep_cols.append("block_id")
    keep_cols += other_nums
    keep_cols = [c for c in keep_cols if c in agg.columns]
    agg = agg.loc[:, ~pd.Index(agg.columns).duplicated()].copy()
    agg = agg[keep_cols].copy()

    # 9) Priors (mevsimsellik anahtarları: DOW (+block_id))
    season_keys = ["day_of_week"] + (["block_id"] if "block_id" in agg.columns else [])
    agg = prior_rolling(agg, window="28D",  suffix="28d",  keys=season_keys)
    agg = prior_rolling(agg, window="180D", suffix="180d", keys=season_keys)

    # 10) Sırala + t0_local'ı düşür + yaz
    order_cols = ["GEOID", "t0"] + (["block_id"] if "block_id" in agg.columns else [])
    agg = agg.sort_values(order_cols).reset_index(drop=True)
    if "t0_local" in agg.columns:
        agg = agg.drop(columns=["t0_local"])

    dst.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(dst, index=False)

    # 11) Bilgi
    y_rate = float(agg["Y_label"].mean())
    print(f"[OK] Yazıldı: {dst}  satır={len(agg):,}  GEOID={agg['GEOID'].nunique():,}")
    print(f"[INFO] Y_label(1) % ≈ {100*y_rate:.4f} | 0 % ≈ {100*(1-y_rate):.4f}")
    print(f"[INFO] Zaman aralığı (t0 UTC): {agg['t0'].min()} → {agg['t0'].max()}")
    if "block_id" in agg.columns:
        vc = agg["block_id"].value_counts().sort_index()
        print(f"[INFO] block_id dağılımı:\n{vc}")

if __name__ == "__main__":
    main()
