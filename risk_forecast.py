#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
risk_forecast.py
- Çoklu pencere (3H, 8H, 1D, 1W, 1M) için runtime-anchored gelecek tahmini
- Eğitimden çıkan modelleri (models/sutam_{3h,8h,1d,1w,1m}.joblib) kullanır.
- Özellikler: calendar + (persist) priors + son gözlemlerden yan değişkenler
- Girdi: geçmiş agregasyon dosyaları (sf_crime_grid_{3h,8h,1d,1w,1m}.parquet)
- Çıktı: CSV + JSON (varsayılan: forecasts/forecast_<freq>.{csv,json})

Kullanım:
  python risk_forecast.py --freq auto --horizon 72h --geoid 94107 --topk 50
  python risk_forecast.py --freq 1W  --horizon 90d --topk 100
  python risk_forecast.py --freq auto --horizon 1M
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

try:
    import joblib
except Exception as e:
    raise SystemExit("joblib gerekli: pip install joblib") from e


# ------------ Sabitler & Yol Haritası ------------
HERE = Path(__file__).resolve().parent

MODEL_PATHS = {
    "3H": HERE / "models" / "sutam_3h.joblib",
    "8H": HERE / "models" / "sutam_8h.joblib",
    "1D": HERE / "models" / "sutam_1d.joblib",
    "1W": HERE / "models" / "sutam_1w.joblib",
    "1M": HERE / "models" / "sutam_1m.joblib",
}

AGG_PATHS = {
    "3H": HERE / "sf_crime_grid_3h.parquet",
    "8H": HERE / "sf_crime_grid_8h.parquet",
    "1D": HERE / "sf_crime_grid_1d.parquet",
    "1W": HERE / "sf_crime_grid_1w.parquet",
    "1M": HERE / "sf_crime_grid_1m.parquet",
}

# pandas uyarısını önlemek için küçük harf (h/d) kullanalım
FREQ_TO_PANDAS = {
    "3H": "3h",
    "8H": "8h",
    "1D": "1d",
    "1W": "7d",   # runtime-anchored: 7 gün ileri
    "1M": None,   # 30g ileri özel
}


# ------------ Yardımcılar ------------
def now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def pick_freq_auto(horizon_str: str) -> str:
    """
    AUTO seçim — SADECE mevcut frekanslardan (3H, 8H, 1D, 1W, 1M) döner.
    - ≤72h → 3H
    - ≤30d → 1D
    - ≤90d → 1W
    - >90d → 1M (yaklaşık 30g adımlarla)
    """
    s = horizon_str.lower().strip()
    if s.endswith("h"):
        hours = float(s[:-1])
    elif s.endswith("d"):
        hours = float(s[:-1]) * 24
    elif s.endswith("w"):
        hours = float(s[:-1]) * 24 * 7
    elif s.endswith("m"):
        hours = float(s[:-1]) * 24 * 30
    else:
        raise ValueError("horizon biçimi: 24h, 72h, 14d, 8w, 1m ...")

    if hours <= 72:
        return "3H"
    if hours <= 24 * 30:
        return "1D"
    if hours <= 24 * 90:
        return "1W"
    return "1M"


def parse_horizon(h: str) -> timedelta:
    s = h.lower().strip()
    if s.endswith("h"):
        return timedelta(hours=float(s[:-1]))
    if s.endswith("d"):
        return timedelta(days=float(s[:-1]))
    if s.endswith("w"):
        return timedelta(weeks=float(s[:-1]))
    if s.endswith("m"):
        # 1M ≈ 30 gün
        return timedelta(days=float(s[:-1]) * 30)
    raise ValueError("horizon biçimi: 24h, 10d, 8w, 1m ...")


def block_id_for_hour(h: int, freq: str) -> int:
    """8H için 0,1,2 (00-08,08-16,16-24); 3H için 0..7; diğerleri -1."""
    if freq == "8H":
        return int(h // 8)
    if freq == "3H":
        return int(h // 3)
    return -1


def build_future_index(freq: str, horizon: timedelta, start_utc: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    t0 = start_utc.tz_convert("UTC") if isinstance(start_utc, pd.Timestamp) else now_utc()
    if freq in ("3H", "8H", "1D", "1W"):
        pd_freq = FREQ_TO_PANDAS[freq]
        end = t0 + horizon
        # inclusive='left' → t0 sonrası slotları üret
        rng = pd.date_range(t0.floor("s"), end.ceil("s"), freq=pd_freq, tz="UTC", inclusive="left")
        if len(rng) and rng[0] < t0:
            rng = rng[rng >= t0]
        return rng
    elif freq == "1M":
        # her adım ≈ 30 gün ileri
        steps = max(1, int(np.ceil(horizon / timedelta(days=30))))
        vals = [t0 + i * timedelta(days=30) for i in range(steps)]
        return pd.DatetimeIndex(pd.to_datetime(vals, utc=True))
    else:
        raise ValueError(f"Bilinmeyen freq: {freq}")


def add_calendar(df: pd.DataFrame, t_col: str, freq: str) -> pd.DataFrame:
    s = pd.to_datetime(df[t_col], utc=True)
    df["year"] = s.dt.year.astype("int16")
    df["month"] = s.dt.month.astype("int8")
    df["day_of_week"] = s.dt.dayofweek.astype("int8")
    if freq in ("3H", "8H", "1D"):
        df["hour_start"] = s.dt.hour.astype("int8")
    if freq in ("3H", "8H"):
        df["block_id"] = df["hour_start"].apply(lambda h: block_id_for_hour(int(h), freq)).astype("int8")
    return df


def last_known_snapshot(agg_path: Path, keys_keep: list[str]) -> pd.DataFrame:
    """Agregasyondan, her GEOID için son satırı al (persist özellikler için)."""
    if not agg_path.exists():
        raise FileNotFoundError(f"Girdi yok: {agg_path}")
    df = pd.read_parquet(agg_path)
    need = {"GEOID", "t0"} | set(keys_keep)
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"{agg_path.name} eksik kolon(lar): {miss}")
    df = df.sort_values(["GEOID", "t0"]).groupby("GEOID", as_index=False).tail(1)
    return df[["GEOID"] + keys_keep].copy()


def prepare_features(freq: str, horizon: timedelta, geoid: str | None) -> tuple[pd.DataFrame, str]:
    """Gelecek pencereler için X oluştur. Priors/yan değişkenleri son bilinen değerlerle doldurur."""
    agg_path = AGG_PATHS.get(freq)
    if agg_path is None or not agg_path.exists():
        raise SystemExit(f"Geçmiş agregasyon dosyası bulunamadı: {freq} → {agg_path}")

    # Son bilinen özetler (persist edilecek kolonlar)
    base_numeric_maybe = [
        "crime_count",
        "prior_cnt_28d", "prior_p_28d",
        "prior_cnt_180d", "prior_p_180d",
        # yan değişken örnekleri (varsa persist):
        "wx_tavg", "wx_prcp", "poi_total_count", "bus_stop_count", "train_stop_count", "population",
    ]

    # 🔧 HIZLI ŞEMA OKUMA: read_parquet(..., columns=[]).columns  (nrows desteklenmez!)
    try:
        present_cols = set(pd.read_parquet(agg_path, columns=[]).columns)
    except Exception:
        # bazı okumalarda columns=[] boş dönebilir; tam okuyup sadece kolon alalım
        present_cols = set(pd.read_parquet(agg_path).columns)

    keep_cols = [c for c in base_numeric_maybe if c in present_cols]
    if not keep_cols:
        # Hiçbiri yoksa, en azından priors ve crime_count'ı bekleyelim; yoksa sadece calendar ile ilerleriz
        keep_cols = [c for c in ["crime_count", "prior_cnt_28d", "prior_p_28d", "prior_cnt_180d", "prior_p_180d"] if c in present_cols]

    snap = last_known_snapshot(agg_path, keys_keep=keep_cols)

    # Hedef GEOID listesi
    if geoid:
        geoids = [str(geoid)]
        snap = snap[snap["GEOID"].isin(geoids)]
        if snap.empty:
            raise SystemExit(f"GEOID son özetlerde yok: {geoid}")
    else:
        geoids = snap["GEOID"].astype(str).unique().tolist()

    # Gelecek zaman dizisi
    future_t0 = build_future_index(freq, horizon)
    if len(future_t0) == 0:
        raise SystemExit("Horizon çok kısa görünüyor; üretilecek ileri pencere yok.")

    grid = pd.MultiIndex.from_product([geoids, future_t0], names=["GEOID", "t0"]).to_frame(index=False)

    # Takvim
    grid = add_calendar(grid, "t0", freq)

    # Persist özellikler (GEOID bazlı sabitler)
    X = grid.merge(snap, on="GEOID", how="left")

    # Eksik numerikleri median ile doldur
    for c in [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    # Model tarafının beklediği tipler
    for c in ("day_of_week", "block_id"):
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="ignore")
            if not pd.api.types.is_integer_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(-1).astype("int8")

    return X, agg_path.name


def load_model(freq: str):
    p = MODEL_PATHS.get(freq)
    if p is None:
        raise SystemExit(f"Model yolu haritasında olmayan frekans: {freq}")
    if not p.exists():
        raise SystemExit(f"Model yok: {p}")
    return joblib.load(p)


def score_and_rank(model, X: pd.DataFrame, topk: int | None, geoid: str | None) -> pd.DataFrame:
    # Modelin pipeline'ı (OneHot/Scaler vs.) olduğunu varsayıyoruz; GEOID/t0 çıkar
    feat = X.drop(columns=["GEOID", "t0"], errors="ignore")
    proba = model.predict_proba(feat)[:, 1].astype(np.float32)

    out = X[["GEOID", "t0"]].copy()
    out["prob"] = proba

    # Basit beşli tier
    if len(out) >= 5:
        q = out["prob"].quantile([0.8, 0.6, 0.4, 0.2]).to_list()
    else:
        q = [0.8, 0.6, 0.4, 0.2]

    def tier(p):
        if p >= q[0]: return "Çok Yüksek"
        if p >= q[1]: return "Yüksek"
        if p >= q[2]: return "Orta"
        if p >= q[3]: return "Düşük"
        return "Çok Düşük"

    out["tier"] = out["prob"].apply(tier)

    if geoid:
        # tek geoid için zaman sıralı
        out = out.sort_values("t0", ascending=True)
    else:
        # şehir geneli top-k (aynı t0’lar üzerinde)
        out = out.sort_values(["t0", "prob"], ascending=[True, False])
        if topk and topk > 0:
            out = out.groupby("t0", as_index=False).head(int(topk))

    return out.reset_index(drop=True)


def save_outputs(df: pd.DataFrame, freq: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    csv_p = outdir / f"forecast_{freq.lower()}.csv"
    json_p = outdir / f"forecast_{freq.lower()}.json"
    df_out = df.copy()
    df_out["t0_iso"] = pd.to_datetime(df_out["t0"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%SZ")
    df_out.to_csv(csv_p, index=False)
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(df_out.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    return csv_p, json_p


# ------------ CLI ------------
def parse_args():
    p = argparse.ArgumentParser(description="SUTAM çoklu pencere ileri tahmin")
    p.add_argument("--freq", type=str, default="auto",
                   help="auto | 3H | 8H | 1D | 1W | 1M")
    p.add_argument("--horizon", type=str, default="72h",
                   help="Örn: 24h, 72h, 14d, 90d, 1m")
    p.add_argument("--geoid", type=str, default=None,
                   help="Tek GEOID için tahmin (boşsa şehir geneli)")
    p.add_argument("--topk", type=int, default=50,
                   help="Şehir geneli için her pencerede en yüksek K")
    p.add_argument("--outdir", type=Path, default=Path("forecasts"))
    return p.parse_args()


def main():
    args = parse_args()
    freq = args.freq.upper().strip()
    if freq == "AUTO":
        freq = pick_freq_auto(args.horizon)

    if freq not in MODEL_PATHS or freq not in AGG_PATHS:
        raise SystemExit(f"Desteklenmeyen freq: {freq} (izinli: {list(MODEL_PATHS.keys())})")

    horizon_td = parse_horizon(args.horizon)

    print(f"[INFO] freq={freq}  horizon={args.horizon}  geoid={args.geoid or 'ALL'}")

    # Özellikleri hazırla
    X, src_name = prepare_features(freq, horizon_td, args.geoid)
    print(f"[OK] Özellikler hazır (kaynak={src_name}) — satır={len(X):,}, GEOID={X['GEOID'].nunique():,}")

    # Modeli yükle
    mdl = load_model(freq)
    print(f"[OK] Model yüklendi: {MODEL_PATHS[freq].name}")

    # Skorla ve sırala
    out = score_and_rank(mdl, X, args.topk, args.geoid)

    # Kaydet
    csv_p, json_p = save_outputs(out, freq, args.outdir)
    print(f"[DONE] Kaydedildi → {csv_p}  &  {json_p}")

    # Özet
    by_tier = out["tier"].value_counts().to_dict()
    print("[SUMMARY] tier dağılımı:", by_tier)
    print(out.head(min(10, len(out))).to_string(index=False))


if __name__ == "__main__":
    main()
