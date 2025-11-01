#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
risk_forecast.py  (REVIZE — 1D-öncelikli, sağlam horizon parse, priors adı uyumlu)
- Çoklu pencere (3H, 8H, 1D, 1W, 1M) için ileri tarih tahmini
- Eğitimden çıkan modelleri (models/sutam_{3h,8h,1d,1w,1m}.joblib) kullanır
- Girdi: geçmiş agregasyonlar (sf_crime_grid_{3h,8h,1d,1w,1m}.parquet)
- Çıktı: forecasts/forecast_<freq>.csv + .json

Kullanım:
  python risk_forecast.py --freq auto --horizon 72h --geoid 94107 --topk 50
  python risk_forecast.py --freq 1D   --horizon 30d  --topk 100
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Optional, Set

import numpy as np
import pandas as pd

try:
    import joblib
except Exception as e:
    raise SystemExit("joblib gerekli: pip install joblib") from e

# ---- Yollar
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
FREQ_TO_PANDAS = {"3H":"3h","8H":"8h","1D":"1d","1W":"7d","1M":None}

# ---- Yardımcılar
def now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))

# Toleranslı horizon parse
_HORIZON_RX = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([hdwmHDWM]?)\s*$")

def _parse_horizon_hours(horizon_str: str) -> float:
    """
    Ufku saat cinsinden döndürür.
    Kabul: '72h', '30D', '8w', '1m', '  24h ', '72', '72 H'
    Birim verilmezse saat varsayılır.
    """
    if not horizon_str:
        return 72.0
    s = (horizon_str or "").strip().replace("\u00A0", " ").lower()
    m = _HORIZON_RX.match(s)
    if m:
        val = float(m.group(1))
        unit = (m.group(2) or "h").lower()
        if unit == "h": return val
        if unit == "d": return val * 24.0
        if unit == "w": return val * 24.0 * 7.0
        if unit == "m": return val * 24.0 * 30.0  # yaklaşık ay
    # Pandas fallback
    try:
        td = pd.to_timedelta(s)
        return float(td / pd.Timedelta(hours=1))
    except Exception:
        try:
            return float(s)  # "72" → saat
        except Exception:
            raise ValueError("horizon biçimi: 24h, 72h, 14d, 8w, 1m ...")

def pick_freq_auto(h: str) -> str:
    """
    - ≤72h → 3H
    - ≤30d → 1D (1D-öncelikli)
    - ≤90d → 1W
    - >90d → 1M
    """
    hours = _parse_horizon_hours(h)
    if hours <= 72:              return "3H"
    if hours <= 24 * 30:         return "1D"
    if hours <= 24 * 90:         return "1W"
    return "1M"

def parse_horizon(h: str) -> timedelta:
    return timedelta(hours=_parse_horizon_hours(h))

def block_id_for_hour(h: int, freq: str) -> int:
    if freq == "8H": return int(h // 8)
    if freq == "3H": return int(h // 3)
    return -1

def build_future_index(freq: str, horizon: timedelta, start_utc: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    t0 = start_utc.tz_convert("UTC") if isinstance(start_utc, pd.Timestamp) else now_utc()
    if freq in ("3H","8H","1D","1W"):
        pd_freq = FREQ_TO_PANDAS[freq]
        # 1D ve 1W: gün başına hizala (runtime-anchored)
        if freq in ("1D","1W"):
            t0 = t0.floor("D")
        end = t0 + horizon
        rng = pd.date_range(t0.floor("s"), end.ceil("s"), freq=pd_freq, tz="UTC", inclusive="left")
        if len(rng) and rng[0] < t0:
            rng = rng[rng >= t0]
        return rng
    if freq == "1M":
        steps = max(1, int(np.ceil(horizon / timedelta(days=30))))
        vals = [t0 + i * timedelta(days=30) for i in range(steps)]
        return pd.DatetimeIndex(pd.to_datetime(vals, utc=True))
    raise ValueError(f"Bilinmeyen freq: {freq}")

def add_calendar(df: pd.DataFrame, t_col: str, freq: str) -> pd.DataFrame:
    s = pd.to_datetime(df[t_col], utc=True)
    df["year"]        = s.dt.year.astype("int16")
    df["month"]       = s.dt.month.astype("int8")
    df["day_of_week"] = s.dt.dayofweek.astype("int8")
    if freq in ("3H","8H","1D"):
        df["hour_start"] = (s.dt.hour if freq != "1D" else 0).astype("int8")
    if freq in ("3H","8H"):
        df["block_id"] = df["hour_start"].apply(lambda h: block_id_for_hour(int(h), freq)).astype("int8")
    return df

def last_known_snapshot(agg_path: Path, keys_keep: List[str]) -> pd.DataFrame:
    if not agg_path.exists():
        raise FileNotFoundError(f"Girdi yok: {agg_path}")
    df = pd.read_parquet(agg_path)
    need = {"GEOID","t0"} | set(keys_keep)
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"{agg_path.name} eksik kolon(lar): {miss}")
    df = df.sort_values(["GEOID","t0"]).groupby("GEOID", as_index=False).tail(1)
    return df[["GEOID", *keys_keep]].copy()

# ---- Model → beklenen giriş kolonları
def _collect_expected_columns_from_ct(ct) -> List[str]:
    cols: List[str] = []
    for _, trans, sel in getattr(ct, "transformers", []):
        if trans == "drop" or sel is None: continue
        if isinstance(sel, list):
            cols.extend([c for c in sel if isinstance(c, str)])
    seen=set(); ordered=[]
    for c in cols:
        if c not in seen: ordered.append(c); seen.add(c)
    return ordered

def expected_input_columns(model) -> Optional[List[str]]:
    try:
        from sklearn.compose import ColumnTransformer  # noqa: F401
    except Exception:
        return None
    pipe = model
    if hasattr(pipe, "named_steps"):
        pre = pipe.named_steps.get("pre")
        if pre is None:
            for step in pipe.named_steps.values():
                if step.__class__.__name__ == "ColumnTransformer":
                    pre = step; break
        if pre is not None and pre.__class__.__name__ == "ColumnTransformer":
            return _collect_expected_columns_from_ct(pre)
    cols = getattr(model, "feature_names_in_", None)
    return list(cols) if cols is not None else None

def align_X_to_model(model, X: pd.DataFrame) -> Tuple[pd.DataFrame, Set[str], Set[str]]:
    want = expected_input_columns(model)
    X = X.copy()
    if not want:
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce")
        return X.fillna(0.0), set(), set()

    have_set = set(X.columns); want_set = set(want)
    missing = want_set - have_set
    extra   = have_set - want_set

    for c in sorted(missing): X[c] = 0.0
    if extra: X = X.drop(columns=sorted(extra), errors="ignore")
    X = X.reindex(columns=want)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.fillna(0.0), missing, extra

# ---- Özellik hazırlama
def _present_columns(path: Path) -> Set[str]:
    try:
        import pyarrow.parquet as pq
        return set(pq.read_schema(path).names)
    except Exception:
        return set(pd.read_parquet(path, columns=None).columns)

def prepare_features(freq: str, horizon: timedelta, geoid: Optional[str]) -> tuple[pd.DataFrame, str]:
    agg_path = AGG_PATHS.get(freq)
    if not agg_path or not agg_path.exists():
        raise SystemExit(f"Agregasyon dosyası yok: {freq} → {agg_path}")

    # Priors adı uyumlu (28d/180d VEYA 3m/12m)
    priors_pairs = [
        ("prior_cnt_28d","prior_p_28d","prior_cnt_180d","prior_p_180d"),
        ("prior_cnt_3m","prior_p_3m","prior_cnt_12m","prior_p_12m"),
    ]
    present = _present_columns(agg_path)

    keep = ["crime_count"]
    for a,b,c,d in priors_pairs:
        if {a,b,c,d}.issubset(present):
            keep += [a,b,c,d]
            break
    # yan değişkenlerden mevcut olanları ekle (opsiyonel)
    maybes = [
        "wx_tavg","wx_prcp","wx_tmin","wx_tmax","wx_temp_range","wx_is_rainy","wx_is_hot_day",
        "poi_total_count","poi_count_300m","poi_count_600m","poi_count_900m",
        "poi_risk_score","poi_risk_300m","poi_risk_600m","poi_risk_900m",
        "bus_stop_count","train_stop_count","population",
        "neighbor_crime_24h","neighbor_crime_72h","neighbor_crime_7d",
        "daily_cnt","hr_cnt","311_request_count","911_geo_hr_last3d","911_geo_hr_last7d",
        "distance_to_police","distance_to_government_building",
    ]
    keep += [c for c in maybes if c in present]
    keep = list(dict.fromkeys(keep))  # uniq, sıralı

    snap = last_known_snapshot(agg_path, keys_keep=keep)

    # GEOID filtresi
    if geoid:
        geoid = str(geoid)
        snap = snap[snap["GEOID"].astype(str) == geoid]
        if snap.empty:
            raise SystemExit(f"GEOID son özetlerde yok: {geoid}")
        geoids = [geoid]
    else:
        geoids = snap["GEOID"].astype(str).unique().tolist()

    # Gelecek zaman dizisi (1D: gün başı)
    future_t0 = build_future_index(freq, horizon)
    if len(future_t0) == 0:
        raise SystemExit("Ufuk kısa: üretilecek ileri pencere bulunamadı.")
    grid = pd.MultiIndex.from_product([geoids, future_t0], names=["GEOID","t0"]).to_frame(index=False)

    # Takvim
    X = add_calendar(grid, "t0", freq)

    # Persist özellikleri GEOID ile bind et
    X = X.merge(snap, on="GEOID", how="left")

    # Eksik numerikleri median ile doldur
    for c in [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    # Takvim tipleri
    for c in ("day_of_week","block_id","hour_start","month","year"):
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            if c == "year":
                X[c] = X[c].fillna(-1).astype("int16")
            else:
                X[c] = X[c].fillna(-1).astype("int8")

    return X, agg_path.name

def load_model(freq: str):
    p = MODEL_PATHS.get(freq)
    if not p or not p.exists():
        raise SystemExit(f"Model yok: {p}")
    return joblib.load(p)

def score_and_rank(model, X: pd.DataFrame, topk: int | None, geoid: Optional[str]) -> pd.DataFrame:
    meta = X[["GEOID","t0"]].copy()
    feats = X.drop(columns=["GEOID","t0"], errors="ignore")

    feats, missing, extra = align_X_to_model(model, feats)
    if missing:
        m = sorted(list(missing))
        print(f"[WARN] Modelin beklediği {len(missing)} kolon X'te yoktu; 0 ile eklendi: {m[:10]}{' ...' if len(m)>10 else ''}")
    if extra:
        print(f"[INFO] X içinde modele gereksiz {len(extra)} kolon vardı; çıkarıldı.")

    proba = model.predict_proba(feats)[:, 1].astype(np.float32)

    out = meta.copy()
    out["prob"] = proba

    # 5 parçalı tier
    if len(out) >= 5:
        q = out["prob"].quantile([0.8,0.6,0.4,0.2]).to_list()
    else:
        q = [0.8,0.6,0.4,0.2]
    def tier(p):
        if p >= q[0]: return "Çok Yüksek"
        if p >= q[1]: return "Yüksek"
        if p >= q[2]: return "Orta"
        if p >= q[3]: return "Düşük"
        return "Çok Düşük"
    out["tier"] = out["prob"].apply(tier)

    if geoid:
        return out.sort_values("t0").reset_index(drop=True)
    out = out.sort_values(["t0","prob"], ascending=[True,False])
    if topk and topk > 0:
        out = out.groupby("t0", as_index=False).head(int(topk))
    return out.reset_index(drop=True)

def save_outputs(df: pd.DataFrame, freq: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    csv_p = outdir / f"forecast_{freq.lower()}.csv"
    json_p = outdir / f"forecast_{freq.lower()}.json"
    df_out = df.copy()
    df_out["t0"] = pd.to_datetime(df_out["t0"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df_out.to_csv(csv_p, index=False)
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(df_out.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    return csv_p, json_p

# ---- CLI
def parse_args():
    p = argparse.ArgumentParser(description="SUTAM ileri tahmin (1D-öncelikli)")
    p.add_argument("--freq", type=str, default="auto", help="auto | 3H | 8H | 1D | 1W | 1M")
    p.add_argument("--horizon", type=str, default="30d", help="Örn: 24h, 72h, 30d, 90d, 1m")
    p.add_argument("--geoid", type=str, default=None, help="Tek GEOID için tahmin")
    p.add_argument("--topk", type=int, default=50, help="Şehir geneli için her pencerede en yüksek K")
    p.add_argument("--outdir", type=Path, default=Path("forecasts"))
    return p.parse_args()

def main():
    args = parse_args()
    h_str = (args.horizon or "30d").strip()
    freq_in = (args.freq or "auto").upper().strip()
    if freq_in == "AUTO":
        freq = pick_freq_auto(h_str)
    else:
        freq = freq_in

    if freq not in MODEL_PATHS or freq not in AGG_PATHS:
        raise SystemExit(f"Desteklenmeyen freq: {freq} (izinli: {list(MODEL_PATHS.keys())})")

    horizon_td = parse_horizon(h_str)
    print(f"[INFO] freq={freq}  horizon={h_str}  geoid={args.geoid or 'ALL'}")

    X, src_name = prepare_features(freq, horizon_td, args.geoid)
    print(f"[OK] Özellikler hazır (kaynak={src_name}) — satır={len(X):,}, GEOID={X['GEOID'].nunique():,}")

    mdl = load_model(freq)
    print(f"[OK] Model yüklendi: {MODEL_PATHS[freq].name}")

    out = score_and_rank(mdl, X, args.topk, args.geoid)

    csv_p, json_p = save_outputs(out, freq, args.outdir)
    print(f"[DONE] Kaydedildi → {csv_p}  &  {json_p}")

    by_tier = out["tier"].value_counts().to_dict()
    print("[SUMMARY] tier dağılımı:", by_tier)
    print(out.head(min(10, len(out))).to_string(index=False))

if __name__ == "__main__":
    main()
