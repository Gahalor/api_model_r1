# utils.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict, Any, Tuple, List, Optional

# -----------------------------
# Helpers
# -----------------------------
def safe_get(data: Any, *keys, default=None):
    cur = data
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def _as_clean_float_array(a: Any) -> np.ndarray:
    """Convierte a np.array(float) y reemplaza NaN/Inf por 0.0 (o ajusta según tu preferencia)."""
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 0:
        arr = np.array([arr], dtype=float)
    # reemplazo seguro
    bad = ~np.isfinite(arr)
    if bad.any():
        arr[bad] = 0.0
    return arr

def _ensure_min_len_for_filtfilt(arr: np.ndarray, padlen: int = 9) -> np.ndarray:
    """filtfilt requiere len(x) > padlen. Si no, devuelve sin filtrar."""
    return arr if arr.size > padlen else arr.copy()

# -----------------------------
# 1) LECTURA DEL PAYLOAD
# -----------------------------
def read_json_data(payload: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Acepta dos variantes:
      A) { "seismoelectric": { "sampleRate": 3333, "data": { "v1": [], "v2": [], "deep": [] } } }
      B) { "samples": { "seismoelectric": { "sampleRate": 3333, "data": {...} } } }
    Devuelve:
      df con columnas ['BLUE','RED','Depth']  (alias de v1, v2, deep)
      meta = { "sampling": fs }
    """
    se = (
        safe_get(payload, "seismoelectric", "data")
        or safe_get(payload, "samples", "seismoelectric", "data")
        or {}
    )
    fs = (
        safe_get(payload, "seismoelectric", "sampleRate")
        or safe_get(payload, "samples", "seismoelectric", "sampleRate")
        or 3333
    )

    v1 = _as_clean_float_array(se.get("v1", []))
    v2 = _as_clean_float_array(se.get("v2", []))
    depth = _as_clean_float_array(se.get("deep", []))

    # Normaliza largo por si vienen desfasados
    n = int(min(v1.size, v2.size, depth.size)) if depth.size else int(min(v1.size, v2.size))
    v1, v2 = v1[:n], v2[:n]
    depth = depth[:n] if depth.size else np.arange(n, dtype=float)

    df = pd.DataFrame({
        "BLUE": v1,          # mantengo nombres históricos de tus otras APIs
        "RED": v2,
        "Depth": depth
    })

    meta = {"sampling": int(fs)}
    return df, meta

# -----------------------------
# 2) CONFIG DE FILTROS
# -----------------------------
def get_filter_configuration(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Espera algo como:
    {
      "filters": {
        "type": "lowpass" | "highpass" | "bandpass" | "notch" | "none",
        "cutoff": 60,
        "order": 2,
        "band": [5, 120],        # si bandpass
        "notch": { "freq": 50, "Q": 30 },
        "mode": "global" | "by_ranges",
        "ranges": [
          { "start": 100, "end": 500, "type": "lowpass", "cutoff": 60, "order": 2 }
        ]
      }
    }
    """
    filters = safe_get(payload, "filters") or {}
    cfg = {
        "type": filters.get("type", "none"),
        "cutoff": filters.get("cutoff", 60),
        "order": int(filters.get("order", 2)),
        "band": filters.get("band", None),
        "notch": filters.get("notch", {"freq": 50, "Q": 30}),
        "mode": filters.get("mode", "global"),
        "ranges": filters.get("ranges", []),
    }
    return cfg

# -----------------------------
# 3) APLICACIÓN DE FILTROS
# -----------------------------
def _apply_butterworth(x: np.ndarray, fs: float, ftype: str, order: int = 2,
                       cutoff: Optional[float] = None, band: Optional[List[float]] = None) -> np.ndarray:
    if x.size == 0:
        return x
    nyq = fs * 0.5
    x = _ensure_min_len_for_filtfilt(x)

    if ftype == "lowpass" and cutoff:
        b, a = signal.butter(order, cutoff / nyq, btype="low", analog=False)
    elif ftype == "highpass" and cutoff:
        b, a = signal.butter(order, cutoff / nyq, btype="high", analog=False)
    elif ftype == "bandpass" and band and len(band) == 2:
        low = min(band) / nyq
        high = max(band) / nyq
        b, a = signal.butter(order, [low, high], btype="band", analog=False)
    else:
        return x  # tipo no válido -> sin cambios

    try:
        return signal.filtfilt(b, a, x, method="pad")
    except ValueError:
        # Si falla por longitud, devolvemos sin filtrar
        return x

def _apply_notch(x: np.ndarray, fs: float, f0: float = 50.0, Q: float = 30.0) -> np.ndarray:
    if x.size == 0:
        return x
    x = _ensure_min_len_for_filtfilt(x)
    b, a = signal.iirnotch(f0, Q, fs)
    try:
        return signal.filtfilt(b, a, x, method="pad")
    except ValueError:
        return x

def _apply_cfg_to_series(x: np.ndarray, fs: float, cfg: Dict[str, Any]) -> np.ndarray:
    t = cfg.get("type", "none")
    if t in ("lowpass", "highpass", "bandpass"):
        return _apply_butterworth(
            x, fs, ftype=t,
            order=int(cfg.get("order", 2)),
            cutoff=cfg.get("cutoff"),
            band=cfg.get("band"),
        )
    elif t == "notch":
        notch = cfg.get("notch", {}) or {}
        return _apply_notch(x, fs, float(notch.get("freq", 50.0)), float(notch.get("Q", 30.0)))
    return x

def aplicar_filtros(df: pd.DataFrame, cfg: Dict[str, Any], fs: float) -> pd.DataFrame:
    """
    Aplica filtros (globales o por rangos) sobre BLUE y RED. Devuelve un nuevo DataFrame.
    - mode: "global" o "by_ranges"
    - by_ranges: lista de dicts con {start, end, ...misma config de filtro}
      Los índices son sobre las muestras (0-based).
    """
    out = df.copy()

    if cfg.get("mode", "global") == "by_ranges":
        ranges: List[Dict[str, Any]] = cfg.get("ranges", [])
        for col in ("BLUE", "RED"):
            arr = out[col].to_numpy().copy()
            for r in ranges:
                s = max(int(r.get("start", 0)), 0)
                e = min(int(r.get("end", arr.size)), arr.size)
                if s >= e:
                    continue
                seg = arr[s:e]
                arr[s:e] = _apply_cfg_to_series(seg, fs, r)
            out[col] = arr
        return out

    # Global
    for col in ("BLUE", "RED"):
        out[col] = _apply_cfg_to_series(out[col].to_numpy(), fs, cfg)

    return out

# -----------------------------
# 4) FEATURES PARA EL MOJO
# -----------------------------
def build_features_dataframe(df: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Devuelve el DataFrame de features que consume el MOJO.
    Si tu modelo espera columnas específicas, colócalas aquí.
    Base mínima: BLUE, RED, Depth + derivados simples.
    """
    out = pd.DataFrame(index=df.index)

    # columnas base (históricamente usadas en tus APIs)
    out["BLUE"] = df["BLUE"].astype(float)
    out["RED"] = df["RED"].astype(float)
    out["Depth"] = df["Depth"].astype(float)

    # Derivadas simples (puedes quitarlas si tu MOJO no las necesita)
    out["BLUE_abs"] = np.abs(out["BLUE"])
    out["RED_abs"] = np.abs(out["RED"])

    # Derivada primera (diferencia discreta; relleno con 0 el primer valor)
    out["BLUE_d1"] = np.r_[0.0, np.diff(out["BLUE"].to_numpy())]
    out["RED_d1"]  = np.r_[0.0, np.diff(out["RED"].to_numpy())]

    # Normalizaciones opcionales (comenta si no aplica)
    # eps = 1e-9
    # out["BLUE_norm"] = (out["BLUE"] - out["BLUE"].mean()) / (out["BLUE"].std(ddof=0) + eps)
    # out["RED_norm"]  = (out["RED"]  - out["RED"].mean())  / (out["RED"].std(ddof=0)  + eps)

    # Asegura que no queden NaN/Inf (H2O se queja)
    for c in out.columns:
        col = out[c].to_numpy(dtype=float)
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = 0.0
        out[c] = col

    return out
