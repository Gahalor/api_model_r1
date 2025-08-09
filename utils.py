# utils_r1.py
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import iirnotch, filtfilt, coherence, get_window
from scipy.interpolate import interp1d
from typing import Any

MU0 = 4 * np.pi * 1e-7  # H/m

def safe_get(data, *keys, default=None):
    for k in keys:
        if isinstance(data, dict) and k in data:
            data = data[k]
        else:
            return default
    return data

def read_json_payload(payload, buffer_in_seconds: float = 0.0):
    data = payload.get("data", payload)

    geophone = safe_get(data, "geophone", "data") or safe_get(data, "samples", "geophone", "data") or {}
    se_data  = safe_get(data, "seismoelectric", "data") or safe_get(data, "samples", "seismoelectric", "data") or {}
    mag_data = safe_get(data, "magnetometer", "data") or safe_get(data, "samples", "magnetometer", "data") or {}

    if safe_get(data, "samples", "magnetometer", "sampleRate") is not None:
        samplerate = safe_get(data, "samples", "magnetometer", "sampleRate")
    else:
        samplerate = safe_get(data, "magnetometer", "samplerate")

    gx = np.array(geophone.get("x", []), float)
    gy = np.array(geophone.get("y", []), float)
    gz = np.array(geophone.get("z", []), float)

    v1 = np.array(se_data.get("v1", []), float) * 1e-3  # V
    v2 = np.array(se_data.get("v2", []), float) * 1e-3

    bx = np.array(mag_data.get("x", []), float) * 1e-9  # T
    by = np.array(mag_data.get("y", []), float) * 1e-9
    bz = np.array(mag_data.get("z", []), float) * 1e-9

    fs = float(samplerate) if samplerate else 3333.0
    dt = 1.0 / fs
    valid_lengths = [len(a) for a in [gx, gy, gz, v1, v2, bx, by, bz] if len(a)]
    n = min(valid_lengths) if valid_lengths else 0

    time_s = np.arange(0, n * dt, dt)
    gx, gy, gz, v1, v2, bx, by, bz = [a[:n] for a in (gx, gy, gz, v1, v2, bx, by, bz)]

    df = pd.DataFrame(
        np.array([time_s, gx, gy, gz, v1, v2, bx, by, bz]).T,
        columns=["time", "Gx", "Gy", "Gz", "V1", "V2", "Bx", "By", "Bz"]
    )

    if buffer_in_seconds and n:
        df = df[df["time"] >= buffer_in_seconds].copy()
        df["time"] = df["time"] - df["time"].iloc[0]
        df.reset_index(drop=True, inplace=True)

    meta = {
        "projectName": data.get("projectName", "undefined"),
        "timezone": data.get("timezone", "undefined"),
        "timestamp": data.get("timestamp", "undefined"),
        "geolocation": data.get("geolocation", []),
        "deviceId": data.get("deviceId", "undefined"),
        "temperature": data.get("temperature", None),
        "humidity": data.get("humidity", None),
        "sampling": fs,
    }
    return df, meta

def apply_taper(x: np.ndarray, window: str = "hann", alpha: float = 0.1) -> np.ndarray:
    n = len(x)
    if n == 0:
        return x
    if window == "tukey":
        w = get_window(("tukey", alpha), n, fftbins=False)
    else:
        w = get_window(window, n, fftbins=False)
    return x * w

def waveform_preprocessing(
    x: np.ndarray,
    fs: float,
    *,
    demean: bool = True,
    notch: bool = False,
    notch_freq: float = 50.0,
    notch_q: float = 30.0,
    taper: bool = False,
    taper_window: str = "hann",
    taper_alpha: float = 0.1,
) -> np.ndarray:
    y = np.asarray(x, float)
    if y.size == 0:
        return y
    if demean:
        y = y - np.nanmean(y)
    if notch:
        b, a = iirnotch(notch_freq, Q=notch_q, fs=fs)
        if y.size > max(len(a), len(b)) * 3:
            y = filtfilt(b, a, y)
    if taper:
        y = apply_taper(y, window=taper_window, alpha=taper_alpha)
    return y

def convert_B_to_H(B: np.ndarray, mur: float = 1.0, mu0: float = MU0) -> np.ndarray:
    return B / (mu0 * mur)

def compute_spectral_analysis(E: np.ndarray, H: np.ndarray, fs: float, nperseg: int = 256, noverlap: int | None = None):
    dt = 1.0 / fs
    n = len(E)
    if n == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    E_fft = fft(E)
    H_fft = fft(H)
    freqs = fftfreq(n, dt)
    pos = freqs > 0
    f = freqs[pos]
    E_amp = (2.0 / n) * np.abs(E_fft[pos])
    H_amp = (2.0 / n) * np.abs(H_fft[pos])
    E_pha = np.angle(E_fft[pos], deg=True)
    H_pha = np.angle(H_fft[pos], deg=True)

    if noverlap is None:
        noverlap = nperseg // 2
    nperseg_eff = max(16, min(nperseg, n // 2 if n < nperseg else nperseg))
    noverlap_eff = max(8, min(noverlap, nperseg_eff // 2))

    f_coh, coh = coherence(E, H, fs=fs, window="hann", nperseg=nperseg_eff, noverlap=noverlap_eff)
    if f.size and f_coh.size:
        coh = np.interp(f, f_coh, coh)
    else:
        coh = np.array([])

    return E_amp, H_amp, E_pha, H_pha, coh, f

def calculate_resistivity(E_amp: np.ndarray, H_amp: np.ndarray, mur: float = 1.0, mu0: float = MU0) -> np.ndarray:
    if E_amp.size == 0 or H_amp.size == 0:
        return np.array([])
    H_amp = np.where(H_amp == 0, 1e-12, H_amp)
    Z = E_amp / H_amp
    return (Z ** 2) / (mu0 * mur)

def calculate_skin_depth(rho: np.ndarray, f_vec: np.ndarray, mu0: float = MU0, zmin: float = 0.0, zmax: float = 300.0):
    if rho.size == 0 or f_vec.size == 0:
        return np.array([]), np.array([])
    w = 2 * np.pi * f_vec
    w = np.where(w == 0, 1e-12, w)
    delta = np.sqrt((2.0 * rho) / (w * mu0))
    idx = np.argsort(delta)
    delta = delta[idx]
    rho = rho[idx]
    mask = (delta >= zmin) & (delta <= zmax)
    return delta[mask], rho[mask]

def resitivity_depth_interpolation(x: np.ndarray, y: np.ndarray, num_points: int = 10, kind: str = "nearest", space: str = "log"):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size == 0:
        return np.array([]), np.array([])
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]
    x_min, x_max = x_sorted[0], x_sorted[-1]
    if x_min <= 0:
        x_min = np.nextafter(0, 1)
    x_new = np.logspace(np.log10(x_min), np.log10(x_max), num=num_points) if space == "log" else np.linspace(x_min, x_max, num=num_points)
    f = interp1d(x_sorted, y_sorted, kind=kind, fill_value="extrapolate")
    y_new = f(x_new)
    return x_new, y_new

def jsonify_nan(obj: Any):
    if isinstance(obj, dict):
        return {k: jsonify_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonify_nan(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonify_nan(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj
