# model.py
from flask import Flask, Blueprint, request, jsonify
import numpy as np
from utils_r1 import (
    read_json_payload,
    waveform_preprocessing,
    convert_B_to_H,
    compute_spectral_analysis,
    calculate_resistivity,
    calculate_skin_depth,
    resitivity_depth_interpolation,
    jsonify_nan,
)

app = Flask(__name__)
bp = Blueprint("model_r1", __name__)

DEFAULTS = {
    "electrodes_distance_in_meters": 2.0,
    "buffer_in_seconds": 0.0382,
    "mur": 1.0,
    "demean": True,
    "taper": True,
    "taper_window": "hann",
    "taper_alpha": 0.1,
    "notch": True,
    "notch_freq": 50.0,
    "notch_q": 30.0,
    "max_time_msec": 200,
    "coh_threshold": 0.5,
    "nperseg": 256,
    "noverlap": None,
}

@bp.route("/api/model-r1", methods=["POST"])
def model_r1():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        cfg = DEFAULTS.copy()
        cfg.update(payload.get("config", {}) or {})

        # Lectura
        df, meta = read_json_payload(payload, buffer_in_seconds=cfg["buffer_in_seconds"])
        fs = float(meta["sampling"])
        timevec = df["time"].to_numpy()

        # Señales
        V1 = df["V1"].to_numpy()
        V2 = df["V2"].to_numpy()
        Bx = df["Bx"].to_numpy()
        By = df["By"].to_numpy()
        Bz = df["Bz"].to_numpy()

        # Preprocesamiento
        pre = dict(
            fs=fs,
            demean=bool(cfg["demean"]),
            notch=bool(cfg["notch"]),
            notch_freq=float(cfg["notch_freq"]),
            notch_q=float(cfg["notch_q"]),
            taper=bool(cfg["taper"]),
            taper_window=str(cfg["taper_window"]),
            taper_alpha=float(cfg["taper_alpha"]),
        )
        V1 = waveform_preprocessing(V1, **pre)
        V2 = waveform_preprocessing(V2, **pre)
        Bx = waveform_preprocessing(Bx, **pre)
        By = waveform_preprocessing(By, **pre)
        Bz = waveform_preprocessing(Bz, **pre)

        # Campos
        d = float(cfg["electrodes_distance_in_meters"])
        mur = float(cfg["mur"])
        E1 = V1 / d
        E2 = V2 / d
        Hx = convert_B_to_H(Bx, mur)
        Hy = convert_B_to_H(By, mur)
        Hz = convert_B_to_H(Bz, mur)

        # Ventana temporal
        mask = timevec <= (float(cfg["max_time_msec"]) / 1000.0)
        time_ms = (timevec[mask] * 1000.0).astype(float)
        E1 = E1[mask]; E2 = E2[mask]
        Hx = Hx[mask]; Hy = Hy[mask]; Hz = Hz[mask]

        def analyze(E, H):
            E_amp, H_amp, _, _, coh, f = compute_spectral_analysis(
                E, H, fs, nperseg=int(cfg["nperseg"]), noverlap=cfg["noverlap"]
            )
            rho = calculate_resistivity(E_amp, H_amp, mur)
            if coh.size and f.size and rho.size:
                valid = coh > float(cfg["coh_threshold"])
                fv = f[valid]; rhov = rho[valid]
                delta, rhoc = calculate_skin_depth(rhov, fv)
            else:
                delta, rhoc = np.array([]), np.array([])
            return {"f": f, "coh": coh, "rho": rho, "delta": delta, "rho_cut": rhoc}

        e1hx = analyze(E1, Hx)
        e1hy = analyze(E1, Hy)
        e1hz = analyze(E1, Hz)
        e2hx = analyze(E2, Hx)
        e2hy = analyze(E2, Hy)
        e2hz = analyze(E2, Hz)

        def interp(dlt, rho):
            if dlt.size and rho.size:
                d_new, r_new = resitivity_depth_interpolation(dlt, rho)
                return d_new, r_new
            return np.array([]), np.array([])

        d_e1hx, r_e1hx = interp(e1hx["delta"], e1hx["rho_cut"])
        d_e1hy, r_e1hy = interp(e1hy["delta"], e1hy["rho_cut"])
        d_e1hz, r_e1hz = interp(e1hz["delta"], e1hz["rho_cut"])
        d_e2hx, r_e2hx = interp(e2hx["delta"], e2hx["rho_cut"])
        d_e2hy, r_e2hy = interp(e2hy["delta"], e2hy["rho_cut"])
        d_e2hz, r_e2hz = interp(e2hz["delta"], e2hz["rho_cut"])

        out = {
            "electric_field": {"E1": E1.tolist(), "E2": E2.tolist(), "time_ms": time_ms.tolist()},
            "magnetic_field": {"Hx": Hx.tolist(), "Hy": Hy.tolist(), "Hz": Hz.tolist(), "time_ms": time_ms.tolist()},
            "resistivity_e1hx": {"rho_xx_1": e1hx["rho_cut"].tolist(), "delta_xx_1": e1hx["delta"].tolist(), "rho_new": r_e1hx.tolist(), "depth_new": d_e1hx.tolist()},
            "resistivity_e1hy": {"rho_xy_1": e1hy["rho_cut"].tolist(), "delta_xy_1": e1hy["delta"].tolist(), "rho_new": r_e1hy.tolist(), "depth_new": d_e1hy.tolist()},
            "resistivity_e1hz": {"rho_xz_1": e1hz["rho_cut"].tolist(), "delta_xz_1": e1hz["delta"].tolist(), "rho_new": r_e1hz.tolist(), "depth_new": d_e1hz.tolist()},
            "resistivity_e2hx": {"rho_xx_2": e2hx["rho_cut"].tolist(), "delta_xx_2": e2hx["delta"].tolist(), "rho_new": r_e2hx.tolist(), "depth_new": d_e2hx.tolist()},
            "resistivity_e2hy": {"rho_xy_2": e2hy["rho_cut"].tolist(), "delta_xy_2": e2hy["delta"].tolist(), "rho_new": r_e2hy.tolist(), "depth_new": d_e2hy.tolist()},
            "resistivity_e2hz": {"rho_xz_2": e2hz["rho_cut"].tolist(), "delta_xz_2": e2hz["delta"].tolist(), "rho_new": r_e2hz.tolist(), "depth_new": d_e2hz.tolist()},
            "spectra": {
                "e1hx": {"f": e1hx["f"].tolist(), "coh": e1hx["coh"].tolist()},
                "e1hy": {"f": e1hy["f"].tolist(), "coh": e1hy["coh"].tolist()},
                "e1hz": {"f": e1hz["f"].tolist(), "coh": e1hz["coh"].tolist()},
                "e2hx": {"f": e2hx["f"].tolist(), "coh": e2hx["coh"].tolist()},
                "e2hy": {"f": e2hy["f"].tolist(), "coh": e2hy["coh"].tolist()},
                "e2hz": {"f": e2hz["f"].tolist(), "coh": e2hz["coh"].tolist()},
            },
            "meta": meta,
            "config_used": cfg,
        }

        return jsonify(status="ok", data=jsonify_nan(out)), 200

    except Exception as e:
        return jsonify(status="error", error=str(e)), 400

# (opcional) healthcheck
@bp.get("/health")
def health():
    return jsonify(ok=True), 200

# registra el blueprint en la app principal
app.register_blueprint(bp)
