"""
Core analysis for the HRV stress module.

Based on Huda T., "Heart Rate Changes as an Indicator of Academic Stress in College
Students" (B.Sc. (H) Biomedical Science dissertation, BCAS, University of Delhi, 2026),
which recorded ECG with BIOPAC MP36 (modified Lead II) and analysed HRV in Kubios.

This module reproduces the standard Kubios-style outputs in open Python:
  RR intervals -> artifact correction -> time domain (HR, SDNN, RMSSD, pNN50)
  -> frequency domain (LF, HF, LF/HF via Welch PSD on 4 Hz resampled RR)
  -> nonlinear (Poincare SD1/SD2) -> explainable stress index.
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid

# Short-term (5-min) resting HRV norms for healthy adults:
# Nunan D, Sandercock GRH, Brodie DA. Pacing Clin Electrophysiol. 2010;33(11):1407-17.
NORMS = {
    "Mean HR (bpm)": (65.0, 7.0),      # from mean RR 926 +/- 90 ms
    "SDNN (ms)": (50.0, 16.0),
    "RMSSD (ms)": (42.0, 15.0),
    "LF/HF": (2.8, 2.6),
}
# Direction: +1 means a HIGHER value points to more stress (sympathetic dominance).
STRESS_DIRECTION = {"Mean HR (bpm)": +1, "SDNN (ms)": -1, "RMSSD (ms)": -1, "LF/HF": +1}
STRESS_WEIGHT = {"Mean HR (bpm)": 1.0, "SDNN (ms)": 1.0, "RMSSD (ms)": 1.5, "LF/HF": 0.75}


# ----------------------------------------------------------------------------
# Input parsing
# ----------------------------------------------------------------------------
def _numeric_table(raw: bytes | str) -> pd.DataFrame:
    """Read a messy text export (BIOPAC/Kubios/smartwatch) into a numeric table."""
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
    rows = []
    for line in text.splitlines():
        parts = [p for p in line.replace(",", " ").replace(";", " ").replace("\t", " ").split() if p]
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue  # skip header / comment lines
        if vals:
            rows.append(vals)
    if not rows:
        raise ValueError("No numeric values were found in the file.")
    width = max(set(len(r) for r in rows), key=[len(r) for r in rows].count)
    rows = [r for r in rows if len(r) == width]
    return pd.DataFrame(rows)


def parse_rr(raw: bytes | str, column: int | None = None) -> np.ndarray:
    """RR intervals in ms. Accepts seconds or ms; one value per line or a column."""
    df = _numeric_table(raw)
    col = df.shape[1] - 1 if column is None else column
    rr = df.iloc[:, col].to_numpy(dtype=float)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if np.median(rr) < 3:  # values look like seconds
        rr = rr * 1000.0
    return rr


def parse_ecg(raw: bytes | str, column: int | None = None) -> np.ndarray:
    df = _numeric_table(raw)
    col = df.shape[1] - 1 if column is None else column
    return df.iloc[:, col].to_numpy(dtype=float)


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Simple Pan-Tompkins-style R-peak detector. Returns RR intervals in ms."""
    nyq = fs / 2
    b, a = signal.butter(3, [5 / nyq, min(15 / nyq, 0.99)], btype="band")
    filt = signal.filtfilt(b, a, ecg)
    # polarity check: R peaks should be the largest absolute deflection
    if np.abs(filt.min()) > np.abs(filt.max()):
        filt = -filt
    energy = np.convolve(np.gradient(filt) ** 2, np.ones(int(0.12 * fs)) / int(0.12 * fs), mode="same")
    peaks, _ = signal.find_peaks(energy, distance=int(0.3 * fs), height=np.percentile(energy, 90) * 0.35)
    # refine to the true local maximum of the filtered ECG
    w = int(0.05 * fs)
    refined = np.array([p - w + np.argmax(filt[max(p - w, 0):p + w]) for p in peaks if p - w >= 0])
    refined = np.unique(refined)
    return np.diff(refined) / fs * 1000.0


def correct_artifacts(rr: np.ndarray, threshold: float = 0.20) -> tuple[np.ndarray, int]:
    """Replace physiologically impossible or ectopic-looking beats by interpolation."""
    rr = np.asarray(rr, dtype=float)
    med = pd.Series(rr).rolling(5, center=True, min_periods=1).median().to_numpy()
    bad = (rr < 300) | (rr > 2000) | (np.abs(rr - med) > threshold * med)
    if bad.all():
        return rr, int(bad.sum())
    idx = np.arange(len(rr))
    clean = rr.copy()
    clean[bad] = np.interp(idx[bad], idx[~bad], rr[~bad])
    return clean, int(bad.sum())


# ----------------------------------------------------------------------------
# HRV metrics
# ----------------------------------------------------------------------------
def time_domain(rr: np.ndarray) -> dict:
    hr = 60000.0 / rr
    diff = np.diff(rr)
    return {
        "Mean RR (ms)": rr.mean(),
        "Mean HR (bpm)": hr.mean(),
        "Min HR (bpm)": hr.min(),
        "Max HR (bpm)": hr.max(),
        "SDNN (ms)": rr.std(ddof=1),
        "RMSSD (ms)": np.sqrt(np.mean(diff ** 2)),
        "pNN50 (%)": 100.0 * np.mean(np.abs(diff) > 50),
    }


def frequency_domain(rr: np.ndarray, fs_resample: float = 4.0) -> dict | None:
    """Welch PSD on evenly resampled RR. Needs >= 2 minutes for a valid LF estimate."""
    t = np.cumsum(rr) / 1000.0
    t -= t[0]
    if t[-1] < 120:
        return None
    grid = np.arange(0, t[-1], 1 / fs_resample)
    rr_i = CubicSpline(t, rr)(grid)
    rr_i = signal.detrend(rr_i)
    f, p = signal.welch(rr_i, fs=fs_resample, nperseg=min(256, len(rr_i)), noverlap=None)
    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return trapezoid(p[m], f[m]) if m.any() else 0.0
    vlf, lf, hf = band(0.0033, 0.04), band(0.04, 0.15), band(0.15, 0.40)
    return {
        "VLF (ms²)": vlf, "LF (ms²)": lf, "HF (ms²)": hf,
        "LF (n.u.)": 100 * lf / (lf + hf) if lf + hf else np.nan,
        "HF (n.u.)": 100 * hf / (lf + hf) if lf + hf else np.nan,
        "LF/HF": lf / hf if hf else np.nan,
        "_freq": f, "_psd": p,
    }


def poincare(rr: np.ndarray) -> dict:
    x, y = rr[:-1], rr[1:]
    sd1 = np.std((y - x) / np.sqrt(2), ddof=1)
    sd2 = np.std((y + x) / np.sqrt(2), ddof=1)
    return {"SD1 (ms)": sd1, "SD2 (ms)": sd2, "SD2/SD1": sd2 / sd1 if sd1 else np.nan}


def analyse(rr_raw: np.ndarray, correct: bool = True) -> dict:
    rr, n_fixed = correct_artifacts(rr_raw) if correct else (np.asarray(rr_raw, float), 0)
    out = {"rr": rr, "n_beats": len(rr), "duration_s": rr.sum() / 1000.0, "n_corrected": n_fixed}
    out["time"] = time_domain(rr)
    out["freq"] = frequency_domain(rr)
    out["poincare"] = poincare(rr)
    return out


def summary_table(res: dict) -> pd.DataFrame:
    rows = dict(res["time"])
    if res["freq"]:
        rows.update({k: v for k, v in res["freq"].items() if not k.startswith("_")})
    rows.update(res["poincare"])
    return pd.DataFrame({"Value": rows}).round(2)


# ----------------------------------------------------------------------------
# Explainable stress index
# ----------------------------------------------------------------------------
def stress_index(values: dict, baseline: dict | None = None) -> dict:
    """
    Transparent stress index (0-100, 50 = typical resting state).

    Each available marker is converted to a z-score against a reference
    (the person's own baseline if given, otherwise published adult norms),
    signed so that positive = more sympathetic / stress-like, weighted, averaged,
    and mapped to 0-100. The per-marker contributions are returned so the user
    can see exactly why the score is what it is (same idea as SHAP in the
    genomics module). This is a research/education indicator, not a diagnosis.
    """
    contribs = {}
    for key, direction in STRESS_DIRECTION.items():
        v = values.get(key)
        if v is None or not np.isfinite(v):
            continue
        if baseline and baseline.get(key) is not None and np.isfinite(baseline[key]):
            ref_mean = baseline[key]
            ref_sd = max(abs(baseline[key]) * 0.25, NORMS[key][1] * 0.5)
            ref = "your baseline"
        else:
            ref_mean, ref_sd = NORMS[key]
            ref = "adult norms"
        z = np.clip((v - ref_mean) / ref_sd, -3, 3) * direction
        contribs[key] = {"value": v, "reference": ref_mean, "z": z,
                         "weight": STRESS_WEIGHT[key], "ref_type": ref}
    if not contribs:
        return {"score": np.nan, "band": "Not enough data", "contributions": pd.DataFrame()}
    wsum = sum(c["weight"] for c in contribs.values())
    combined = sum(c["z"] * c["weight"] for c in contribs.values()) / wsum
    score = float(np.clip(50 + 16.7 * combined, 0, 100))
    band = "Relaxed" if score < 40 else ("Typical" if score <= 60 else ("Elevated" if score <= 75 else "High"))
    df = pd.DataFrame(contribs).T
    df["points"] = 16.7 * df["z"].astype(float) * df["weight"].astype(float) / wsum
    return {"score": score, "band": band, "contributions": df}


def interpret(values: dict, score_res: dict) -> list[str]:
    """Plain-language explanation for students and the public."""
    notes = []
    s = score_res["score"]
    if np.isfinite(s):
        notes.append(f"Overall stress index: {s:.0f}/100 ({score_res['band'].lower()}).")
    r = values.get("RMSSD (ms)")
    if r is not None:
        notes.append("RMSSD reflects the 'rest and digest' (parasympathetic) system. "
                     + ("Yours is on the lower side, which is common under stress, tiredness or poor sleep."
                        if r < 27 else "Yours is in a healthy range."))
    hr = values.get("Mean HR (bpm)")
    if hr is not None:
        notes.append("Average heart rate " + ("is raised compared with typical resting values."
                                              if hr > 80 else "is within a typical resting range."))
    lfhf = values.get("LF/HF")
    if lfhf is not None and np.isfinite(lfhf):
        notes.append("LF/HF ratio " + ("is high, suggesting the sympathetic ('fight or flight') side is dominant."
                                       if lfhf > 4 else "does not show strong sympathetic dominance."))
    return notes


def thesis_baseline() -> pd.DataFrame:
    """Normal-day HRV of 10 healthy students (thesis Table 4.3.1, Kubios, 25-30 s recordings)."""
    return pd.DataFrame({
        "Participant": [f"P{i}" for i in range(1, 11)],
        "Min HR (bpm)": [65.97, 57.85, 60.20, 58.24, 57.07, 56.63, 57.42, 57.82, 57.71, 64.52],
        "Max HR (bpm)": [97.73, 94.28, 97.78, 95.26, 96.55, 94.56, 96.84, 97.43, 92.52, 101.52],
        "PNS index": [-0.98, 2.19, -1.51, -1.40, -1.43, -1.47, -1.54, -1.52, -1.49, -1.45],
        "SNS index": [2.31, 1.82, 3.73, 2.95, 2.53, 2.88, 3.07, 2.33, 3.27, 2.14],
        "SDNN (ms)": [24.36, 19.45, 21.49, 21.12, 20.67, 20.59, 25.47, 20.90, 24.18, 23.23],
        "RMSSD (ms)": [30.50, 23.41, 26.08, 27.22, 24.73, 25.03, 26.03, 25.24, 28.06, 24.93],
    })


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf)
    return buf.getvalue().encode()
