import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import hrv as H  # noqa: E402
from core import plots as P  # noqa: E402

st.set_page_config(page_title="HRV stress check · BioXAI", page_icon="❤️", layout="wide")
st.title("HRV stress check")
st.caption("Heart rate variability from RR intervals or ECG, with an explainable stress index. "
           "Research and education only, not a diagnosis.")


def get_rr(key: str, default_demo: str):
    """Input widget: demo file, RR file, ECG file, or manual Kubios values."""
    mode = st.radio("Input", ["Demo recording", "RR intervals file", "ECG signal file", "Type Kubios values"],
                    horizontal=True, key=f"{key}_mode")
    if mode == "Demo recording":
        raw = (ROOT / "data" / default_demo).read_bytes()
        return "rr", H.parse_rr(raw)
    if mode == "RR intervals file":
        up = st.file_uploader("RR intervals (.txt / .csv). One value per line, ms or seconds. "
                              "Works with Kubios, Polar, Elite HRV and similar exports.",
                              type=["txt", "csv"], key=f"{key}_rr")
        return ("rr", H.parse_rr(up.getvalue())) if up else (None, None)
    if mode == "ECG signal file":
        up = st.file_uploader("ECG (.txt / .csv), e.g. a BIOPAC export", type=["txt", "csv"], key=f"{key}_ecg")
        fs = st.number_input("Sampling rate (Hz)", 100, 5000, 1000, key=f"{key}_fs",
                             help="BIOPAC MP36 lessons commonly record at 1000 Hz. Check your file.")
        col = st.number_input("ECG column number (0 = first, -1 = last)", -1, 20, -1, key=f"{key}_col")
        if up:
            ecg = H.parse_ecg(up.getvalue(), None if col == -1 else int(col))
            rr = H.detect_r_peaks(ecg, fs)
            st.caption(f"Detected {len(rr) + 1} heartbeats in {len(ecg) / fs:.0f} s of ECG.")
            return "rr", rr
        return None, None
    c = st.columns(4)
    vals = {
        "Mean HR (bpm)": c[0].number_input("Mean HR (bpm)", 30.0, 200.0, 75.0, key=f"{key}_hr"),
        "SDNN (ms)": c[1].number_input("SDNN (ms)", 1.0, 300.0, 40.0, key=f"{key}_sdnn"),
        "RMSSD (ms)": c[2].number_input("RMSSD (ms)", 1.0, 300.0, 35.0, key=f"{key}_rmssd"),
        "LF/HF": c[3].number_input("LF/HF (0 = unknown)", 0.0, 50.0, 0.0, key=f"{key}_lfhf"),
    }
    if vals["LF/HF"] == 0:
        vals["LF/HF"] = np.nan
    return "values", vals


def flat_values(res: dict) -> dict:
    v = dict(res["time"])
    if res["freq"]:
        v.update({k: x for k, x in res["freq"].items() if not k.startswith("_")})
    v.update(res["poincare"])
    return v


def show_quality(res):
    d = res["duration_s"]
    msg = f"{res['n_beats']} beats over {d:.0f} s · {res['n_corrected']} beats corrected as artifacts."
    if d < 60:
        st.warning(msg + " Recordings under 1 minute give unstable HRV (frequency analysis needs ≥ 2 min; "
                         "5 min is the standard).")
    elif d < 120:
        st.info(msg + " Frequency-domain values (LF, HF, LF/HF) need at least 2 minutes and are skipped.")
    else:
        st.caption(msg)


def show_stress(values: dict, baseline: dict | None = None):
    s = H.stress_index(values, baseline)
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown(f"#### Stress index: {s['score']:.0f}/100 · {s['band']}")
        st.pyplot(P.stress_gauge(s["score"]))
        for line in H.interpret(values, s):
            st.markdown(f"- {line}")
    with c2:
        if not s["contributions"].empty:
            st.pyplot(P.contribution_bar(s["contributions"]))
    ref = "your normal-day baseline" if baseline else "published 5-minute resting norms for healthy adults (Nunan et al., 2010)"
    st.caption(f"Each marker is compared with {ref}. Lower RMSSD/SDNN and higher heart rate or LF/HF add points. "
               "The index is a transparent research indicator, not a validated clinical score.")
    return s


tab1, tab2, tab3 = st.tabs(["Single recording", "Normal day vs exam day", "Thesis baseline (n = 10)"])

# ------------------------------------------------------------------ single
with tab1:
    kind, data = get_rr("single", "demo_rr_normal_day.txt")
    if kind == "rr" and data is not None and len(data) > 10:
        res = H.analyse(data)
        show_quality(res)
        vals = flat_values(res)
        t = res["time"]
        m = st.columns(5)
        m[0].metric("Mean HR", f"{t['Mean HR (bpm)']:.1f} bpm")
        m[1].metric("SDNN", f"{t['SDNN (ms)']:.1f} ms")
        m[2].metric("RMSSD", f"{t['RMSSD (ms)']:.1f} ms")
        m[3].metric("pNN50", f"{t['pNN50 (%)']:.1f} %")
        m[4].metric("LF/HF", f"{res['freq']['LF/HF']:.2f}" if res["freq"] else "needs ≥ 2 min")
        st.divider()
        show_stress(vals)
        st.divider()
        st.subheader("Kubios-style charts")
        st.pyplot(P.tachogram(res["rr"]))
        c1, c2, c3 = st.columns(3)
        with c1:
            st.pyplot(P.rr_histogram(res["rr"]))
        with c2:
            if res["freq"]:
                st.pyplot(P.psd_plot(res["freq"]))
            else:
                st.info("Power spectrum needs at least 2 minutes of data.")
        with c3:
            st.pyplot(P.poincare_plot(res["rr"], res["poincare"]))
        table = H.summary_table(res)
        with st.expander("All HRV values"):
            st.dataframe(table, use_container_width=True)
        st.download_button("Download HRV report (CSV)", H.to_csv_bytes(table), "hrv_report.csv")
    elif kind == "values":
        show_stress(data)
    else:
        st.info("Upload a file or pick the demo recording.")

# ------------------------------------------------------------------ compare
with tab2:
    st.markdown("Record the **same person** on a normal day and on an exam day (same posture, time of "
                "day and duration, ideally 5 minutes seated). The exam day is then scored against that "
                "person's own baseline.")
    a, b = st.columns(2)
    with a:
        st.markdown("##### Normal day")
        k1, d1 = get_rr("normal", "demo_rr_normal_day.txt")
    with b:
        st.markdown("##### Exam day")
        k2, d2 = get_rr("exam", "demo_rr_exam_day.txt")

    if k1 and k2 and d1 is not None and d2 is not None:
        if k1 == "rr":
            r1 = H.analyse(d1)
            v1 = flat_values(r1)
        else:
            r1, v1 = None, d1
        if k2 == "rr":
            r2 = H.analyse(d2)
            v2 = flat_values(r2)
        else:
            r2, v2 = None, d2
        keys = [k for k in ["Mean HR (bpm)", "Min HR (bpm)", "Max HR (bpm)", "SDNN (ms)", "RMSSD (ms)",
                            "pNN50 (%)", "LF (n.u.)", "HF (n.u.)", "LF/HF", "SD1 (ms)", "SD2 (ms)"]
                if k in v1 and k in v2]
        comp = pd.DataFrame({"Normal day": [v1[k] for k in keys], "Exam day": [v2[k] for k in keys]}, index=keys)
        comp["Change"] = comp["Exam day"] - comp["Normal day"]
        comp["Change (%)"] = 100 * comp["Change"] / comp["Normal day"].replace(0, np.nan)
        expected = {"Mean HR (bpm)": "↑", "SDNN (ms)": "↓", "RMSSD (ms)": "↓", "pNN50 (%)": "↓",
                    "LF (n.u.)": "↑", "HF (n.u.)": "↓", "LF/HF": "↑", "SD1 (ms)": "↓"}
        comp["Expected under stress"] = [expected.get(k, "") for k in keys]
        comp["Matches?"] = ["✓" if (expected.get(k) == "↑" and c > 0) or (expected.get(k) == "↓" and c < 0)
                            else ("" if k not in expected else "✗") for k, c in zip(keys, comp["Change"])]
        st.dataframe(comp.style.format({"Normal day": "{:.2f}", "Exam day": "{:.2f}", "Change": "{:+.2f}",
                                        "Change (%)": "{:+.1f}"}), use_container_width=True)
        st.caption("Expected directions come from the exam-stress literature reviewed in the thesis "
                   "(higher HR and LF/HF; lower SDNN, RMSSD and HF under stress).")
        st.divider()
        st.subheader("Exam day scored against this person's normal day")
        show_stress(v2, baseline=v1)
        if r1 and r2:
            st.divider()
            st.pyplot(P.tachogram(r1["rr"], "RR intervals: normal vs exam day", r2["rr"], ("Normal day", "Exam day")))
            c1, c2 = st.columns(2)
            with c1:
                if r1["freq"] and r2["freq"]:
                    st.pyplot(P.psd_plot(r1["freq"], r2["freq"], ("Normal day", "Exam day")))
            with c2:
                st.pyplot(P.poincare_plot(r1["rr"], r1["poincare"], r2["rr"], ("Normal day", "Exam day")))
        st.download_button("Download comparison (CSV)", comp.to_csv().encode(), "normal_vs_exam.csv")

# ------------------------------------------------------------------ thesis
with tab3:
    base = H.thesis_baseline()
    st.markdown("Normal-day HRV of **10 healthy college students** (PSS-10 ≤ 13), recorded with BIOPAC MP36 "
                "and analysed in Kubios HRV Scientific Lite 4.3.0 (thesis Table 4.3.1).")
    st.dataframe(base, use_container_width=True, hide_index=True)
    summ = base.drop(columns="Participant").agg(["mean", "std", "min", "max"]).T.round(2)
    st.dataframe(summ, use_container_width=True)
    you = None
    if st.checkbox("Compare a recording from the 'Single recording' tab"):
        kind, data = get_rr("cmp", "demo_rr_normal_day.txt")
        if kind == "rr" and data is not None:
            you = H.analyse(data)["time"]
        elif kind == "values":
            you = data
    st.pyplot(P.thesis_cohort(base, you))
    st.caption("These recordings lasted about 25–30 seconds. Ultra-short recordings give lower SDNN than "
               "5-minute recordings, so compare like with like.")
    st.download_button("Download baseline (CSV)", base.to_csv(index=False).encode(), "thesis_baseline.csv")
