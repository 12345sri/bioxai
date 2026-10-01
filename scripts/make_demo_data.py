"""
Generate the demo files in data/.

IMPORTANT: demo_gene_expression.csv and the demo RR files are SYNTHETIC.
They are simulated so people can try the app without downloading anything.
For real results use TCGA-BRCA (see scripts/prepare_tcga_xena.py) and real recordings.

Run:  python scripts/make_demo_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)

# ---------------------------------------------------------------- gene expression
# Gene modules loosely follow known biology so the network/pathway tabs look realistic.
modules = {
    "luminal":       ["GATA3", "FOXA1", "ESR1", "PGR", "AR", "NCOR1", "XBP1"],
    "proliferation": ["CDK2", "CDK4", "CDK6", "MKI67", "CCNB1", "E2F1", "MYC"],
    "pi3k_mtor":     ["PIK3CA", "MTOR", "AKT1", "ATM", "EP300", "BRAF", "KRAS", "PTEN"],
    "stemness":      ["SOX2", "NANOG", "POU5F1", "NOTCH1"],
    "ap1":           ["FOS", "JUN", "EGR1"],
    "her2":          ["ERBB2", "GRB7"],
    "angiogenesis":  ["VEGFA", "PDGFRA", "FGFR1", "HIF1A"],
    "other":         ["RUNX1", "NCOR2", "STAT3", "BRCA1", "TP53", "CDH1"],
}
n = 800
late = (rng.random(n) < 0.25).astype(int)          # ~25 % late stage, like TCGA
factors = {m: rng.normal(size=n) for m in modules}
# late-stage samples: more proliferation / HER2 / angiogenesis, less luminal
factors["proliferation"] += 0.55 * late
factors["her2"] += 0.45 * late
factors["angiogenesis"] += 0.35 * late
factors["luminal"] -= 0.30 * late

cols = {}
for m, genes in modules.items():
    for g in genes:
        loading = rng.uniform(0.6, 0.95) if m != "other" else rng.uniform(0.0, 0.3)
        base = rng.uniform(6, 12)
        cols[g] = base + loading * factors[m] + rng.normal(scale=0.8, size=n)
cols["RUNX1"] += 0.5 * late
cols["NCOR2"] -= 0.35 * late
cols["CDH1"] -= 0.3 * late

df = pd.DataFrame(cols).round(3)
df.insert(0, "sample", [f"DEMO-{i:04d}" for i in range(n)])
df["stage"] = np.where(late == 1, "late", "early")
# survival (days): worse with late stage and high proliferation
hazard = np.exp(0.9 * late + 0.35 * factors["proliferation"] - 0.2 * factors["luminal"])
t_event = rng.exponential(4000 / hazard)
t_cens = rng.uniform(300, 4500, n)
df["OS_time"] = np.minimum(t_event, t_cens).round(0)
df["OS_event"] = (t_event <= t_cens).astype(int)
df.to_csv(DATA / "demo_gene_expression.csv", index=False)
print("gene demo:", df.shape, "late fraction:", late.mean().round(3))


# ---------------------------------------------------------------- RR intervals (5 min)
def simulate_rr(mean_hr, rsa_ms, mayer_ms, noise_ms, seconds=300, seed=0):
    r = np.random.default_rng(seed)
    t, rr = 0.0, []
    mean_rr = 60000 / mean_hr
    while t < seconds:
        v = (mean_rr + rsa_ms * np.sin(2 * np.pi * 0.25 * t)       # breathing (HF)
             + mayer_ms * np.sin(2 * np.pi * 0.1 * t + 1.0)          # baroreflex (LF)
             + r.normal(scale=noise_ms))
        rr.append(v)
        t += v / 1000
    return np.round(np.array(rr), 1)


normal = simulate_rr(mean_hr=70, rsa_ms=45, mayer_ms=30, noise_ms=18, seed=1)
exam = simulate_rr(mean_hr=86, rsa_ms=14, mayer_ms=28, noise_ms=10, seed=2)
np.savetxt(DATA / "demo_rr_normal_day.txt", normal, fmt="%.1f", header="RR intervals (ms) - SYNTHETIC normal day")
np.savetxt(DATA / "demo_rr_exam_day.txt", exam, fmt="%.1f", header="RR intervals (ms) - SYNTHETIC exam day")
print("rr demo:", len(normal), len(exam))
