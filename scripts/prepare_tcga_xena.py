"""
Build a real, app-ready breast cancer dataset from TCGA-BRCA (UCSC Xena),
the same source used in the research paper.

Output: data/tcga_brca_stage.csv
  rows = primary tumour samples, columns = genes (log2(norm_count+1)),
  plus `stage` (early = Stage I-II, late = Stage III-IV), `OS_time`, `OS_event`.

Usage:
  python scripts/prepare_tcga_xena.py                      # paper genes (default list below)
  python scripts/prepare_tcga_xena.py --genes my_genes.txt # your own list, one gene per line
  python scripts/prepare_tcga_xena.py --top-variance 2000  # 2000 most variable genes

If a download link changes, open https://xenabrowser.net -> TCGA Breast Cancer (BRCA)
and copy the new links for: gene expression RNAseq (IlluminaHiSeq), phenotype
(clinicalMatrix), and curated survival data.
"""
import argparse
from pathlib import Path

import pandas as pd

HUB = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
EXPR_URL = HUB + "TCGA.BRCA.sampleMap%2FHiSeqV2.gz"
CLIN_URL = HUB + "TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix"
SURV_URL = HUB + "survival%2FBRCA_survival.txt"

# Genes named in the paper's figures. Replace with your full 70-gene list via --genes.
PAPER_GENES = [
    "GATA3", "PDGFRA", "FOXA1", "PIK3CA", "ATM", "ERBB2", "BRAF", "JUN", "EP300", "CDK2",
    "NCOR1", "CDK4", "POU5F1", "CDK6", "STAT3", "NOTCH1", "SOX2", "FGFR1", "MTOR", "BRCA1",
    "RUNX1", "NCOR2", "FOS", "NANOG", "VEGFA", "KRAS", "AR",
]


def stage_group(s: str):
    if not isinstance(s, str) or "stage" not in s.lower():
        return None
    roman = s.lower().replace("stage", "").strip().rstrip("abc").upper()
    if roman in ("I", "II"):
        return "early"
    if roman in ("III", "IV"):
        return "late"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", help="text file with one gene symbol per line")
    ap.add_argument("--top-variance", type=int, help="keep N most variable genes instead of a list")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "tcga_brca_stage.csv"))
    a = ap.parse_args()

    print("Downloading expression matrix (~150 MB, a few minutes)…")
    expr = pd.read_csv(EXPR_URL, sep="\t", index_col=0, compression="gzip").T  # samples x genes
    print("Downloading clinical and survival data…")
    clin = pd.read_csv(CLIN_URL, sep="\t", index_col=0)
    surv = pd.read_csv(SURV_URL, sep="\t", index_col=0)

    # primary solid tumour samples only (barcode ends in -01)
    expr = expr[expr.index.str[13:15] == "01"]

    if a.genes:
        genes = [g.strip() for g in Path(a.genes).read_text().splitlines() if g.strip()]
    elif a.top_variance:
        genes = list(expr.var().sort_values(ascending=False).index[: a.top_variance])
    else:
        genes = PAPER_GENES
    missing = [g for g in genes if g not in expr.columns]
    if missing:
        print("Not found in the matrix (skipped):", ", ".join(missing))
    genes = [g for g in genes if g in expr.columns]

    df = expr[genes].copy()
    df["stage"] = clin.reindex(df.index)["pathologic_stage"].map(stage_group)
    s = surv.reindex(df.index)
    df["OS_time"] = s.get("OS.time")
    df["OS_event"] = s.get("OS")
    df = df.dropna(subset=["stage"])
    df.insert(0, "sample", df.index)
    Path(a.out).parent.mkdir(exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Saved {a.out}: {len(df)} samples, {len(genes)} genes")
    print(df["stage"].value_counts().to_string())


if __name__ == "__main__":
    main()
