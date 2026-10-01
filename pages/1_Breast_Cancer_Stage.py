import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import genomics as G  # noqa: E402
from core import plots as P  # noqa: E402

st.set_page_config(page_title="Breast cancer stage · BioXAI", page_icon="🧬", layout="wide")
st.title("Breast cancer stage analysis")
st.caption("Early vs late stage from gene expression, with explanations. Research and education only.")


@st.cache_data(show_spinner=False)
def load_demo():
    return pd.read_csv(ROOT / "data" / "demo_gene_expression.csv")


@st.cache_data(show_spinner=False)
def load_upload(data: bytes, name: str):
    from io import BytesIO
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(data))
    sep = "\t" if name.lower().endswith((".tsv", ".txt")) else ","
    return pd.read_csv(BytesIO(data), sep=sep)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1. Data")
    source = st.radio("Data source", ["Demo data (synthetic)", "Upload my own file"])
    df = None
    if source.startswith("Demo"):
        df = load_demo()
        st.info("Demo data is simulated for practice. Use TCGA-BRCA for real results (see the guide below).")
    else:
        up = st.file_uploader("CSV, TSV or Excel", type=["csv", "tsv", "txt", "xlsx"])
        if up is not None:
            df = load_upload(up.getvalue(), up.name)

    st.header("2. Settings")
    log_tf = st.checkbox("Apply log2(x + 1)", value=False,
                         help="Turn on for raw counts or FPKM/TPM. UCSC Xena HiSeqV2 is already log2.")
    top_var = st.number_input("Keep the most variable genes", 10, 5000, 500, step=10,
                              help="Speeds things up on large files. Ignored if you have fewer genes.")
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    use_smote = st.checkbox("Balance classes with SMOTE (training data only)", value=True)
    run_btn = st.button("Run analysis", type="primary", use_container_width=True)

with st.expander("What file format do I need?"):
    st.markdown(
        """
One row per **sample**, one column per **gene** (numbers), plus a column named **`stage`**
with values like `early`/`late`, `0`/`1`, or `Stage I` … `Stage IV` (I–II = early, III–IV = late).

Optional columns: `sample` (an ID), and `OS_time` + `OS_event` (survival days and 1 = died, 0 = alive)
to unlock the survival tab.

To build this file from real TCGA breast cancer data, run `python scripts/prepare_tcga_xena.py`
(from the project's GitHub page).
        """
    )

if df is None:
    st.info("Choose the demo data or upload a file in the sidebar to begin.")
    st.stop()

label_col = G.find_label_column(df)
if label_col is None:
    label_col = st.selectbox("Which column holds the stage label?", df.columns)

# ------------------------------------------------------------------ run
if run_btn or "gx_run" not in st.session_state or st.session_state.get("gx_src") != (source, df.shape):
    try:
        with st.spinner("Cleaning data and training four models…"):
            X, y, label_names = G.preprocess(df, label_col, log_transform=log_tf, top_variance=int(top_var))
            if len(np.unique(y)) < 2 or np.bincount(y).min() < 6:
                st.error("Each class needs at least 6 samples. Check the stage column.")
                st.stop()
            run = G.run_pipeline(X, y, test_size=test_size, use_smote=use_smote)
            imp = G.explain(run)
        st.session_state.update(gx_run=run, gx_X=X, gx_y=y, gx_labels=label_names, gx_imp=imp,
                                gx_src=(source, df.shape), gx_df=df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

run = st.session_state.gx_run
X, y, labels = st.session_state.gx_X, st.session_state.gx_y, st.session_state.gx_labels
imp = st.session_state.gx_imp
df = st.session_state.gx_df

m1, m2, m3, m4 = st.columns(4)
m1.metric("Samples", len(y))
m2.metric("Genes analysed", X.shape[1])
m3.metric(f"{labels[1]} share", f"{y.mean():.0%}")
m4.metric("Best model (ROC-AUC)", run["best"], f"{run['table'].loc[run['best'], 'ROC-AUC']:.3f}")

tabs = st.tabs(["Models", "Explain (SHAP)", "Expression", "Network & hubs", "Pathways", "Survival", "Predict"])

# ------------------------------------------------------------------ models
with tabs[0]:
    st.subheader("How well does each model separate early from late stage?")
    show = run["table"].copy()
    st.dataframe(show.style.format("{:.3f}").highlight_max(axis=0, color="#F4D6E3",
                 subset=pd.IndexSlice[[i for i in show.index if i != "Majority-class baseline"], :]),
                 use_container_width=True)
    base_acc = run["table"].loc["Majority-class baseline", "Accuracy"]
    st.markdown(
        f"**How to read this.** A model that always guesses the bigger class already gets "
        f"**{base_acc:.1%} accuracy**, so accuracy alone can mislead. Look at **ROC-AUC** "
        f"(0.5 = guessing, 1.0 = perfect), **balanced accuracy** and **specificity**."
    )
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.pyplot(P.roc_curves(run["rocs"]))
    with c2:
        pick = st.selectbox("Confusion matrix for", list(run["models"].keys()),
                            index=list(run["models"].keys()).index(run["best"]))
        st.pyplot(P.confusion(run["results"][pick]["_cm"], labels))
        st.pyplot(P.class_balance(run["n_before"], run["n_after"], labels))
    st.caption("The data is split into training and test sets first; scaling and SMOTE are fitted on "
               "training data only, so no synthetic or test information leaks into evaluation.")

# ------------------------------------------------------------------ SHAP
with tabs[1]:
    st.subheader("Which genes drive the predictions?")
    model_pick = st.selectbox("Explain model", list(run["models"].keys()),
                              index=list(run["models"].keys()).index(run["best"]), key="shap_model")
    if model_pick != imp.attrs.get("model"):
        with st.spinner("Computing explanations…"):
            imp = G.explain(run, model_pick)
            st.session_state.gx_imp = imp
    top_n = st.slider("Genes to show", 5, min(40, len(imp)), min(15, len(imp)))
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.pyplot(P.importance_bar(imp, top_n))
    with c2:
        st.dataframe(imp.head(top_n)[["gene", "importance", "effect"]].style.format({"importance": "{:.4f}"}),
                     use_container_width=True, hide_index=True)
    st.markdown(f"Method: **{imp.attrs['method']}**. Longer bars mean the gene changed the prediction more. "
                "Colour shows direction: pink genes push towards late stage when highly expressed.")
    st.download_button("Download gene ranking (CSV)", imp.to_csv(index=False).encode(),
                       "gene_importance.csv", "text/csv")

# ------------------------------------------------------------------ heatmap
with tabs[2]:
    st.subheader("Expression patterns of the top genes")
    n_h = st.slider("Top genes in heatmap", 5, min(40, X.shape[1]), min(20, X.shape[1]), key="hm")
    st.pyplot(P.expression_heatmap(X, y, list(imp.head(n_h)["gene"]), labels))
    st.caption("Each row is a sample, each column a gene (z-scored). Red = higher than average, blue = lower.")

# ------------------------------------------------------------------ network
with tabs[3]:
    st.subheader("Gene interaction network")
    c1, c2 = st.columns(2)
    n_net = c1.slider("Genes in network (top by importance)", 5, min(70, X.shape[1]), min(30, X.shape[1]))
    thr = c2.slider("Correlation threshold |r| ≥", 0.3, 0.9, 0.5, 0.05,
                    help="The paper compared 0.7 (strong links only) with 0.5 (denser network).")
    genes_net = list(imp.head(n_net)["gene"])
    corr, Gr, hubs = G.correlation_network(X, genes_net, thr)
    a, b, c = st.columns(3)
    a.metric("Genes", Gr.number_of_nodes())
    b.metric("Links", Gr.number_of_edges())
    c.metric("Density", f"{(2 * Gr.number_of_edges() / max(1, n_net * (n_net - 1))):.3f}")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.pyplot(P.network(Gr, hubs))
    with c2:
        st.pyplot(P.hub_bar(hubs))
        st.dataframe(hubs.head(10).style.format({"betweenness": "{:.3f}"}),
                     use_container_width=True, hide_index=True)
    with st.expander("Correlation heatmap"):
        st.pyplot(P.correlation_heatmap(corr))
    st.caption("Pink links = positive correlation, teal = negative. Hub genes have the most links and may "
               "act as control points. Correlation shows co-expression, not proof of regulation.")

# ------------------------------------------------------------------ pathways
with tabs[4]:
    st.subheader("KEGG pathway enrichment")
    n_path = st.slider("Use the top genes", 10, min(200, X.shape[1]), min(30, X.shape[1]), key="np")
    gene_list = list(imp.head(n_path)["gene"])
    use_bg = st.checkbox("Use all analysed genes as background (recommended)", value=True,
                         help="Without a background, any cancer gene list looks enriched for cancer pathways.")
    if st.button("Run enrichment"):
        with st.spinner("Querying Enrichr (needs internet)…"):
            res = G.kegg_enrichment(gene_list, background=list(X.columns) if use_bg else None)
        if res is None or res.empty:
            st.warning("Enrichment could not run here (no internet or gseapy missing). "
                       "Copy the genes below into Enrichr and choose KEGG_2021_Human.")
        else:
            st.pyplot(P.enrichment_bar(res))
            st.dataframe(res[["Term", "Overlap", "Adjusted P-value", "Genes"]].head(15),
                         use_container_width=True, hide_index=True)
    st.text_area("Gene list", "\n".join(gene_list), height=150)
    st.markdown(f"[Open Enrichr in a new tab]({G.enrichr_link(gene_list)})")

# ------------------------------------------------------------------ survival
with tabs[5]:
    st.subheader("Survival relevance")
    t_col = G.find_column(df, G.SURVIVAL_TIME)
    e_col = G.find_column(df, G.SURVIVAL_EVENT)
    if t_col is None or e_col is None:
        st.info("Add `OS_time` (days) and `OS_event` (1 = died, 0 = alive) columns to your file to see "
                "whether top genes relate to patient survival. The TCGA preparation script adds these.")
    else:
        aligned = df.loc[df[label_col].notna()].reset_index(drop=True)
        screen = G.survival_screen(aligned, list(imp.head(20)["gene"]), t_col, e_col)
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.dataframe(screen.style.format({"log-rank p": "{:.2e}"}), use_container_width=True, hide_index=True)
        with c2:
            g = st.selectbox("Gene", screen["gene"])
            st.pyplot(P.km_plot(G.survival_by_gene(aligned, g, t_col, e_col), g))
        st.caption("Patients are split at the median expression of the gene (high vs low). "
                   "p-values are not corrected for testing many genes.")

# ------------------------------------------------------------------ predict
with tabs[6]:
    st.subheader("Predict stage for new samples")
    st.markdown("Upload new samples with the **same gene columns**. Missing genes are filled with the "
                "training average.")
    newf = st.file_uploader("New samples (CSV)", type=["csv"], key="newsamples")
    model_p = st.selectbox("Model", list(run["models"].keys()),
                           index=list(run["models"].keys()).index(run["best"]), key="pm")
    if newf is not None:
        new = pd.read_csv(newf)
        ids = new[G.find_column(new, G.ID_CANDIDATES)] if G.find_column(new, G.ID_CANDIDATES) else new.index
        Xn = new.reindex(columns=run["features"]).apply(pd.to_numeric, errors="coerce")
        if log_tf:
            Xn = np.log2(Xn.clip(lower=0) + 1)
        Xn = Xn.fillna(X.mean())
        prob = run["models"][model_p].predict_proba(run["scaler"].transform(Xn))[:, 1]
        out = pd.DataFrame({"sample": ids, f"P({labels[1]})": prob.round(3),
                            "prediction": np.where(prob >= 0.5, labels[1], labels[0])})
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button("Download predictions", out.to_csv(index=False).encode(), "predictions.csv")
    st.warning("Predictions are for research only and must not guide any patient's care.")
