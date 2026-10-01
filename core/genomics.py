"""
Core analysis for the breast cancer stage module.

Pipeline (based on Biswas, Huda et al., "Interpretable Machine Learning Framework
for Gene Regulatory Network and Pathway Analysis in Breast Cancer"):
  preprocessing -> stratified train/test split -> SMOTE on TRAINING data only
  -> LR / SVM / RF / XGBoost -> metrics + ROC -> SHAP -> correlation network
  -> hub genes -> KEGG enrichment.

Optional libraries (xgboost, shap, imblearn, gseapy) are used when installed;
otherwise transparent fallbacks are used so the app always runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:  # pragma: no cover
    HAS_SHAP = False

try:
    import gseapy
    HAS_GSEAPY = True
except Exception:  # pragma: no cover
    HAS_GSEAPY = False

LABEL_CANDIDATES = ["stage", "label", "class", "target", "stage_group"]


# ----------------------------------------------------------------------------
# Loading and preprocessing
# ----------------------------------------------------------------------------
def find_label_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.strip().lower() in LABEL_CANDIDATES:
            return c
    return None


def encode_labels(y: pd.Series) -> tuple[np.ndarray, dict]:
    """Map labels to 0 (early) / 1 (late). Accepts text or numbers."""
    s = y.astype(str).str.strip().str.lower()
    early_words = {"early", "0", "stage i", "stage ii", "i", "ii", "early stage", "early-stage"}
    late_words = {"late", "1", "stage iii", "stage iv", "iii", "iv", "late stage", "late-stage"}
    if set(s.unique()) <= early_words | late_words:
        enc = s.map(lambda v: 1 if v in late_words else 0).to_numpy()
        return enc, {0: "Early stage", 1: "Late stage"}
    classes = sorted(s.unique())
    if len(classes) != 2:
        raise ValueError(f"The label column must have exactly 2 classes; found {len(classes)}: {classes[:6]}")
    mapping = {c: i for i, c in enumerate(classes)}
    return s.map(mapping).to_numpy(), {i: c for c, i in mapping.items()}


SURVIVAL_TIME = ["os_time", "os.time", "survival_time", "time", "days_to_death_or_followup", "os_days"]
SURVIVAL_EVENT = ["os_event", "os", "event", "status", "vital_status", "death"]
ID_CANDIDATES = ["sample", "sample_id", "sampleid", "patient", "patient_id", "id", "barcode"]


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in df.columns:
        if str(c).strip().lower() in candidates:
            return c
    return None


def non_gene_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    cols = [label_col]
    for cands in (SURVIVAL_TIME, SURVIVAL_EVENT, ID_CANDIDATES):
        c = find_column(df, cands)
        if c is not None and c not in cols:
            cols.append(c)
    return cols


def preprocess(df: pd.DataFrame, label_col: str, log_transform: bool = False,
               max_missing: float = 0.2, top_variance: int | None = None):
    """Clean the matrix: numeric genes only, drop sparse genes, median-impute, optional log2."""
    df = df.dropna(subset=[label_col])
    y, label_names = encode_labels(df[label_col])
    X = df.drop(columns=non_gene_columns(df, label_col)).apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.isna().mean() <= max_missing]
    X = X.loc[:, X.std(skipna=True) > 0]
    X = X.fillna(X.median())
    if log_transform:
        X = np.log2(X.clip(lower=0) + 1)
    if top_variance and X.shape[1] > top_variance:
        keep = X.var().sort_values(ascending=False).index[:top_variance]
        X = X[keep]
    return X, y, label_names


def smote(X: np.ndarray, y: np.ndarray, k: int = 5, random_state: int = 42):
    """Minimal SMOTE (Chawla et al., 2002). Applied to training data only."""
    try:
        from imblearn.over_sampling import SMOTE as _SMOTE
        return _SMOTE(random_state=random_state, k_neighbors=k).fit_resample(X, y)
    except Exception:
        pass
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() == counts.max():
        return X, y
    minority = classes[np.argmin(counts)]
    Xm = X[y == minority]
    n_new = counts.max() - counts.min()
    k = max(1, min(k, len(Xm) - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xm)
    _, idx = nn.kneighbors(Xm)
    base = rng.integers(0, len(Xm), n_new)
    neigh = idx[base, rng.integers(1, k + 1, n_new)]
    gap = rng.random((n_new, 1))
    synth = Xm[base] + gap * (Xm[neigh] - Xm[base])
    return np.vstack([X, synth]), np.concatenate([y, np.full(n_new, minority)])


# ----------------------------------------------------------------------------
# Modelling
# ----------------------------------------------------------------------------
def build_models(random_state: int = 42) -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=0.5),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8,
                                          eval_metric="logloss", random_state=random_state)
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(random_state=random_state)
    return models


def _metrics(y_true, prob, threshold=0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall (sensitivity)": recall_score(y_true, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan,
        "_cm": np.array([[tn, fp], [fn, tp]]),
    }


def run_pipeline(X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2,
                 use_smote: bool = True, cv_folds: int = 5, random_state: int = 42) -> dict:
    """Leakage-safe pipeline: split first, then scale and oversample using training data only."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y,
                                              random_state=random_state)
    scaler = StandardScaler().fit(X_tr)
    Xtr_s = scaler.transform(X_tr)
    Xte_s = scaler.transform(X_te)
    n_before = np.bincount(y_tr, minlength=2)
    if use_smote:
        Xtr_b, ytr_b = smote(Xtr_s, y_tr, random_state=random_state)
    else:
        Xtr_b, ytr_b = Xtr_s, y_tr
    n_after = np.bincount(ytr_b, minlength=2)

    results, fitted, rocs = {}, {}, {}
    for name, model in build_models(random_state).items():
        model.fit(Xtr_b, ytr_b)
        prob = model.predict_proba(Xte_s)[:, 1]
        m = _metrics(y_te, prob)
        # Cross-validated AUC on the ORIGINAL training data (SMOTE inside each fold would
        # need imblearn's Pipeline; plain CV here avoids synthetic samples leaking into folds).
        try:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            m["CV ROC-AUC (train)"] = cross_val_score(build_models(random_state)[name], Xtr_s, y_tr,
                                                      cv=cv, scoring="roc_auc").mean()
        except Exception:
            m["CV ROC-AUC (train)"] = np.nan
        results[name] = m
        fitted[name] = model
        fpr, tpr, _ = roc_curve(y_te, prob)
        rocs[name] = (fpr, tpr, m["ROC-AUC"])

    # Majority-class baseline: what accuracy you get by always predicting the larger class.
    majority = int(np.bincount(y_tr).argmax())
    base_prob = np.full(len(y_te), float(majority))
    results["Majority-class baseline"] = _metrics(y_te, base_prob)
    results["Majority-class baseline"]["ROC-AUC"] = 0.5
    results["Majority-class baseline"]["CV ROC-AUC (train)"] = 0.5

    table = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                          for k, v in results.items()}).T
    best = table.drop(index="Majority-class baseline")["ROC-AUC"].idxmax()
    return {
        "table": table, "results": results, "models": fitted, "rocs": rocs, "best": best,
        "scaler": scaler, "features": list(X.columns),
        "X_train": Xtr_s, "X_test": Xte_s, "y_train": y_tr, "y_test": y_te,
        "n_before": n_before, "n_after": n_after,
    }


def explain(run: dict, model_name: str | None = None, max_samples: int = 300) -> pd.DataFrame:
    """Mean |SHAP| per gene (TreeExplainer/LinearExplainer); permutation importance fallback."""
    name = model_name or run["best"]
    model = run["models"][name]
    Xte = run["X_test"][:max_samples]
    feats = run["features"]
    method = "Permutation importance (ROC-AUC drop)"
    values = None
    signed = None
    if HAS_SHAP:
        try:
            if name in ("XGBoost", "Random Forest", "Gradient Boosting"):
                sv = shap.TreeExplainer(model).shap_values(Xte)
            elif name == "Logistic Regression":
                sv = shap.LinearExplainer(model, run["X_train"]).shap_values(Xte)
            else:
                bg = shap.sample(run["X_train"], 50, random_state=0)
                sv = shap.KernelExplainer(lambda d: model.predict_proba(d)[:, 1], bg).shap_values(Xte[:60])
                Xte = Xte[:60]
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            values = np.abs(sv).mean(axis=0)
            # direction: correlation between gene value and its SHAP value
            signed = np.array([np.corrcoef(Xte[:, i], sv[:, i])[0, 1] if np.std(sv[:, i]) > 0 else 0
                               for i in range(sv.shape[1])])
            method = "Mean |SHAP value|"
        except Exception:
            values = None
    if values is None:
        pi = permutation_importance(model, run["X_test"], run["y_test"], scoring="roc_auc",
                                    n_repeats=10, random_state=0)
        values = np.clip(pi.importances_mean, 0, None)
        prob = model.predict_proba(run["X_test"])[:, 1]
        signed = np.array([np.corrcoef(run["X_test"][:, i], prob)[0, 1]
                           if np.std(run["X_test"][:, i]) > 0 else 0 for i in range(len(feats))])
    df = pd.DataFrame({"gene": feats, "importance": values, "direction": signed})
    df["effect"] = np.where(df["direction"] > 0.1, "Higher expression → late stage",
                            np.where(df["direction"] < -0.1, "Higher expression → early stage", "Mixed / unclear"))
    df.attrs["method"] = method
    df.attrs["model"] = name
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Biological validation
# ----------------------------------------------------------------------------
def correlation_network(X: pd.DataFrame, genes: list[str], threshold: float = 0.5,
                        method: str = "pearson"):
    corr = X[genes].corr(method=method)
    G = nx.Graph()
    G.add_nodes_from(genes)
    for i, a in enumerate(genes):
        for b in genes[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= threshold:
                G.add_edge(a, b, weight=float(r))
    hubs = pd.DataFrame({
        "gene": list(G.nodes),
        "connections": [G.degree(n) for n in G.nodes],
        "betweenness": list(nx.betweenness_centrality(G).values()),
    }).sort_values(["connections", "betweenness"], ascending=False).reset_index(drop=True)
    return corr, G, hubs


def kegg_enrichment(genes: list[str], background: list[str] | None = None,
                    library: str = "KEGG_2021_Human") -> pd.DataFrame | None:
    """Over-representation analysis through Enrichr (needs internet + gseapy)."""
    if not HAS_GSEAPY or not genes:
        return None
    try:
        enr = gseapy.enrichr(gene_list=list(genes), gene_sets=library, organism="human",
                             background=background, outdir=None, no_plot=True)
        res = enr.results.copy()
        res = res.sort_values("Adjusted P-value")
        res["-log10(adj p)"] = -np.log10(res["Adjusted P-value"].clip(lower=1e-300))
        return res
    except Exception:
        return None


def enrichr_link(genes: list[str]) -> str:
    return "https://maayanlab.cloud/Enrichr/"


# ----------------------------------------------------------------------------
# Survival relevance (Kaplan-Meier + log-rank), when survival columns exist
# ----------------------------------------------------------------------------
def kaplan_meier(time: np.ndarray, event: np.ndarray):
    order = np.argsort(time)
    time, event = np.asarray(time, float)[order], np.asarray(event, int)[order]
    times, surv, s = [0.0], [1.0], 1.0
    at_risk = len(time)
    for t in np.unique(time):
        here = time == t
        d = event[here].sum()
        if d > 0 and at_risk > 0:
            s *= 1 - d / at_risk
            times.append(float(t))
            surv.append(s)
        at_risk -= here.sum()
    return np.array(times), np.array(surv)


def logrank_test(time, event, group) -> float:
    """Two-group log-rank test; returns the p-value."""
    from scipy.stats import chi2
    time, event, group = np.asarray(time, float), np.asarray(event, int), np.asarray(group, int)
    O_minus_E, V = 0.0, 0.0
    for t in np.unique(time[event == 1]):
        risk = time >= t
        n, n1 = risk.sum(), (risk & (group == 1)).sum()
        d = ((time == t) & (event == 1)).sum()
        d1 = ((time == t) & (event == 1) & (group == 1)).sum()
        if n < 2:
            continue
        O_minus_E += d1 - d * n1 / n
        V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    if V <= 0:
        return np.nan
    return float(chi2.sf(O_minus_E ** 2 / V, df=1))


def survival_by_gene(df: pd.DataFrame, gene: str, time_col: str, event_col: str) -> dict:
    """Split patients at the median expression of a gene and compare survival."""
    sub = df[[gene, time_col, event_col]].apply(pd.to_numeric, errors="coerce").dropna()
    high = (sub[gene] > sub[gene].median()).astype(int).to_numpy()
    t, e = sub[time_col].to_numpy(), sub[event_col].to_numpy().astype(int)
    return {
        "high": kaplan_meier(t[high == 1], e[high == 1]),
        "low": kaplan_meier(t[high == 0], e[high == 0]),
        "p": logrank_test(t, e, high), "n_high": int(high.sum()), "n_low": int((1 - high).sum()),
    }


def survival_screen(df: pd.DataFrame, genes: list[str], time_col: str, event_col: str) -> pd.DataFrame:
    rows = []
    for g in genes:
        try:
            r = survival_by_gene(df, g, time_col, event_col)
            rows.append({"gene": g, "log-rank p": r["p"]})
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("log-rank p").reset_index(drop=True)
