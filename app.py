"""
BioXAI — explainable AI for biomedical biomarkers.
Run locally:  streamlit run app.py
"""
import streamlit as st

# ---- Edit these to your own details -------------------------------------
CONCEPTS_AND_RESEARCH_BY = "Tanjidul Huda & Sriporna Biswas"
CONTACT = "tanjidulhuda@email.com | sripornab13@email.com"
GITHUB_URL = "https://github.com/12345sri/bioxai"
FEEDBACK_URL = "https://docs.google.com/forms/d/e/1FAIpQLSc676MCjsh_Wm4wsQsOp4mYhtFofCM3aWOchsFxfNC1XVqt3A/viewform"
# -------------------------------------------------------------------------

st.set_page_config(page_title="BioXAI", page_icon="🧬", layout="wide")

st.markdown(
    """
    <style>
      .block-container {max-width: 1100px; padding-top: 2.2rem;}
      .hero h1 {font-size: 2.6rem; line-height: 1.1; margin-bottom: .3rem; color:#1E1B2E;}
      .hero p {font-size: 1.12rem; color:#4A4458; max-width: 60ch;}
      .tool {border-left: 4px solid #B0306A; padding: .2rem 0 .2rem 1rem; margin-bottom: .6rem;}
      .tool.hrv {border-left-color: #2A7F8F;}
      .tool h3 {margin: 0 0 .25rem 0;}
      .tool p {margin: 0; color:#4A4458;}
      .note {background:#F4EEF2; border-radius: 6px; padding: .8rem 1rem; color:#1E1B2E; font-size:.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>See why the model decided, not just what it decided.</h1>
      <p>BioXAI applies explainable machine learning to two kinds of biomedical data:
      gene expression in breast cancer, and heart rate variability under academic stress.
      Every result comes with the reasons behind it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.markdown(
        """<div class="tool"><h3>Breast cancer stage</h3>
        <p>Upload gene expression data. Compare four models, see which genes drive the prediction
        (SHAP), explore the gene network, hub genes, KEGG pathways and survival.</p></div>""",
        unsafe_allow_html=True)
    st.page_link("pages/1_Breast_Cancer_Stage.py", label="Open the genomics tool", icon="🧬")
with c2:
    st.markdown(
        """<div class="tool hrv"><h3>HRV stress check</h3>
        <p>Upload RR intervals or ECG (BIOPAC, Kubios, smartwatch). Get Kubios-style HRV charts, an
        explainable stress index, and a normal day vs exam day comparison.</p></div>""",
        unsafe_allow_html=True)
    st.page_link("pages/2_HRV_Stress_Check.py", label="Open the HRV tool", icon="❤️")
with c3:
    st.markdown(
        """<div class="tool" style="border-left-color:#C9A227"><h3>Learn</h3>
        <p>Plain-language guides on breast cancer, screening, stress, heart rate variability,
        and how explainable AI works.</p></div>""",
        unsafe_allow_html=True)
    st.page_link("pages/3_Learn.py", label="Read the guides", icon="📘")

st.write("")
st.markdown(
    """<div class="note"><b>For research and education only.</b> BioXAI does not diagnose disease
    and is not a medical device. Results must not be used for treatment decisions. If you are worried
    about your health, speak to a doctor.</div>""",
    unsafe_allow_html=True,
)

with st.expander("Who made this and what it is based on"):
    with st.expander("Who made this and what it is based on"):
    st.markdown(f"""
**Concepts & research by {CONCEPTS_AND_RESEARCH_BY}**

[Source code]({GITHUB_URL}) · Contact: {CONTACT}

The genomics tool implements the framework from **Biswas S., Huda T., Pal H., Gupta V. K. et al.**
*Interpretable Machine Learning Framework for Gene Regulatory Network and Pathway Analysis in Breast Cancer*
(TCGA data via UCSC Xena; Logistic Regression, SVM, Random Forest, XGBoost; SHAP; correlation networks; KEGG enrichment).

The HRV tool is based on **Huda T.**, *Heart Rate Changes as an Indicator of Academic Stress in College Students*,
B.Sc. (H) Biomedical Science dissertation, Bhaskaracharya College of Applied Sciences, University of Delhi (2026),
which recorded ECG with BIOPAC MP36 and analysed HRV in Kubios.

The app reproduces those HRV measures in open-source Python and supports the exam-day comparison described in the study.
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("Help us improve BioXAI")
st.write("Share your experience, rating, and suggestions.")

st.link_button("Give Feedback", FEEDBACK_URL)
    )
