import streamlit as st

st.set_page_config(page_title="Learn · BioXAI", page_icon="📘", layout="wide")
st.title("Learn")
st.caption("Plain-language guides. General information only; it does not replace advice from a doctor.")

t1, t2, t3, t4 = st.tabs(["Breast cancer", "Stress and your heart", "How the AI explains itself", "Get help"])

with t1:
    st.subheader("What is breast cancer staging?")
    st.markdown(
        """
Staging describes how far a cancer has grown or spread. **Early stages (I–II)** are usually confined
to the breast and nearby lymph nodes. **Later stages (III–IV)** have spread more widely. Earlier
detection generally means more treatment options and better outcomes.

**Why genes matter.** Tumours switch some genes on and others off. Genes such as *ERBB2* (HER2),
*GATA3* and *FOXA1* help doctors classify tumours and choose treatments; for example, HER2-positive
cancers can be treated with HER2-targeted drugs. Research tools like this one look for patterns in
thousands of genes at once.
        """
    )
    st.subheader("Signs worth checking with a doctor")
    st.markdown(
        """
A new lump in the breast or armpit; a change in breast size or shape; skin dimpling, redness or
thickening; a nipple turning inward; or unusual discharge. Most lumps are **not** cancer, but any
change should be checked promptly.
        """
    )
    st.subheader("Screening")
    st.markdown(
        """
Get to know how your breasts normally look and feel, and report changes. Clinical breast examination
by a trained health worker is offered through many public health programmes in India. Mammography
is generally recommended for women from around age 40–50, depending on national guidelines and
personal risk. People with a strong family history should ask a doctor about earlier or extra
screening.
        """
    )

with t2:
    st.subheader("How stress shows up in your heartbeat")
    st.markdown(
        """
Your heart does not beat like a metronome. The tiny changes in time between beats are called
**heart rate variability (HRV)**. They come from two branches of the nervous system:

the **sympathetic** side ("fight or flight") speeds the heart and makes beats more regular, and the
**parasympathetic** side ("rest and digest") slows the heart and adds healthy variation, especially
as you breathe.

Under stress, such as before exams, heart rate tends to rise and HRV tends to fall. That is why HRV
is used as an objective, non-invasive marker of stress in research.
        """
    )
    st.subheader("The numbers in the HRV tool")
    st.markdown(
        """
**SDNN** is overall variability. **RMSSD** reflects beat-to-beat (parasympathetic) variation and is
the most reliable marker in short recordings. **LF/HF** compares slower and faster rhythms; a higher
ratio can suggest sympathetic dominance, though scientists debate how to interpret it. **SD1/SD2** come
from the Poincaré plot and describe short- and long-term variability.

HRV is also affected by sleep, caffeine, exercise, illness, age, breathing and posture. Compare
yourself with yourself, at the same time of day and in the same position.
        """
    )
    st.subheader("Lowering exam stress")
    st.markdown(
        """
Slow breathing (about 6 breaths per minute: breathe in for 5 seconds, out for 5) can raise HRV within
minutes. Regular sleep, physical activity, short breaks while studying, and talking to friends,
family or a counsellor all help.
        """
    )

with t3:
    st.subheader("Explainable AI in one minute")
    st.markdown(
        """
Many AI models give an answer without saying why. **Explainable AI** shows the reasons.

In the genomics tool, **SHAP values** measure how much each gene pushed a prediction towards early or
late stage, like splitting a team's score fairly between the players. In the HRV tool, the stress
index shows exactly how many points each heart measure added or removed.

Seeing the reasons lets researchers check whether the model learned real biology or a quirk of the
data, which is essential before any tool could be trusted in healthcare.
        """
    )

with t4:
    st.subheader("If you need support")
    st.markdown(
        """
**Health concerns:** see a doctor or visit your nearest government hospital or health centre.

**Feeling overwhelmed by stress:** talk to someone you trust or your college counsellor. In India,
**Tele-MANAS** offers free mental health support at **14416** or **1-800-891-4416**.

**Emergency:** call **112**.
        """
    )
