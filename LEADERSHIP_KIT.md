# BioXAI leadership kit

The app shows skill. These steps turn it into leadership: other people use it, you guide a team,
you teach, and you create impact. Do them in order; replace everything in `<angle brackets>`.

---

## 1. Email to your supervisor / HOD (week 1)

**Subject:** Request: BioXAI demo session and student stress-check pilot

Respected <Dr. Neha Singh / Dr. Aparna Dixit / HOD name>,

I have built **BioXAI**, a free web app that turns my dissertation (HRV and academic stress) and our
breast cancer research paper into tools others can use: <app link>.

I would like your guidance and permission for two activities in the department:

1. A **one-hour demo workshop** for students on explainable AI in biomedical research.
2. A small **HRV stress-check pilot** before the exams, using the department's BIOPAC system with
   informed consent, anonymised data and your supervision. This would also complete the exam-day
   comparison my dissertation could not finish because of the equipment issue.

I would also like to involve 2–3 junior students as a small team. I would be grateful for 10 minutes
of your time to discuss this.

Regards,
Tanjidul Huda, <roll no.>, <phone>

---

## 2. WhatsApp message to recruit juniors (after approval)

> Hi everyone! I've built **BioXAI**, a web app on AI + biomedical data (breast cancer genes and
> heart-rate stress analysis): <link>
> I'm looking for **2–3 students** to join a small team for 4 weeks (2–3 hrs/week). You'll learn
> Python, HRV/ECG analysis, GitHub and how research tools are built, and you'll be credited on the
> project and in the workshop. No coding experience needed, just curiosity.
> Interested? Reply here or DM me by <date>.

---

## 3. Team plan (4 weeks)

| Role | Who | Tasks |
|---|---|---|
| Lead (you) | Tanjidul | Plan, weekly 30-min meeting, review work, final decisions |
| Testing and feedback | Junior 1 | Test every page on phone and laptop, log bugs in GitHub Issues, run the feedback form |
| Data | Junior 2 | Run `prepare_tcga_xena.py`, test on real TCGA data, record results |
| Outreach | Junior 3 | Workshop poster, sign-ups, photos, LinkedIn content |

| Week | Goal |
|---|---|
| 1 | App live, team formed, roles explained, everyone has tried the app |
| 2 | Bugs fixed, real TCGA results added to README, workshop date fixed |
| 3 | Workshop held, feedback collected |
| 4 | Stress-check pilot (if approved), results summary, LinkedIn post, update CV/SOP |

Keep a simple log (date, what was done, who). It becomes evidence for your applications.

---

## 4. Workshop plan (60 minutes)

**Title:** Explainable AI in biomedicine: from genes to heartbeats

| Time | Part |
|---|---|
| 0–10 | Why AI must explain itself in healthcare (black box vs explainable) |
| 10–25 | Live demo 1: breast cancer module. Why 83% accuracy can mean nothing (majority baseline), ROC-AUC, SHAP genes, hub genes |
| 25–40 | Live demo 2: HRV module. Record or load RR data, show the stress index and "Why this score?" |
| 40–50 | Hands-on: everyone opens the link on their phone and tries the demo data |
| 50–55 | Q&A |
| 55–60 | Feedback form (QR code) + invite to join the team |

Bring: laptop, projector, QR code to the app and the form, an attendance sheet, someone taking photos.

---

## 5. Feedback form (Google Forms)

1. Your role: Student / Researcher / Faculty / Other
2. Which part did you try? Breast cancer / HRV / Learn
3. How easy was it to use? (1–5)
4. How clearly did it explain the results? (1–5)
5. Did you learn something new about explainable AI? Yes / Somewhat / No
6. Would you use or recommend this tool? Yes / Maybe / No
7. What should be improved?
8. Want to join the team or get updates? (optional email)

Save the summary (number of responses, average scores, top suggestions). Act on 2–3 suggestions and
record what you changed: "acted on user feedback" is strong evidence.

---

## 6. HRV stress-check pilot (only with written approval)

- Supervisor approval; ask whether institutional ethics review is needed.
- Written informed consent; participants can withdraw any time.
- Same protocol on both days: seated, 5 minutes, same time of day, no caffeine for 2 hours.
- Normal day (2+ weeks before exams) and exam day (before an exam, not after).
- Anonymous IDs only; no names stored with data. Share results privately with each participant.
- Give every participant the Tele-MANAS number (14416) and college counsellor details.
- Report it as a pilot, not as a diagnosis or a clinical study.

---

## 7. LinkedIn post (after launch)

> I turned my undergraduate research into a free tool anyone can use. 🧬❤️
>
> **BioXAI** brings together two projects: my dissertation on heart rate variability as a marker of
> academic stress, and our research on explainable machine learning for breast cancer staging.
>
> What it does:
> • Breast cancer module: compares four ML models on gene expression and shows *which genes* drive
> each prediction (SHAP), plus gene networks, KEGG pathways and survival curves.
> • HRV module: analyses ECG or RR intervals, gives a stress index that explains itself, and compares
> a normal day with an exam day.
> • Learn section: plain-language guides on breast cancer, screening and stress.
>
> The biggest lesson: in healthcare AI, *why* matters as much as *what*. A model can score 83%
> accuracy just by predicting the majority class, so the app always shows an honest baseline.
>
> Built with Python and Streamlit. Open source. For research and education only.
> 🔗 App: <link> · Code: <GitHub link>
>
> Thanks to Dr. Neha Singh, Dr. Aparna Dixit and my co-authors. Feedback very welcome!
>
> #Bioinformatics #ExplainableAI #ComputationalBiology #BreastCancer #HRV #Python

---

## 8. Demo video script (about 2 minutes; record your screen with your voice)

| Time | Show | Say |
|---|---|---|
| 0:00 | Home page | "Hi, I'm Tanjidul. This is BioXAI, a free tool that explains its AI predictions on biomedical data." |
| 0:15 | Breast cancer → Models | "It compares four models. Notice the majority baseline: accuracy alone can fool you, so I focus on ROC-AUC." |
| 0:40 | Explain (SHAP) | "These are the genes driving the predictions, and whether higher expression points to late stage." |
| 0:55 | Network & hubs | "The gene network shows co-expressed clusters; filled nodes are hub genes." |
| 1:10 | HRV → Normal vs exam | "From my thesis: here's a normal day versus an exam day. Heart rate rises, RMSSD falls." |
| 1:30 | Stress index + Why this score? | "The stress index shows exactly what pushed it up." |
| 1:45 | Learn page | "There's also a plain-language section for the public." |
| 1:55 | GitHub | "It's open source. Link below. Thanks for watching!" |

Upload to YouTube (unlisted is fine) and add the link to the README and LinkedIn.

---

## 9. CV and SOP lines (fill in ONLY what actually happened)

**CV**
- *BioXAI (open-source explainable AI web app)*, Creator and team lead, <month year>. Integrated my
  dissertation and research paper into a public tool for breast cancer stage analysis and HRV stress
  assessment; led a team of <n> students; delivered a workshop to <n> participants
  (avg. usefulness <x>/5); <ran an HRV stress-check pilot with <n> students>. <app link>

**SOP paragraph (template)**
> My undergraduate work covered two very different scales of biology: gene expression in breast
> cancer and heart rate variability under academic stress. While analysing our breast cancer models,
> I realised that high accuracy can hide a model that simply predicts the majority class, and that
> explanations matter as much as predictions. To act on this, I built BioXAI, an open-source app that
> applies explainable machine learning to both kinds of data. I recruited and guided <n> junior
> students, ran a workshop for <n> peers, and used their feedback to improve the tool. <If done: With
> my supervisor's approval, we used it in a pilot comparing students' HRV on normal and exam days,
> completing the comparison my dissertation could not.> This experience convinced me to pursue
> doctoral research in <computational biology / biomedical data science>, where I want to build
> models that are both accurate and interpretable.
