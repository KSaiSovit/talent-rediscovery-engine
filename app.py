# =============================
# TALENT REDISCOVERY ENGINE (PRODUCT VERSION)
# =============================

import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# CONFIG
# =============================
APP_TITLE = "Talent Rediscovery + Re-Engagement (Product Version)"

# =============================
# UTILS
# =============================
def clean_text(x):
    if pd.isna(x): return ""
    return str(x).lower().strip()


def tokenize(text):
    text = clean_text(text)
    tokens = re.split(r",|/|\n|\|", text)
    return [t.strip() for t in tokens if t.strip()]


def extract_years(text):
    text = clean_text(text)
    m = re.search(r"(\d+)", text)
    return float(m.group(1)) if m else None

# =============================
# JD PARSING (LLM READY)
# =============================
def parse_jd(jd):
    jd = clean_text(jd)

    must_have = []
    good_to_have = []

    for line in jd.split("\n"):
        if "must" in line:
            must_have += tokenize(line)
        elif "good" in line:
            good_to_have += tokenize(line)

    return {
        "must_have": list(set(must_have)),
        "good_to_have": list(set(good_to_have)),
        "text": jd
    }

# =============================
# HARD FILTER
# =============================
def passes_filter(row, jd_struct):
    skills = tokenize(row.get("Skills", ""))
    must = jd_struct["must_have"]

    match = sum([1 for m in must if any(m in s for s in skills)])

    return match >= max(1, int(0.3 * len(must)))

# =============================
# SCORING
# =============================
def compute_scores(df, jd_struct):
    corpus = df["combined"].tolist() + [jd_struct["text"]]
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform(corpus)

    sims = cosine_similarity(mat[:-1], mat[-1]).flatten()

    results = []

    for i, row in df.iterrows():
        skills = tokenize(row["Skills"])
        must = jd_struct["must_have"]

        skill_match = len([m for m in must if any(m in s for s in skills)])
        skill_score = skill_match / max(len(must), 1)

        semantic_score = sims[i]
        exp_score = min(1.0, extract_years(row["Experience"]) / 5 if extract_years(row["Experience"]) else 0.5)

        final_score = (
            0.35 * skill_score +
            0.25 * semantic_score +
            0.15 * exp_score
        ) * 100

        results.append((final_score, skill_score, semantic_score, exp_score))

    df["score"], df["skill"], df["semantic"], df["exp"] = zip(*results)
    return df

# =============================
# PRIORITY
# =============================
def assign_priority(score, skill_score):
    if score >= 75 and skill_score >= 0.5:
        return "High"
    elif score >= 55:
        return "Medium"
    return "Low"

# =============================
# EXPLANATIONS
# =============================
def explain(row, jd_struct):
    skills = tokenize(row["Skills"])
    must = jd_struct["must_have"]

    matched = [m for m in must if any(m in s for s in skills)]
    missing = [m for m in must if m not in matched]

    return matched, missing

# =============================
# MESSAGES (LLM READY)
# =============================
def generate_message(name, role, skills):
    top = ", ".join(skills[:3]) if skills else "your experience"

    whatsapp = f"Hi {name}, your experience in {top} looks relevant for a {role} role. Open to explore?"

    email = f"""Subject: Opportunity - {role}

Hi {name},

Your background in {top} aligns well with our role.

Would you be open to a quick chat?

Best,
Recruiter"""

    return whatsapp, email

# =============================
# MAIN APP
# =============================
def main():
    st.set_page_config(layout="wide")
    st.title(APP_TITLE)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    jd = st.text_area("Paste Job Description")

    if uploaded and jd:
        df = pd.read_csv(uploaded)

        df["combined"] = df.astype(str).agg(" ".join, axis=1)

        jd_struct = parse_jd(jd)

        df = df[df.apply(lambda x: passes_filter(x, jd_struct), axis=1)]

        if df.empty:
            st.warning("No candidates passed the hard filter. Try relaxing the job description or check your CSV columns.")
            st.stop()

        df = compute_scores(df, jd_struct)

        df["priority"] = df.apply(lambda x: assign_priority(x["score"], x["skill"]), axis=1)

        explanations = df.apply(lambda x: explain(x, jd_struct), axis=1)
        df["matched"], df["missing"] = zip(*explanations)

        df = df.sort_values(by="score", ascending=False)

        st.subheader("Top Candidates")

        for _, row in df.head(10).iterrows():
            with st.container():
                st.markdown(f"### {row['Name']} - {row['Current / last job title']}")
                st.write(f"Score: {round(row['score'],1)} | Priority: {row['priority']}")

                st.write(f"Matched: {', '.join(row['matched'])}")
                st.write(f"Missing: {', '.join(row['missing'])}")

                wa, em = generate_message(row['Name'], "AI Role", row['matched'])

                with st.expander("Outreach"):
                    st.write("WhatsApp:")
                    st.write(wa)
                    st.write("Email:")
                    st.code(em)

        st.download_button("Download CSV", df.to_csv(index=False), "shortlist.csv")


if __name__ == "__main__":
    main()
