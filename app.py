# =============================
# TALENT REDISCOVERY ENGINE (PRODUCT VERSION) - CLEAN & FIXED
# =============================

import streamlit as st
import pandas as pd
import re
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
    if pd.isna(x):
        return ""
    return str(x).lower().strip()

def tokenize_candidate_skills(text):
    """Split candidate skills by commas, slashes, newlines, or pipes. Keep multi-word phrases."""
    text = clean_text(text)
    tokens = re.split(r"[,/;\n|\|]+", text)
    return [t.strip() for t in tokens if t.strip()]

def extract_years(text):
    text = clean_text(text)
    m = re.search(r"(\d+(?:\.\d+)?)", text)  # matches 4.5, 3, etc.
    return float(m.group(1)) if m else None

# =============================
# JD PARSING (MULTI-WORD + "OR" HANDLING)
# =============================
def parse_jd(jd_text):
    """
    Extracts must-have skills from a job description.
    Looks for lines after "Must have:" (case-insensitive) that start with '-', '*', '•', or numbers.
    Splits " or " into separate skills.
    """
    must_have = []
    lines = jd_text.split("\n")
    capture = False

    for line in lines:
        line_lower = line.lower()
        # Start capturing after "must have" (with optional colon)
        if "must have" in line_lower:
            capture = True
            # Also extract from the same line after colon if present
            if ":" in line:
                after_colon = line.split(":", 1)[1]
                must_have.extend(_extract_skills_from_text(after_colon))
            continue

        if capture:
            stripped = line.strip()
            # Stop capturing when we hit an empty line or a new section heading
            if not stripped or any(h in line_lower for h in ["good to have", "preferred", "nice to have", "qualifications"]):
                capture = False
                continue
            # Capture bullet points or numbered items
            if stripped.startswith(("-", "*", "•")) or re.match(r"^\d+\.", stripped):
                # Remove bullet/number prefix
                content = re.sub(r"^[\-\*\•\d\.]+\s*", "", stripped)
                must_have.extend(_extract_skills_from_text(content))

    # Remove duplicates while preserving order
    unique_skills = []
    for s in must_have:
        if s not in unique_skills:
            unique_skills.append(s)
    return {
        "must_have": unique_skills,
        "text": clean_text(jd_text)
    }

def _extract_skills_from_text(text):
    """
    Split text by commas or "or", clean each part, and filter out very short words.
    Example: "NLP or LLM applications, Python" -> ["NLP", "LLM applications", "Python"]
    """
    # Replace " or " with a comma to split uniformly
    text = re.sub(r"\s+or\s+", ",", text)
    parts = [p.strip() for p in text.split(",") if p.strip()]
    skills = []
    for p in parts:
        # Remove trailing/leading punctuation and extra spaces
        p = re.sub(r"^[^\w]+|[^\w]+$", "", p)
        if len(p) > 1 and p not in {"and", "with", "the", "a", "an", "of", "for"}:
            skills.append(p)
    return skills

# =============================
# HARD FILTER (USING PHRASE MATCHING)
# =============================
def passes_filter(row, jd_struct, threshold=0.3):
    candidate_skills = tokenize_candidate_skills(row.get("Skills", ""))
    must_have = jd_struct["must_have"]

    if not must_have:
        return True, 0, 0, [], candidate_skills

    matched = []
    for must_skill in must_have:
        # Substring match (case-insensitive because everything is lowercased)
        if any(must_skill in cand_skill for cand_skill in candidate_skills):
            matched.append(must_skill)

    match_count = len(matched)
    required = max(1, int(threshold * len(must_have)))
    passed = match_count >= required
    return passed, match_count, required, matched, candidate_skills

# =============================
# SCORING (TF-IDF + SKILL OVERLAP + EXPERIENCE)
# =============================
def compute_scores(df, jd_struct):
    df = df.reset_index(drop=True)
    corpus = df["combined"].tolist() + [jd_struct["text"]]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1]).flatten()

    results = []
    for i, row in df.iterrows():
        candidate_skills = tokenize_candidate_skills(row["Skills"])
        must_skills = jd_struct["must_have"]

        skill_match = sum(1 for ms in must_skills if any(ms in cs for cs in candidate_skills))
        skill_score = skill_match / max(len(must_skills), 1)

        semantic_score = similarities[i]
        exp_years = extract_years(row["Experience"])
        exp_score = min(1.0, exp_years / 5) if exp_years else 0.5

        final_score = (0.35 * skill_score + 0.25 * semantic_score + 0.15 * exp_score) * 100
        results.append((final_score, skill_score, semantic_score, exp_score))

    df["score"], df["skill"], df["semantic"], df["exp"] = zip(*results)
    return df

def assign_priority(score, skill_score):
    if score >= 75 and skill_score >= 0.5:
        return "High"
    if score >= 55:
        return "Medium"
    return "Low"

def generate_message(name, role, matched_skills):
    top = ", ".join(matched_skills[:3]) if matched_skills else "your experience"
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

    uploaded_file = st.file_uploader("Upload CSV (with a column containing 'skill')", type=["csv"])
    jd_text = st.text_area("Paste Job Description", height=200)

    if not uploaded_file or not jd_text:
        st.info("📂 Please upload a CSV file and paste a job description.")
        return

    # Auto-detect skills column
    df = pd.read_csv(uploaded_file)
    skills_col = next((col for col in df.columns if "skill" in col.lower()), None)
    if not skills_col:
        st.error("❌ No column with 'skill' in its name found. Rename your skills column (e.g., 'Skills', 'Technical Skills').")
        return
    df.rename(columns={skills_col: "Skills"}, inplace=True)
    st.success(f"✅ Using column '{skills_col}' as 'Skills'")

    with st.expander("Preview CSV data (first 5 rows)"):
        st.dataframe(df.head(5))

    # Parse JD and show extracted skills
    jd_struct = parse_jd(jd_text)
    must_have = jd_struct["must_have"]
    st.subheader("📌 Extracted Must‑Have Skills")
    if must_have:
        st.write(must_have)
    else:
        st.warning("No must‑have skills detected. Check that your JD includes a 'Must have:' section with bullet points or commas.")

    # Hard filter threshold
    threshold_pct = st.slider(
        "Minimum % of must‑have skills required to pass hard filter",
        min_value=0, max_value=100, value=30, step=5
    ) / 100.0
    if must_have:
        required_min = max(1, int(threshold_pct * len(must_have)))
        st.caption(f"🔍 Requires at least **{required_min}** out of {len(must_have)} skill(s).")

    if st.button("🚀 Rank Candidates", type="primary"):
        # Prepare combined text for semantic scoring
        df["combined"] = df.astype(str).agg(" ".join, axis=1)

        # Apply hard filter
        filter_results = df.apply(lambda row: passes_filter(row, jd_struct, threshold_pct)[0], axis=1)
        df_filtered = df[filter_results].copy()

        if df_filtered.empty:
            st.warning("❌ No candidates passed the hard filter. Try lowering the threshold or check skill spelling.")
            st.stop()

        # Compute scores and rank
        df_filtered = compute_scores(df_filtered, jd_struct)
        df_filtered["priority"] = df_filtered.apply(lambda x: assign_priority(x["score"], x["skill"]), axis=1)

        # Show matched/missing skills
        matched_missing = df_filtered.apply(
            lambda row: (
                [m for m in must_have if any(m in s for s in tokenize_candidate_skills(row["Skills"]))],
                [m for m in must_have if not any(m in s for s in tokenize_candidate_skills(row["Skills"]))]
            ), axis=1
        )
        df_filtered["matched"], df_filtered["missing"] = zip(*matched_missing)
        df_filtered = df_filtered.sort_values("score", ascending=False)

        st.subheader("✅ Ranked Candidates")
        for _, row in df_filtered.head(10).iterrows():
            name = row.get("Name", "Unknown")
            title = row.get("Current / last job title", "N/A")
            with st.container():
                st.markdown(f"### {name} - {title}")
                st.write(f"**Score:** {round(row['score'], 1)} | **Priority:** {row['priority']}")
                st.write(f"**Matched skills:** {', '.join(row['matched'])}")
                st.write(f"**Missing skills:** {', '.join(row['missing'])}")
                wa, em = generate_message(name, "AI Role", row['matched'])
                with st.expander("📨 Outreach Templates"):
                    st.text("WhatsApp:")
                    st.write(wa)
                    st.text("Email:")
                    st.code(em)

        st.download_button("📥 Download Shortlist CSV", df_filtered.to_csv(index=False), "shortlist.csv")

if __name__ == "__main__":
    main()