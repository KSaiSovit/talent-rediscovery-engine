# =============================
# TALENT REDISCOVERY ENGINE
# =============================

import streamlit as st
import pandas as pd
import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_TITLE = "Talent Rediscovery + Re-Engagement (Product Version)"

# ------------------------------
# UTILS
# ------------------------------
def clean_text(x):
    return "" if pd.isna(x) else str(x).lower().strip()

def tokenize_candidate_skills(text):
    """Keep multi‑word phrases (split only by commas, slashes, newlines, pipes)."""
    text = clean_text(text)
    tokens = re.split(r"[,/;\n|\|]+", text)
    return [t.strip() for t in tokens if t.strip()]

def extract_years(text):
    m = re.search(r"(\d+(?:\.\d+)?)", clean_text(text))
    return float(m.group(1)) if m else None

# ------------------------------
# JD PARSING (PRESERVES SLASHES & MULTI‑WORD PHRASES)
# ------------------------------
def parse_jd(jd_text):
    must_have = []
    capture = False

    for line in jd_text.split("\n"):
        line_lower = line.lower()
        if "must have" in line_lower:
            capture = True
            if ":" in line:
                after_colon = line.split(":", 1)[1]
                must_have.extend(_extract_skill_phrases(after_colon))
            continue

        if capture:
            stripped = line.strip()
            if not stripped or any(h in line_lower for h in ["good to have", "preferred", "nice to have"]):
                capture = False
                continue
            if stripped.startswith(("-", "*", "•")) or re.match(r"^\d+\.", stripped):
                content = re.sub(r"^[\-\*\•\d\.]+\s*", "", stripped)
                must_have.extend(_extract_skill_phrases(content))

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for s in must_have:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return {"must_have": unique, "text": clean_text(jd_text)}

def _extract_skill_phrases(text):
    """
    Split by commas or "or", but keep slashes and spaces inside a phrase.
    Example: "RAG or search/retrieval experience" -> ["rag", "search/retrieval experience"]
    """
    # Replace " or " with a comma
    text = re.sub(r"\s+or\s+", ",", text)
    parts = [p.strip() for p in text.split(",") if p.strip()]
    skills = []
    for p in parts:
        # Remove only leading/trailing punctuation (keep internal slashes, spaces)
        p = re.sub(r"^[^\w\s/]+|[^\w\s/]+$", "", p)
        if len(p) > 1 and p not in {"and", "with", "the", "a", "an", "of", "for"}:
            skills.append(p)
    return skills

# ------------------------------
# HARD FILTER (SUBPHRASE MATCHING)
# ------------------------------
def passes_filter(row, jd_struct, threshold=0.3):
    candidate_skills = tokenize_candidate_skills(row["Skills"])
    must_have = jd_struct["must_have"]
    if not must_have:
        return True, 0, 0, [], candidate_skills

    matched = []
    for must_skill in must_have:
        # Substring match (case‑insensitive already due to lowercasing)
        if any(must_skill in cand for cand in candidate_skills):
            matched.append(must_skill)

    match_count = len(matched)
    required = max(1, math.ceil(threshold * len(must_have)))
    passed = match_count >= required
    return passed, match_count, required, matched, candidate_skills

# ------------------------------
# SCORING
# ------------------------------
def compute_scores(df, jd_struct):
    df = df.reset_index(drop=True)
    corpus = df["combined"].tolist() + [jd_struct["text"]]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(tfidf[:-1], tfidf[-1]).flatten()

    results = []
    for i, row in df.iterrows():
        cand_skills = tokenize_candidate_skills(row["Skills"])
        must = jd_struct["must_have"]
        skill_match = sum(1 for ms in must if any(ms in cs for cs in cand_skills))
        skill_score = skill_match / max(len(must), 1)

        semantic = sims[i]
        exp_years = extract_years(row["Experience"])
        exp_score = min(1.0, exp_years / 5) if exp_years else 0.5

        final = (0.35 * skill_score + 0.25 * semantic + 0.15 * exp_score) * 100
        results.append((final, skill_score, semantic, exp_score))

    df["score"], df["skill"], df["semantic"], df["exp"] = zip(*results)
    return df

def assign_priority(score, skill_score):
    if score >= 75 and skill_score >= 0.5:
        return "High"
    if score >= 55:
        return "Medium"
    return "Low"

def generate_message(name, role, matched):
    top = ", ".join(matched[:3]) if matched else "your experience"
    wa = f"Hi {name}, your experience in {top} looks relevant for a {role} role. Open to explore?"
    email = f"""Subject: Opportunity - {role}

Hi {name},

Your background in {top} aligns well with our role.

Would you be open to a quick chat?

Best,
Recruiter"""
    return wa, email

# ------------------------------
# MAIN APP
# ------------------------------
def main():
    st.set_page_config(layout="wide")
    st.title(APP_TITLE)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    jd = st.text_area("Paste Job Description", height=200)

    if not uploaded or not jd:
        st.info("📂 Upload a CSV and paste a job description.")
        return

    # Auto‑detect skills column
    df = pd.read_csv(uploaded)
    skills_col = next((c for c in df.columns if "skill" in c.lower()), None)
    if not skills_col:
        st.error("❌ No column with 'skill' in its name. Rename it to 'Skills'.")
        return
    df.rename(columns={skills_col: "Skills"}, inplace=True)
    st.success(f"✅ Using column: {skills_col}")

    with st.expander("Preview CSV (first 5 rows)"):
        st.dataframe(df.head(5))

    # Parse JD and show extracted must‑haves
    jd_struct = parse_jd(jd)
    must_have = jd_struct["must_have"]
    st.subheader("📌 Extracted Must‑Have Skills")
    if must_have:
        st.write(must_have)
    else:
        st.warning("No must‑have skills found. Check that your JD has a 'Must have:' section with bullet points.")

    # Threshold slider
    threshold_pct = st.slider("Minimum % of must‑have skills required", 0, 100, 30, 5) / 100.0
    if must_have:
        required_min = max(1, math.ceil(threshold_pct * len(must_have)))
        st.caption(f"🔍 Requires at least **{required_min}** out of {len(must_have)} skill(s).")

    if st.button("🚀 Rank Candidates", type="primary"):
        df["combined"] = df.astype(str).agg(" ".join, axis=1)

        # ---- DEBUG: Show exactly why each candidate passes/fails ----
        debug_rows = []
        for idx, row in df.iterrows():
            passed, match_cnt, req, matched, cand_skills = passes_filter(row, jd_struct, threshold_pct)
            debug_rows.append({
                "Name": row.get("Name", f"Row {idx}"),
                "Candidate Skills (tokenized)": cand_skills,
                "Must‑Have Skills": must_have,
                "Matched Must‑Haves": matched,
                "Match Count": match_cnt,
                "Required": req,
                "Passes Filter": passed
            })
        debug_df = pd.DataFrame(debug_rows)
        with st.expander("🔍 Debug: Hard Filter Evaluation (full list)"):
            st.dataframe(debug_df)

        # Apply filter
        filter_mask = df.apply(lambda row: passes_filter(row, jd_struct, threshold_pct)[0], axis=1)
        df_filtered = df[filter_mask].copy()

        if df_filtered.empty:
            st.error("❌ No candidates passed the hard filter.")
            st.info("💡 Check the debug table above. Common reasons:\n"
                    "- Must‑have skills like 'search/retrieval experience' don't appear in any candidate\n"
                    "- Typos or different wording (e.g., 'LLM' vs 'LLMs')\n"
                    "- Try lowering the threshold to 0% to see everyone")
            st.stop()

        # Score and rank
        df_filtered = compute_scores(df_filtered, jd_struct)
        df_filtered["priority"] = df_filtered.apply(lambda x: assign_priority(x["score"], x["skill"]), axis=1)

        # Show matched/missing for final output
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
                st.write(f"**Score:** {round(row['score'],1)} | **Priority:** {row['priority']}")
                st.write(f"**Matched:** {', '.join(row['matched'])}")
                st.write(f"**Missing:** {', '.join(row['missing'])}")
                wa, em = generate_message(name, "AI Role", row['matched'])
                with st.expander("📨 Outreach"):
                    st.text("WhatsApp:")
                    st.write(wa)
                    st.text("Email:")
                    st.code(em)

        st.download_button("📥 Download Shortlist", df_filtered.to_csv(index=False), "shortlist.csv")

if __name__ == "__main__":
    main()