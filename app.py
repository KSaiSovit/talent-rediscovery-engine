# # =============================
# # TALENT REDISCOVERY ENGINE (PRODUCT VERSION) - DEBUGGING FIX
# # =============================

# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# from typing import List, Dict

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # =============================
# # CONFIG
# # =============================
# APP_TITLE = "Talent Rediscovery + Re-Engagement (Product Version)"

# # =============================
# # UTILS
# # =============================
# def clean_text(x):
#     if pd.isna(x): return ""
#     return str(x).lower().strip()

# def tokenize(text):
#     """Split by commas, slashes, newlines, or pipes, then strip."""
#     text = clean_text(text)
#     # Also split by semicolon and space if needed? Adding ; and space for safety
#     tokens = re.split(r"[,/;\n|\| ]+", text)
#     return [t.strip() for t in tokens if t.strip()]

# def extract_years(text):
#     text = clean_text(text)
#     m = re.search(r"(\d+)", text)
#     return float(m.group(1)) if m else None

# # =============================
# # JD PARSING (LLM READY)
# # =============================
# def parse_jd(jd):
#     jd = clean_text(jd)
#     must_have = []
#     good_to_have = []

#     for line in jd.split("\n"):
#         if "must" in line.lower():
#             must_have += tokenize(line)
#         elif "good" in line.lower():
#             good_to_have += tokenize(line)

#     return {
#         "must_have": list(set(must_have)),
#         "good_to_have": list(set(good_to_have)),
#         "text": jd
#     }

# # =============================
# # HARD FILTER (with debug)
# # =============================
# def passes_filter(row, jd_struct, threshold=0.3):
#     skills_raw = row.get("Skills", "")
#     skills = tokenize(skills_raw)
#     must = jd_struct["must_have"]
    
#     if not must:
#         return True
    
#     # For each must-have skill, check if it appears as a substring in any candidate skill
#     match_count = 0
#     matched_skills = []
#     for m in must:
#         found = any(m in s for s in skills)
#         if found:
#             match_count += 1
#             matched_skills.append(m)
    
#     required = max(1, int(threshold * len(must)))
#     passed = match_count >= required
    
#     # Store debug info in row (we'll return it later, but we need to pass back)
#     # Instead, we'll compute debug in main loop
#     return passed, match_count, required, matched_skills, skills

# # =============================
# # SCORING
# # =============================
# def compute_scores(df, jd_struct):
#     corpus = df["combined"].tolist() + [jd_struct["text"]]
#     vec = TfidfVectorizer(stop_words="english")
#     mat = vec.fit_transform(corpus)

#     sims = cosine_similarity(mat[:-1], mat[-1]).flatten()

#     results = []

#     for i, row in df.iterrows():
#         skills = tokenize(row["Skills"])
#         must = jd_struct["must_have"]

#         skill_match = len([m for m in must if any(m in s for s in skills)])
#         skill_score = skill_match / max(len(must), 1)

#         semantic_score = sims[i]
#         exp_score = min(1.0, extract_years(row["Experience"]) / 5 if extract_years(row["Experience"]) else 0.5)

#         final_score = (
#             0.35 * skill_score +
#             0.25 * semantic_score +
#             0.15 * exp_score
#         ) * 100

#         results.append((final_score, skill_score, semantic_score, exp_score))

#     df["score"], df["skill"], df["semantic"], df["exp"] = zip(*results)
#     return df

# def assign_priority(score, skill_score):
#     if score >= 75 and skill_score >= 0.5:
#         return "High"
#     elif score >= 55:
#         return "Medium"
#     return "Low"

# def explain(row, jd_struct):
#     skills = tokenize(row["Skills"])
#     must = jd_struct["must_have"]
#     matched = [m for m in must if any(m in s for s in skills)]
#     missing = [m for m in must if m not in matched]
#     return matched, missing

# def generate_message(name, role, skills):
#     top = ", ".join(skills[:3]) if skills else "your experience"
#     whatsapp = f"Hi {name}, your experience in {top} looks relevant for a {role} role. Open to explore?"
#     email = f"""Subject: Opportunity - {role}

# Hi {name},

# Your background in {top} aligns well with our role.

# Would you be open to a quick chat?

# Best,
# Recruiter"""
#     return whatsapp, email

# # =============================
# # MAIN APP
# # =============================
# def main():
#     st.set_page_config(layout="wide")
#     st.title(APP_TITLE)

#     uploaded = st.file_uploader("Upload CSV", type=["csv"])
#     jd = st.text_area("Paste Job Description")

#     if uploaded and jd:
#         df = pd.read_csv(uploaded)
        
#         # ----- AUTO-DETECT SKILLS COLUMN -----
#         skills_col = None
#         for col in df.columns:
#             if 'skill' in col.lower():
#                 skills_col = col
#                 break
#         if skills_col is None:
#             st.error("❌ No column with 'skill' in its name found. Please rename your skills column (e.g., 'Skills', 'Technical Skills').")
#             st.stop()
#         # Rename to standard 'Skills'
#         df.rename(columns={skills_col: 'Skills'}, inplace=True)
#         st.success(f"✅ Using column '{skills_col}' as 'Skills'")
        
#         # ----- PREVIEW DATA -----
#         with st.expander("Preview CSV data (first 3 rows)"):
#             st.dataframe(df.head(3))
        
#         # Create combined text for scoring
#         df["combined"] = df.astype(str).agg(" ".join, axis=1)
        
#         # Parse JD
#         jd_struct = parse_jd(jd)
#         must_have = jd_struct["must_have"]
#         st.subheader("📌 Extracted Must‑Have Skills")
#         st.write(must_have if must_have else "None detected. (Make sure your JD contains lines with 'Must have:' or 'Must:' followed by skills.)")
        
#         # ----- THRESHOLD SLIDER -----
#         threshold_pct = st.slider(
#             "Minimum % of must‑have skills required to pass hard filter",
#             min_value=0, max_value=100, value=30, step=5
#         ) / 100.0
        
#         if must_have:
#             required_min = max(1, int(threshold_pct * len(must_have)))
#             st.caption(f"🔍 Requires at least **{required_min}** out of {len(must_have)} must‑have skill(s).")
#         else:
#             st.caption("🔍 No must‑have skills – all candidates will pass.")
#             # If no must-have, we can skip filtering
#             df_filtered = df
#         # ----- END SLIDER -----
        
#         # ----- DEBUG: Show matching details for each candidate -----
#         debug_data = []
#         for idx, row in df.iterrows():
#             passed, match_cnt, req, matched_skills, tokenized_skills = passes_filter(row, jd_struct, threshold_pct)
#             debug_data.append({
#                 "Name": row.get("Name", f"Row {idx}"),
#                 "Raw Skills": row["Skills"],
#                 "Tokenized Skills": tokenized_skills,
#                 "Matched Must-Have": matched_skills,
#                 "Match Count": match_cnt,
#                 "Required": req,
#                 "Passed": passed
#             })
        
#         debug_df = pd.DataFrame(debug_data)
#         with st.expander("🔍 Debug: Hard Filter Evaluation (first 10 rows)"):
#             st.dataframe(debug_df.head(10))
        
#         # Apply the filter
#         # We need to re-run passes_filter because it returns tuple; we'll use apply with result
#         filter_results = df.apply(lambda row: passes_filter(row, jd_struct, threshold_pct)[0], axis=1)
#         df_filtered = df[filter_results].copy()
        
#         if df_filtered.empty:
#             st.warning("❌ No candidates passed the hard filter. Check the debug table above to see why.")
#             st.info("💡 Tips:\n"
#                     "- Lower the threshold (try 0% to see everyone).\n"
#                     "- Make sure your JD's must-have skills exactly match words in the Skills column (substrings work).\n"
#                     "- Ensure the Skills column contains comma-separated values like 'Python, SQL'.\n"
#                     "- If your JD says 'Machine Learning', but candidate has 'ML', that won't match – adjust JD.")
#             st.stop()
        
#         # Compute scores on filtered candidates
#         df_filtered = compute_scores(df_filtered, jd_struct)
#         df_filtered["priority"] = df_filtered.apply(lambda x: assign_priority(x["score"], x["skill"]), axis=1)
#         explanations = df_filtered.apply(lambda x: explain(x, jd_struct), axis=1)
#         df_filtered["matched"], df_filtered["missing"] = zip(*explanations)
#         df_filtered = df_filtered.sort_values(by="score", ascending=False)
        
#         st.subheader("✅ Ranked Candidates")
#         for _, row in df_filtered.head(10).iterrows():
#             with st.container():
#                 st.markdown(f"### {row.get('Name', 'Unknown')} - {row.get('Current / last job title', 'N/A')}")
#                 st.write(f"Score: {round(row['score'],1)} | Priority: {row['priority']}")
#                 st.write(f"Matched: {', '.join(row['matched'])}")
#                 st.write(f"Missing: {', '.join(row['missing'])}")
#                 wa, em = generate_message(row.get('Name', 'Candidate'), "AI Role", row['matched'])
#                 with st.expander("Outreach"):
#                     st.write("WhatsApp:")
#                     st.write(wa)
#                     st.write("Email:")
#                     st.code(em)
        
#         st.download_button("Download CSV", df_filtered.to_csv(index=False), "shortlist.csv")

# if __name__ == "__main__":
#     main()

# =============================
# TALENT REDISCOVERY ENGINE (PRODUCT VERSION) - FULLY FIXED
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

# Common stopwords to ignore when parsing JD skills
STOPWORDS = {"must", "have", "has", "skills", "and", "or", "with", "including", "like", "know", "knowledge", "experience", "proficient", "familiar"}

# =============================
# UTILS
# =============================
def clean_text(x):
    if pd.isna(x): return ""
    return str(x).lower().strip()

def tokenize(text, split_on_spaces=False):
    """
    Split text by commas, slashes, newlines, or pipes.
    If split_on_spaces is True, also split on spaces (use for JD lines only).
    """
    text = clean_text(text)
    if split_on_spaces:
        # For JD line parsing: split on spaces as well
        tokens = re.split(r"[,/;\n|\| ]+", text)
    else:
        # For candidate skills: keep spaces (e.g., "machine learning" stays together)
        tokens = re.split(r"[,/;\n|\|]+", text)
    return [t.strip() for t in tokens if t.strip()]

def extract_years(text):
    text = clean_text(text)
    m = re.search(r"(\d+)", text)
    return float(m.group(1)) if m else None

# =============================
# JD PARSING (IMPROVED)
# =============================
def parse_jd(jd):
    jd = clean_text(jd)
    must_have = []
    good_to_have = []

    lines = jd.split("\n")
    for line in lines:
        line_lower = line.lower()
        # Determine if line is about must-have or good-to-have
        if "must" in line_lower or "required" in line_lower:
            # Extract content after colon or bullet point
            content = line
            for separator in [":", "•", "-", "*"]:
                if separator in line:
                    content = line.split(separator, 1)[1]
                    break
            # Tokenize (split on spaces here to get individual skill words/phrases)
            skill_tokens = tokenize(content, split_on_spaces=True)
            # Filter out stopwords and very short tokens
            for tok in skill_tokens:
                if len(tok) > 1 and tok not in STOPWORDS:
                    must_have.append(tok)
        elif "good" in line_lower or "plus" in line_lower:
            content = line
            for separator in [":", "•", "-", "*"]:
                if separator in line:
                    content = line.split(separator, 1)[1]
                    break
            skill_tokens = tokenize(content, split_on_spaces=True)
            for tok in skill_tokens:
                if len(tok) > 1 and tok not in STOPWORDS:
                    good_to_have.append(tok)

    return {
        "must_have": list(set(must_have)),
        "good_to_have": list(set(good_to_have)),
        "text": jd
    }

# =============================
# HARD FILTER
# =============================
def passes_filter(row, jd_struct, threshold=0.3):
    skills_raw = row.get("Skills", "")
    skills = tokenize(skills_raw, split_on_spaces=False)   # keep multi-word skills
    must = jd_struct["must_have"]
    
    if not must:
        return True, 0, 0, [], skills
    
    match_count = 0
    matched_skills = []
    for m in must:
        # Check if must-have skill appears as a substring in any candidate skill
        if any(m in s for s in skills):
            match_count += 1
            matched_skills.append(m)
    
    required = max(1, int(threshold * len(must)))
    passed = match_count >= required
    return passed, match_count, required, matched_skills, skills

# =============================
# SCORING (with index reset)
# =============================
def compute_scores(df, jd_struct):
    # Reset index to avoid misalignment with sims array
    df = df.reset_index(drop=True)
    corpus = df["combined"].tolist() + [jd_struct["text"]]
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform(corpus)

    sims = cosine_similarity(mat[:-1], mat[-1]).flatten()

    results = []
    for i, row in df.iterrows():
        skills = tokenize(row["Skills"], split_on_spaces=False)
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

def assign_priority(score, skill_score):
    if score >= 75 and skill_score >= 0.5:
        return "High"
    elif score >= 55:
        return "Medium"
    return "Low"

def explain(row, jd_struct):
    skills = tokenize(row["Skills"], split_on_spaces=False)
    must = jd_struct["must_have"]
    matched = [m for m in must if any(m in s for s in skills)]
    missing = [m for m in must if m not in matched]
    return matched, missing

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
    jd = st.text_area("Paste Job Description", height=200)
    
    if uploaded and jd:
        threshold_pct = st.slider(
            "Minimum % of must‑have skills required to pass hard filter",
            min_value=0, max_value=100, value=30, step=5
        ) / 100.0
    else:
        threshold_pct = 0.3

    run_button = st.button("🚀 Rank Candidates", type="primary", disabled=not (uploaded and jd))

    if run_button:
        if not uploaded or not jd:
            st.error("Please upload a CSV file and paste a job description.")
            st.stop()
        
        df = pd.read_csv(uploaded)
        
        # Auto-detect skills column
        skills_col = None
        for col in df.columns:
            if 'skill' in col.lower():
                skills_col = col
                break
        if skills_col is None:
            st.error("❌ No column with 'skill' in its name found. Please rename your skills column (e.g., 'Skills', 'Technical Skills').")
            st.stop()
        df.rename(columns={skills_col: 'Skills'}, inplace=True)
        st.success(f"✅ Using column '{skills_col}' as 'Skills'")
        
        with st.expander("Preview CSV data (first 3 rows)"):
            st.dataframe(df.head(3))
        
        df["combined"] = df.astype(str).agg(" ".join, axis=1)
        jd_struct = parse_jd(jd)
        must_have = jd_struct["must_have"]
        
        st.subheader("📌 Extracted Must‑Have Skills")
        if must_have:
            st.write(must_have)
        else:
            st.warning("No must‑have skills detected. Make sure your JD includes lines like: 'Must have: Python, SQL' or bullet points.")
        
        if must_have:
            required_min = max(1, int(threshold_pct * len(must_have)))
            st.caption(f"🔍 Requires at least **{required_min}** out of {len(must_have)} must‑have skill(s).")
        else:
            st.caption("🔍 No must‑have skills – all candidates will pass.")
        
        # Debug evaluation
        debug_data = []
        for idx, row in df.iterrows():
            passed, match_cnt, req, matched_skills, tokenized_skills = passes_filter(row, jd_struct, threshold_pct)
            debug_data.append({
                "Name": row.get("Name", f"Row {idx}"),
                "Raw Skills": row["Skills"],
                "Tokenized Skills": tokenized_skills,
                "Matched Must-Have": matched_skills,
                "Match Count": match_cnt,
                "Required": req,
                "Passed": passed
            })
        debug_df = pd.DataFrame(debug_data)
        with st.expander("🔍 Debug: Hard Filter Evaluation (first 10 rows)"):
            st.dataframe(debug_df.head(10))
        
        # Apply filter
        filter_results = df.apply(lambda row: passes_filter(row, jd_struct, threshold_pct)[0], axis=1)
        df_filtered = df[filter_results].copy()
        
        if df_filtered.empty:
            st.warning("❌ No candidates passed the hard filter. Check the debug table above.")
            st.info("💡 Tips:\n- Lower the threshold (try 0%)\n- Make sure must-have skills are spelled similarly in CSV and JD\n- Ensure skills are comma-separated in CSV")
            st.stop()
        
        df_filtered = compute_scores(df_filtered, jd_struct)
        df_filtered["priority"] = df_filtered.apply(lambda x: assign_priority(x["score"], x["skill"]), axis=1)
        explanations = df_filtered.apply(lambda x: explain(x, jd_struct), axis=1)
        df_filtered["matched"], df_filtered["missing"] = zip(*explanations)
        df_filtered = df_filtered.sort_values(by="score", ascending=False)
        
        st.subheader("✅ Ranked Candidates")
        for _, row in df_filtered.head(10).iterrows():
            with st.container():
                st.markdown(f"### {row.get('Name', 'Unknown')} - {row.get('Current / last job title', 'N/A')}")
                st.write(f"Score: {round(row['score'],1)} | Priority: {row['priority']}")
                st.write(f"Matched: {', '.join(row['matched'])}")
                st.write(f"Missing: {', '.join(row['missing'])}")
                wa, em = generate_message(row.get('Name', 'Candidate'), "AI Role", row['matched'])
                with st.expander("Outreach"):
                    st.write("WhatsApp:")
                    st.write(wa)
                    st.write("Email:")
                    st.code(em)
        
        st.download_button("Download CSV", df_filtered.to_csv(index=False), "shortlist.csv")

if __name__ == "__main__":
    main()