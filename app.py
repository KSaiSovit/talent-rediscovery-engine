# # =============================
# # TALENT REDISCOVERY ENGINE - ANY MATCH VERSION
# # =============================

# import streamlit as st
# import pandas as pd
# import re
# import math
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# APP_TITLE = "Talent Rediscovery + Re-Engagement (Any Match)"

# # ------------------------------
# # UTILS
# # ------------------------------
# def clean_text(x):
#     return "" if pd.isna(x) else str(x).lower().strip()

# def tokenize_candidate_skills(text):
#     """Split by commas, slashes, newlines, pipes. Keep multi-word phrases."""
#     text = clean_text(text)
#     tokens = re.split(r"[,/;\n|\|]+", text)
#     return [t.strip() for t in tokens if t.strip()]

# def extract_years(text):
#     m = re.search(r"(\d+(?:\.\d+)?)", clean_text(text))
#     return float(m.group(1)) if m else None

# # ------------------------------
# # JD PARSING - PRESERVES PHRASES, HANDLES "or"
# # ------------------------------
# def parse_jd(jd_text):
#     must_have = []
#     capture = False

#     for line in jd_text.split("\n"):
#         line_lower = line.lower()
#         if "must have" in line_lower:
#             capture = True
#             if ":" in line:
#                 after = line.split(":", 1)[1]
#                 must_have.extend(_extract_skill_phrases(after))
#             continue

#         if capture:
#             stripped = line.strip()
#             if not stripped or any(h in line_lower for h in ["good to have", "preferred", "nice to have"]):
#                 capture = False
#                 continue
#             if stripped.startswith(("-", "*", "•")) or re.match(r"^\d+\.", stripped):
#                 content = re.sub(r"^[\-\*\•\d\.]+\s*", "", stripped)
#                 must_have.extend(_extract_skill_phrases(content))

#     # Remove duplicates and stopwords
#     stopwords = {"a", "an", "and", "of", "for", "with", "the", "to", "in", "on", "at", "by"}
#     seen = set()
#     unique = []
#     for s in must_have:
#         if s not in seen and len(s) > 1 and s not in stopwords:
#             seen.add(s)
#             unique.append(s)
#     return {"must_have": unique, "text": clean_text(jd_text)}

# def _extract_skill_phrases(text):
#     """Split by commas or 'or', keep slashes and spaces inside a phrase."""
#     text = re.sub(r"\s+or\s+", ",", text)
#     parts = [p.strip() for p in text.split(",") if p.strip()]
#     skills = []
#     for p in parts:
#         p = p.strip(" .,;:!?()[]{}'\"")
#         if len(p) > 1:
#             skills.append(p)
#     return skills

# # ------------------------------
# # HARD FILTER - ANY MATCH (>=1)
# # ------------------------------
# def passes_filter(row, jd_struct):
#     """Returns True if candidate has at least ONE must-have skill (substring match)."""
#     candidate_skills = tokenize_candidate_skills(row["Skills"])
#     must_have = jd_struct["must_have"]
#     if not must_have:
#         return True, 0, [], candidate_skills

#     matched = []
#     for must_skill in must_have:
#         if any(must_skill in cand for cand in candidate_skills):
#             matched.append(must_skill)

#     match_count = len(matched)
#     passed = match_count >= 1                 # ANY match
#     return passed, match_count, matched, candidate_skills

# # ------------------------------
# # SCORING
# # ------------------------------
# def compute_scores(df, jd_struct):
#     df = df.reset_index(drop=True)
#     corpus = df["combined"].tolist() + [jd_struct["text"]]
#     vectorizer = TfidfVectorizer(stop_words="english")
#     tfidf = vectorizer.fit_transform(corpus)
#     sims = cosine_similarity(tfidf[:-1], tfidf[-1]).flatten()

#     results = []
#     for i, row in df.iterrows():
#         cand_skills = tokenize_candidate_skills(row["Skills"])
#         must = jd_struct["must_have"]
#         skill_match = sum(1 for ms in must if any(ms in cs for cs in cand_skills))
#         skill_score = skill_match / max(len(must), 1)

#         semantic = sims[i]
#         exp_years = extract_years(row["Experience"])
#         exp_score = min(1.0, exp_years / 5) if exp_years else 0.5

#         final = (0.35 * skill_score + 0.25 * semantic + 0.15 * exp_score) * 100
#         results.append((final, skill_score, semantic, exp_score))

#     df["score"], df["skill"], df["semantic"], df["exp"] = zip(*results)
#     return df

# def assign_priority(score, skill_score):
#     if score >= 75 and skill_score >= 0.5:
#         return "High"
#     if score >= 55:
#         return "Medium"
#     return "Low"

# def generate_message(name, role, matched):
#     top = ", ".join(matched[:3]) if matched else "your experience"
#     wa = f"Hi {name}, your experience in {top} looks relevant for a {role} role. Open to explore?"
#     email = f"""Subject: Opportunity - {role}

# Hi {name},

# Your background in {top} aligns well with our role.

# Would you be open to a quick chat?

# Best,
# Recruiter"""
#     return wa, email

# # ------------------------------
# # MAIN APP
# # ------------------------------
# def main():
#     st.set_page_config(layout="wide")
#     st.title(APP_TITLE)

#     uploaded = st.file_uploader("Upload CSV (with columns: Name, Skills, Experience, Current / last job title, etc.)", type=["csv"])
#     jd = st.text_area("Paste Job Description", height=200)

#     if not uploaded or not jd:
#         st.info("📂 Upload a CSV and paste a job description.")
#         return

#     # Load CSV
#     df = pd.read_csv(uploaded)

#     # --- Validate required columns ---
#     required_cols = ["Name", "Skills", "Experience", "Current / last job title"]
#     missing = [col for col in required_cols if col not in df.columns]
#     if missing:
#         st.error(f"❌ Missing required columns: {missing}. Please ensure your CSV has: {required_cols}")
#         st.stop()

#     st.success(f"✅ Found columns: {list(df.columns)}")

#     with st.expander("Preview CSV (first 5 rows)"):
#         st.dataframe(df.head(5))

#     # Parse JD and show extracted must‑haves
#     jd_struct = parse_jd(jd)
#     must_have = jd_struct["must_have"]
#     st.subheader("📌 Extracted Must‑Have Skills")
#     if must_have:
#         st.write(must_have)
#         st.caption(f"✅ **Selection rule:** Candidate is selected if they have **any** of the above skills (≥1 match).")
#     else:
#         st.warning("No must‑have skills found. Check that your JD has a 'Must have:' section with bullet points. All candidates will be selected.")

#     if st.button("🚀 Rank Candidates", type="primary"):
#         # Create combined text for semantic search
#         df["combined"] = df.astype(str).agg(" ".join, axis=1)

#         # ---- DEBUG TABLE (shows "Passes Filter" based on ANY match) ----
#         debug_rows = []
#         for idx, row in df.iterrows():
#             passed, match_cnt, matched, cand_skills = passes_filter(row, jd_struct)
#             debug_rows.append({
#                 "Name": row["Name"],
#                 "Candidate Skills (tokenized)": ", ".join(cand_skills),
#                 "Matched Must‑Haves": ", ".join(matched),
#                 "Match Count": match_cnt,
#                 "Passes Filter (any)": passed
#             })
#         debug_df = pd.DataFrame(debug_rows)
#         st.subheader("🔍 Hard Filter Evaluation (all candidates)")
#         st.dataframe(debug_df)

#         # Apply filter (any match)
#         filter_mask = df.apply(lambda row: passes_filter(row, jd_struct)[0], axis=1)
#         df_filtered = df[filter_mask].copy()

#         if df_filtered.empty:
#             st.error("❌ No candidates passed the hard filter (none have any of the must‑have skills).")
#             st.info("💡 Check that the extracted must‑have skills match the wording in the CSV 'Skills' column (e.g., 'LLM applications' vs 'LLMs').")
#             st.stop()

#         # Score and rank
#         df_filtered = compute_scores(df_filtered, jd_struct)
#         df_filtered["priority"] = df_filtered.apply(lambda x: assign_priority(x["score"], x["skill"]), axis=1)

#         # Show matched/missing for final output
#         matched_missing = df_filtered.apply(
#             lambda row: (
#                 [m for m in must_have if any(m in s for s in tokenize_candidate_skills(row["Skills"]))],
#                 [m for m in must_have if not any(m in s for s in tokenize_candidate_skills(row["Skills"]))]
#             ), axis=1
#         )
#         df_filtered["matched"], df_filtered["missing"] = zip(*matched_missing)
#         df_filtered = df_filtered.sort_values("score", ascending=False)

#         st.subheader("✅ Ranked Candidates (selected if they have ≥1 must‑have skill)")
#         for _, row in df_filtered.head(10).iterrows():
#             name = row["Name"]
#             title = row.get("Current / last job title", "N/A")
#             with st.container():
#                 st.markdown(f"### {name} - {title}")
#                 st.write(f"**Score:** {round(row['score'],1)} | **Priority:** {row['priority']}")
#                 st.write(f"**Matched:** {', '.join(row['matched'])}")
#                 st.write(f"**Missing:** {', '.join(row['missing'])}")
#                 wa, em = generate_message(name, "AI Role", row['matched'])
#                 with st.expander("📨 Outreach"):
#                     st.text("WhatsApp:")
#                     st.write(wa)
#                     st.text("Email:")
#                     st.code(em)

#         st.download_button("📥 Download Shortlist", df_filtered.to_csv(index=False), "shortlist.csv")

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_TITLE = "Talent Rediscovery Engine Pro"

# ------------------------------
# UTILS & PARSING
# ------------------------------
def clean_text(x):
    return "" if pd.isna(x) else str(x).lower().strip()

def tokenize_candidate_skills(text):
    text = clean_text(text)
    # Split by common delimiters but keep phrases like "Machine Learning"
    tokens = re.split(r"[,/;\n|\|]+", text)
    return [t.strip() for t in tokens if t.strip()]

def extract_years(text):
    if not text or pd.isna(text): return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(text))
    return float(m.group(1)) if m else None

def parse_jd(jd_text):
    """More robust parsing for 'Must Have' sections."""
    must_have = []
    capture = False
    # Look for common requirement headers
    trigger_keywords = ["must have", "requirements", "qualifications", "what you need"]
    stop_keywords = ["nice to have", "preferred", "plus", "bonus", "about the role"]
    
    lines = jd_text.split("\n")
    for line in lines:
        line_lower = line.lower()
        if any(k in line_lower for k in trigger_keywords):
            capture = True
            if ":" in line:
                after = line.split(":", 1)[1]
                must_have.extend(_extract_skill_phrases(after))
            continue

        if capture:
            stripped = line.strip()
            if not stripped or any(s in line_lower for s in stop_keywords):
                capture = False
                continue
            # Extract from bullet points
            if stripped.startswith(("-", "*", "•")) or re.match(r"^\d+\.", stripped):
                content = re.sub(r"^[\-\*\•\d\.]+\s*", "", stripped)
                must_have.extend(_extract_skill_phrases(content))

    unique_skills = list(dict.fromkeys([s for s in must_have if len(s) > 1]))
    return {"must_have": unique_skills, "text": clean_text(jd_text)}

def _extract_skill_phrases(text):
    text = re.sub(r"\s+or\s+", ",", text)
    return [p.strip(" .,;:!?()[]{}'\"") for p in text.split(",") if p.strip()]

# ------------------------------
# FILTERING & RANKING
# ------------------------------
def passes_filter(row, jd_struct):
    """Requirement 2: Check for AT LEAST one matching skill."""
    candidate_skills = tokenize_candidate_skills(row["Skills"])
    must_have = jd_struct["must_have"]
    if not must_have: return True, 0, [], candidate_skills

    matched = [m for m in must_have if any(m.lower() in cand.lower() for cand in candidate_skills)]
    return len(matched) >= 1, len(matched), matched, candidate_skills

def compute_scores(df, jd_struct):
    """Requirement 4: Optimized ranking with optional experience."""
    df = df.reset_index(drop=True)
    
    # Semantic similarity (TF-IDF)
    corpus = df["combined"].tolist() + [jd_struct["text"]]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(tfidf[:-1], tfidf[-1]).flatten()

    results = []
    for i, row in df.iterrows():
        cand_skills = tokenize_candidate_skills(row["Skills"])
        must = jd_struct["must_have"]
        
        # 1. Skill Match Ratio
        skill_match_count = sum(1 for ms in must if any(ms.lower() in cs.lower() for cs in cand_skills))
        skill_score = skill_match_count / max(len(must), 1)

        # 2. Semantic Score
        semantic = sims[i]

        # 3. Experience Score (Optional)
        exp_years = extract_years(row.get("Experience", ""))
        
        # DYNAMIC WEIGHTING: If exp is missing, redistribute weights to skills and semantic
        if exp_years is None:
            # No experience provided: 60% Skills, 40% Semantic
            final = (0.60 * skill_score + 0.40 * semantic) * 100
            exp_val = 0
        else:
            # Experience provided: 50% Skills, 30% Semantic, 20% Exp
            exp_score = min(1.0, exp_years / 5) # Cap at 5 years for max points
            final = (0.50 * skill_score + 0.30 * semantic + 0.20 * exp_score) * 100
            exp_val = exp_score

        results.append((final, skill_score, semantic, exp_val))

    df["score"], df["skill_score"], df["semantic_score"], df["exp_score"] = zip(*results)
    return df

# ------------------------------
# MAIN INTERFACE
# ------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(f"🎯 {APP_TITLE}")

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("Upload Candidate CSV", type=["csv"])
    with col2:
        jd = st.text_area("Paste Job Description here...", height=150)

    if uploaded and jd:
        df = pd.read_csv(uploaded)
        jd_struct = parse_jd(jd)
        
        # Sidebar summary
        st.sidebar.header("JD Analysis")
        st.sidebar.write("**Must-Have Skills detected:**")
        st.sidebar.write(jd_struct["must_have"] if jd_struct["must_have"] else "None found.")

        if st.button("🚀 Rank Candidates"):
            df["combined"] = df.astype(str).agg(" ".join, axis=1)
            
            # Step 2 & 3: Filter those with >= 1 match
            mask = df.apply(lambda r: passes_filter(r, jd_struct)[0], axis=1)
            df_filtered = df[mask].copy()

            if df_filtered.empty:
                st.warning("No candidates matched at least one must-have skill.")
            else:
                # Step 4: Rank
                df_ranked = compute_scores(df_filtered, jd_struct)
                df_ranked = df_ranked.sort_values("score", ascending=False)

                # Step 5: Show Ranking
                st.subheader(f"✅ Found {len(df_ranked)} Matching Candidates")
                
                for _, row in df_ranked.iterrows():
                    with st.expander(f"{row['Name']} — Score: {round(row['score'], 1)}%"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write(f"**Current Role:** {row.get('Current / last job title', 'N/A')}")
                            st.write(f"**Experience:** {row.get('Experience', 'N/A')}")
                        with c2:
                            st.write(f"**Skills:** {row['Skills']}")
                        
                        st.progress(row['score']/100)

if __name__ == "__main__":
    main()