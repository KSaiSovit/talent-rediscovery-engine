# import streamlit as st
# import pandas as pd
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# APP_TITLE = "Talent Rediscovery Engine"

# # UTILS & PARSING
# def clean_text(x):
#     return "" if pd.isna(x) else str(x).lower().strip()

# def tokenize_candidate_skills(text):
#     text = clean_text(text)
#     # Split by common delimiters but keep phrases like "Machine Learning"
#     tokens = re.split(r"[,/;\n|\|]+", text)
#     return [t.strip() for t in tokens if t.strip()]

# def extract_years(text):
#     if not text or pd.isna(text): return None
#     m = re.search(r"(\d+(?:\.\d+)?)", str(text))
#     return float(m.group(1)) if m else None

# def parse_jd(jd_text):
#     """More robust parsing for 'Must Have' sections."""
#     must_have = []
#     capture = False
#     # Look for common requirement headers
#     trigger_keywords = ["must have", "requirements", "qualifications", "what you need"]
#     stop_keywords = ["nice to have", "preferred", "plus", "bonus", "about the role"]
    
#     lines = jd_text.split("\n")
#     for line in lines:
#         line_lower = line.lower()
#         if any(k in line_lower for k in trigger_keywords):
#             capture = True
#             if ":" in line:
#                 after = line.split(":", 1)[1]
#                 must_have.extend(_extract_skill_phrases(after))
#             continue

#         if capture:
#             stripped = line.strip()
#             if not stripped or any(s in line_lower for s in stop_keywords):
#                 capture = False
#                 continue
#             # Extract from bullet points
#             if stripped.startswith(("-", "*", "•")) or re.match(r"^\d+\.", stripped):
#                 content = re.sub(r"^[\-\*\•\d\.]+\s*", "", stripped)
#                 must_have.extend(_extract_skill_phrases(content))

#     unique_skills = list(dict.fromkeys([s for s in must_have if len(s) > 1]))
#     return {"must_have": unique_skills, "text": clean_text(jd_text)}

# def _extract_skill_phrases(text):
#     text = re.sub(r"\s+or\s+", ",", text)
#     return [p.strip(" .,;:!?()[]{}'\"") for p in text.split(",") if p.strip()]

# # FILTERING & RANKING
# def passes_filter(row, jd_struct):
#     """Requirement 2: Check for AT LEAST one matching skill."""
#     candidate_skills = tokenize_candidate_skills(row["Skills"])
#     must_have = jd_struct["must_have"]
#     if not must_have: return True, 0, [], candidate_skills

#     matched = [m for m in must_have if any(m.lower() in cand.lower() for cand in candidate_skills)]
#     return len(matched) >= 1, len(matched), matched, candidate_skills

# def compute_scores(df, jd_struct):
#     """Requirement 4: Optimized ranking with optional experience."""
#     df = df.reset_index(drop=True)
    
#     # Semantic similarity (TF-IDF)
#     corpus = df["combined"].tolist() + [jd_struct["text"]]
#     vectorizer = TfidfVectorizer(stop_words="english")
#     tfidf = vectorizer.fit_transform(corpus)
#     sims = cosine_similarity(tfidf[:-1], tfidf[-1]).flatten()

#     results = []
#     for i, row in df.iterrows():
#         cand_skills = tokenize_candidate_skills(row["Skills"])
#         must = jd_struct["must_have"]
        
#         # Skill Match Ratio
#         skill_match_count = sum(1 for ms in must if any(ms.lower() in cs.lower() for cs in cand_skills))
#         skill_score = skill_match_count / max(len(must), 1)

#         # Semantic Score
#         semantic = sims[i]

#         # Experience Score (Optional)
#         exp_years = extract_years(row.get("Experience", ""))
        
#         # DYNAMIC WEIGHTING: If exp is missing, redistribute weights to skills and semantic
#         if exp_years is None:
#             # No experience provided: 60% Skills, 40% Semantic
#             final = (0.60 * skill_score + 0.40 * semantic) * 100
#             exp_val = 0
#         else:
#             # Experience provided: 50% Skills, 30% Semantic, 20% Exp
#             exp_score = min(1.0, exp_years / 5) # Cap at 5 years for max points
#             final = (0.50 * skill_score + 0.30 * semantic + 0.20 * exp_score) * 100
#             exp_val = exp_score

#         results.append((final, skill_score, semantic, exp_val))

#     df["score"], df["skill_score"], df["semantic_score"], df["exp_score"] = zip(*results)
#     return df

# # MAIN INTERFACE
# def main():
#     st.set_page_config(page_title=APP_TITLE, layout="wide")
#     st.title(f"🎯 {APP_TITLE}")

#     col1, col2 = st.columns([1, 1])
#     with col1:
#         uploaded = st.file_uploader("Upload Candidate CSV", type=["csv"])
#     with col2:
#         jd = st.text_area("Paste Job Description here...", height=150)

#     if uploaded and jd:
#         df = pd.read_csv(uploaded)
#         jd_struct = parse_jd(jd)
        
#         # Sidebar summary
#         st.sidebar.header("JD Analysis")
#         st.sidebar.write("**Must-Have Skills detected:**")
#         st.sidebar.write(jd_struct["must_have"] if jd_struct["must_have"] else "None found.")

#         if st.button("🚀 Rank Candidates"):
#             df["combined"] = df.astype(str).agg(" ".join, axis=1)
            
#             # Filter those with >= 1 match
#             mask = df.apply(lambda r: passes_filter(r, jd_struct)[0], axis=1)
#             df_filtered = df[mask].copy()

#             if df_filtered.empty:
#                 st.warning("No candidates matched at least one must-have skill.")
#             else:
#                 # Rank
#                 df_ranked = compute_scores(df_filtered, jd_struct)
#                 df_ranked = df_ranked.sort_values("score", ascending=False)

#                 # Show Ranking
#                 st.subheader(f"✅ Found {len(df_ranked)} Matching Candidates")
                
#                 for _, row in df_ranked.iterrows():
#                     with st.expander(f"{row['Name']} — Score: {round(row['score'], 1)}%"):
#                         c1, c2 = st.columns(2)
#                         with c1:
#                             st.write(f"**Current Role:** {row.get('Current / last job title', 'N/A')}")
#                             st.write(f"**Experience:** {row.get('Experience', 'N/A')}")
#                         with c2:
#                             st.write(f"**Skills:** {row['Skills']}")
                        
#                         st.progress(row['score']/100)

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import re
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_TITLE = "Talent Rediscovery Engine"

# UTILS & PARSING
def clean_text(x):
    return "" if pd.isna(x) else str(x).lower().strip()

def tokenize_candidate_skills(text):
    text = clean_text(text)
    tokens = re.split(r"[,/;\n|\|]+", text)
    return [t.strip() for t in tokens if t.strip()]

def extract_years(text):
    if not text or pd.isna(text): return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(text))
    return float(m.group(1)) if m else None

def parse_jd(jd_text):
    must_have = []
    capture = False
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
            if stripped.startswith(("-", "*", "•")) or re.match(r"^\d+\.", stripped):
                content = re.sub(r"^[\-\*\•\d\.]+\s*", "", stripped)
                must_have.extend(_extract_skill_phrases(content))

    unique_skills = list(dict.fromkeys([s for s in must_have if len(s) > 1]))
    return {"must_have": unique_skills, "text": clean_text(jd_text)}

def _extract_skill_phrases(text):
    text = re.sub(r"\s+or\s+", ",", text)
    return [p.strip(" .,;:!?()[]{}'\"") for p in text.split(",") if p.strip()]

# FILTERING & RANKING
def passes_filter(row, jd_struct):
    candidate_skills = tokenize_candidate_skills(row["Skills"])
    must_have = jd_struct["must_have"]
    if not must_have: return True, 0, [], candidate_skills
    matched = [m for m in must_have if any(m.lower() in cand.lower() for cand in candidate_skills)]
    return len(matched) >= 1, len(matched), matched, candidate_skills

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
        skill_match_count = sum(1 for ms in must if any(ms.lower() in cs.lower() for cs in cand_skills))
        skill_score = skill_match_count / max(len(must), 1)
        semantic = sims[i]
        exp_years = extract_years(row.get("Experience", ""))
        
        if exp_years is None:
            final = (0.60 * skill_score + 0.40 * semantic) * 100
            exp_val = 0
        else:
            exp_score = min(1.0, exp_years / 5)
            final = (0.50 * skill_score + 0.30 * semantic + 0.20 * exp_score) * 100
            exp_val = exp_score
        results.append((final, skill_score, semantic, exp_val))

    df["score"], df["skill_score"], df["semantic_score"], df["exp_score"] = zip(*results)
    return df

# MAIN INTERFACE
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(f"🎯 {APP_TITLE}")

    # --- SIDEBAR: MESSAGING TEMPLATES ---
    st.sidebar.header("Outreach Templates")
    st.sidebar.info("Use {name} as a placeholder for the candidate's name.")
    
    wa_template = st.sidebar.text_area(
        "WhatsApp Template", 
        "Hi {name}, long time no see! We have a new role that perfectly fits your profile. Are you open to a quick chat?",
        height=100
    )
    
    email_template = st.sidebar.text_area(
        "Email Template", 
        "Dear {name},\n\nI hope you're doing well. Our team is currently looking for experts with your background. Would you be interested in discussing a new opportunity?\n\nBest regards,\nRecruitment Team",
        height=150
    )

    # --- MAIN UI ---
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("Upload Candidate CSV", type=["csv"])
    with col2:
        jd = st.text_area("Paste Job Description here...", height=150)

    if uploaded and jd:
        df = pd.read_csv(uploaded)
        jd_struct = parse_jd(jd)
        
        st.sidebar.divider()
        st.sidebar.header("JD Analysis")
        st.sidebar.write("**Must-Have Skills detected:**")
        st.sidebar.write(jd_struct["must_have"] if jd_struct["must_have"] else "None found.")

        if st.button("🚀 Rank Candidates"):
            df["combined"] = df.astype(str).agg(" ".join, axis=1)
            mask = df.apply(lambda r: passes_filter(r, jd_struct)[0], axis=1)
            df_filtered = df[mask].copy()

            if df_filtered.empty:
                st.warning("No candidates matched at least one must-have skill.")
            else:
                df_ranked = compute_scores(df_filtered, jd_struct)
                df_ranked = df_ranked.sort_values("score", ascending=False)
                # Store in session state to persist through selection
                st.session_state['ranked_results'] = df_ranked

    # --- SELECTION & EXPORT SECTION ---
    if 'ranked_results' in st.session_state:
        results = st.session_state['ranked_results']
        
        st.subheader(f"✅ Found {len(results)} Matching Candidates")
        st.write("Select the candidates you want to re-engage with:")

        # Add a selection column
        results.insert(0, "Select", False)
        
        # Use Data Editor for easy selection
        edited_df = st.data_editor(
            results[["Select", "Name", "score", "Current / last job title", "Skills", "Experience"]],
            hide_index=True,
            use_container_width=True,
            disabled=["Name", "score", "Current / last job title", "Skills", "Experience"]
        )

        selected_names = edited_df[edited_df["Select"] == True]["Name"].tolist()
        
        if selected_names:
            st.success(f"{len(selected_names)} candidates selected for re-engagement.")
            
            if st.button("📦 Generate Export File"):
                # Filter original ranked data for selected names
                final_selection = results[results["Name"].isin(selected_names)].copy()
                
                # Format messages
                final_selection["whatsapp message"] = final_selection["Name"].apply(lambda x: wa_template.format(name=x))
                final_selection["email message"] = final_selection["Name"].apply(lambda x: email_template.format(name=x))
                
                # Rename columns for final CSV requirements
                export_df = final_selection[[
                    "Name", "Phone", "Email", "whatsapp message", "email message"
                ]].rename(columns={
                    "Name": "name",
                    "Phone": "phone number",
                    "Email": "email id"
                })

                # CSV Export logic
                csv_buffer = io.StringIO()
                export_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="📥 Download Outreach CSV",
                    data=csv_buffer.getvalue(),
                    file_name="reengagement_list.csv",
                    mime="text/csv"
                )
                
                st.dataframe(export_df)

if __name__ == "__main__":
    main()