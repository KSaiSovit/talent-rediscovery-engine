import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import base64

# ----------------------------
# 1. Text preprocessing
# ----------------------------
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'a', 'an', 'and', 'are', 'as', 'at',
    'be', 'by', 'for', 'from', 'has', 'have', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
    'or', 'that', 'the', 'this', 'to', 'was', 'were', 'will', 'with', 'about', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'over'
])

def clean_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def extract_skill_tokens(text):
    """Return set of tokens from job description (potential skills)."""
    return set(clean_text(text))

# ----------------------------
# 2. Matching engine
# ----------------------------
def compute_match_score(jd_tokens, candidate_text, candidate_skills_list, jd_full_text):
    """
    Returns: (score, matched_skills, missing_skills)
    """
    # TF‑IDF similarity
    vectorizer = TfidfVectorizer(tokenizer=clean_text, token_pattern=None)
    try:
        tfidf_matrix = vectorizer.fit_transform([jd_full_text, candidate_text])
        cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        cos_sim = 0.0
    tfidf_score = cos_sim * 100

    # Skill overlap
    candidate_skill_set = set([s.lower().strip() for s in candidate_skills_list if isinstance(s, str)])
    if len(jd_tokens) > 0:
        matched = candidate_skill_set.intersection(jd_tokens)
        overlap_ratio = len(matched) / len(jd_tokens)
        skill_score = overlap_ratio * 100
    else:
        matched = set()
        skill_score = 0

    final_score = 0.7 * tfidf_score + 0.3 * skill_score
    missing = jd_tokens - candidate_skill_set
    return round(final_score, 1), list(matched), list(missing)

# ----------------------------
# 3. Message generation (template)
# ----------------------------
def generate_messages(candidate_name, job_title, company_name, score, matched_skills, missing_skills):
    wa = f"👋 Hi {candidate_name},\n\nWe're reconnecting because your background in {', '.join(matched_skills[:3])} caught our eye for a {job_title} role. Your profile matches {score}% of our requirements.\n\nWould you be open to a quick chat? Let me know!\n\nBest,\nRecruitment Team"
    email_subj = f"Exciting opportunity: {job_title} role at {company_name}"
    email_body = f"""Dear {candidate_name},

I hope you're doing well.

We came across your previous application and believe your experience with {', '.join(matched_skills[:3])} makes you a strong candidate for the {job_title} position at {company_name}.

Match Score: {score}%
Key Strengths: {', '.join(matched_skills)}
Gaps to note: {', '.join(missing_skills[:3]) if missing_skills else 'None – great fit!'}

Would you be interested in discussing this role further? Please let me know a convenient time for a quick call.

Looking forward to hearing from you.

Best regards,
Talent Acquisition Team
"""
    return wa, email_subj, email_body

# ----------------------------
# 4. Export helper
# ----------------------------
def get_csv_download_link(df, filename="shortlisted_candidates.csv"):
    output = BytesIO()
    df.to_csv(output, index=False)
    b64 = base64.b64encode(output.getvalue()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'
    return href

# ----------------------------
# 5. Main Streamlit app
# ----------------------------
st.set_page_config(page_title="Talent Rediscovery Engine", layout="wide")
st.title("🔍 Talent Rediscovery + Re‑Engagement Engine")
st.markdown("Upload a CSV of past candidates, paste a new job description, and get ranked matches with personalized outreach.")

with st.sidebar:
    st.header("📂 Upload & Input")
    uploaded_file = st.file_uploader("Candidates CSV", type=["csv"])
    jd_text = st.text_area("Job Description", height=200,
                           placeholder="Paste the job description here...")
    run = st.button("🚀 Rank Candidates", type="primary")
    st.markdown("---")
    st.markdown("**Expected CSV columns (at least):**")
    st.code("Name, Email, Phone, Skills, Current / last job title, Resume text or profile summary")
    st.markdown("Other columns (Experience, Location, etc.) are optional but enrich the output.")

if uploaded_file and jd_text and run:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} candidates")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # Preprocess JD
    jd_tokens = extract_skill_tokens(jd_text)
    jd_full_text = jd_text

    results = []
    progress = st.progress(0)

    for idx, row in df.iterrows():
        name = row.get("Name", "Candidate")
        job_title = row.get("Current / last job title", "")
        skills_raw = row.get("Skills", "")
        resume = row.get("Resume text or profile summary", "")
        exp = row.get("Experience", "")
        company = row.get("Current company", "our company")
        email = row.get("Email", "")
        phone = row.get("Phone", "")
        location = row.get("Location", "")
        notice = row.get("Notice period", "")
        last_int = row.get("Last interaction date", "")

        # Build candidate text for similarity
        candidate_text = f"{job_title} {skills_raw} {resume} {exp}"
        # Parse skills list
        if isinstance(skills_raw, str):
            skills_list = [s.strip() for s in re.split(r'[;,|]', skills_raw) if s.strip()]
        else:
            skills_list = []

        score, matched, missing = compute_match_score(jd_tokens, candidate_text, skills_list, jd_full_text)

        # Priority
        if score >= 70:
            priority = "High"
        elif score >= 40:
            priority = "Medium"
        else:
            priority = "Low"

        # Fit summary
        if matched:
            fit = f"Matches {len(matched)} key skills: {', '.join(matched[:3])}. "
        else:
            fit = "No direct skill match found. "
        if exp:
            fit += f"Experience: {exp}. "
        fit += f"Current role: {job_title if job_title else 'Not specified'}."

        # Messages
        wa_msg, email_subj, email_body = generate_messages(
            name, job_title, company, score, matched, missing
        )

        results.append({
            "Name": name,
            "Email": email,
            "Phone": phone,
            "Current Job Title": job_title,
            "Company": company,
            "Location": location,
            "Notice Period": notice,
            "Last Interaction": last_int,
            "Match Score (%)": score,
            "Priority": priority,
            "Fit Summary": fit,
            "Matching Skills": ", ".join(matched),
            "Missing Skills": ", ".join(missing[:5]),
            "WhatsApp Message": wa_msg,
            "Email Subject": email_subj,
            "Email Body": email_body
        })
        progress.progress((idx + 1) / len(df))

    results_df = pd.DataFrame(results).sort_values("Match Score (%)", ascending=False)

    st.subheader("🏆 Ranked Candidates")
    priority_filter = st.multiselect("Filter by Priority", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    filtered = results_df[results_df["Priority"].isin(priority_filter)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Candidates", len(results_df))
    col2.metric("High Priority", len(results_df[results_df["Priority"] == "High"]))
    col3.metric("Medium/Low", len(results_df[results_df["Priority"].isin(["Medium", "Low"])]))

    if st.button("Export visible candidates to CSV"):
        st.markdown(get_csv_download_link(filtered), unsafe_allow_html=True)

    for _, cand in filtered.iterrows():
        with st.expander(f"⭐ {cand['Name']} — Score {cand['Match Score (%)']}% — {cand['Priority']}"):
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown(f"**📋 Fit Summary:** {cand['Fit Summary']}")
                st.markdown(f"**✅ Matching Skills:** {cand['Matching Skills']}")
                st.markdown(f"**⚠️ Missing Skills:** {cand['Missing Skills']}")
                st.markdown(f"**💼 Role:** {cand['Current Job Title']} @ {cand['Company']}")
                st.markdown(f"**📍 Location:** {cand['Location']}  |  **📞 Phone:** {cand['Phone']}")
            with col_b:
                st.markdown("**📨 Outreach**")
                st.text_area("WhatsApp", cand["WhatsApp Message"], height=100, key=f"wa_{cand['Name']}")
                st.text_input("Email Subject", cand["Email Subject"], key=f"subj_{cand['Name']}")
                st.text_area("Email Body", cand["Email Body"], height=150, key=f"email_{cand['Name']}")

    with st.expander("📊 Full data table"):
        st.dataframe(filtered)

elif run:
    st.warning("Please upload a CSV and paste a job description first.")