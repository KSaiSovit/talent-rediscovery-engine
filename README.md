# Talent Rediscovery + Re-Engagement Engine

## Setup
1. `pip install -r requirements.txt`
2. `python generate_sample_csv.py`
3. `streamlit run app.py`

## Usage
- Upload the CSV (use sample_candidates.csv or your own)
- Paste a job description
- Click "Rank Candidates"
- Filter by priority, edit messages, export shortlist

## Sample job description to test
"We need a Data Scientist with Python, SQL, and Machine Learning. Experience with NLP is a plus."

## AI tools used
No external AI APIs – ranking uses TF‑IDF + skill overlap. See report for discussion on AI enhancement.