"""
AI-Powered Resume Analyzer
===========================
Main application file — handles the Streamlit UI and orchestrates
the full analysis pipeline:

  1. User uploads resume (PDF or TXT)
  2. User pastes job description
  3. App extracts text from resume
  4. LLM extracts skills from resume and job description
  5. App computes match score and identifies missing skills
  6. LLM generates personalised improvement suggestions
  7. Results are displayed in a clean Streamlit UI
"""

import os
import streamlit as st
from dotenv import load_dotenv

from utils.parser import extract_text_from_file
from utils.llm_utils import extract_skills_from_text, generate_suggestions
from utils.analyzer import compute_match_score, find_missing_skills

# ── Load environment variables (.env) ─────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── Streamlit page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📄 AI-Powered Resume Analyzer")
st.markdown(
    "Upload your resume and paste a job description. "
    "The AI will extract skills, score your match, and suggest improvements."
)
st.divider()

# ── API key check ──────────────────────────────────────────────────────────────
if not OPENAI_API_KEY:
    st.error(
        "⚠️ OpenAI API key not found. "
        "Create a `.env` file with `OPENAI_API_KEY=your_key_here`."
    )
    st.stop()

# ── Two-column layout: inputs on left, results on right ───────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

# ═══════════════════════════════════════════════════════════════
# LEFT COLUMN — Inputs
# ═══════════════════════════════════════════════════════════════
with left_col:
    st.subheader("📂 Step 1: Upload Your Resume")
    uploaded_file = st.file_uploader(
        label="Choose a PDF or TXT file",
        type=["pdf", "txt"],
        help="Your resume will be parsed locally — no data is stored.",
    )

    st.subheader("📋 Step 2: Paste Job Description")
    job_description = st.text_area(
        label="Job Description",
        placeholder="Paste the full job description here…",
        height=300,
    )

    analyze_button = st.button(
        label="🔍 Analyze Resume",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None or not job_description.strip()),
    )

# ═══════════════════════════════════════════════════════════════
# RIGHT COLUMN — Results
# ═══════════════════════════════════════════════════════════════
with right_col:
    st.subheader("📊 Analysis Results")

    # Only run analysis when button is clicked
    if analyze_button:

        # ── Step 1: Extract text from the uploaded resume ──────────────────
        with st.spinner("Reading your resume…"):
            resume_text = extract_text_from_file(uploaded_file)

        if not resume_text.strip():
            st.error("Could not extract text from the uploaded file. Try a different file.")
            st.stop()

        # ── Step 2: Extract skills from resume using LLM ───────────────────
        with st.spinner("Extracting skills from resume…"):
            resume_skills_data = extract_skills_from_text(
                text=resume_text,
                source_label="resume",
                api_key=OPENAI_API_KEY,
            )

        # ── Step 3: Extract skills from job description using LLM ──────────
        with st.spinner("Extracting skills from job description…"):
            jd_skills_data = extract_skills_from_text(
                text=job_description,
                source_label="job description",
                api_key=OPENAI_API_KEY,
            )

        resume_skills = resume_skills_data.get("skills", [])
        jd_skills = jd_skills_data.get("skills", [])

        # ── Step 4: Compute match score ────────────────────────────────────
        match_score = compute_match_score(resume_skills, jd_skills)

        # ── Step 5: Find missing skills ────────────────────────────────────
        missing_skills = find_missing_skills(resume_skills, jd_skills)

        # ── Step 6: Generate improvement suggestions via LLM ───────────────
        with st.spinner("Generating improvement suggestions…"):
            suggestions_data = generate_suggestions(
                resume_skills=resume_skills,
                missing_skills=missing_skills,
                job_description=job_description,
                api_key=OPENAI_API_KEY,
            )

        suggestions = suggestions_data.get("suggestions", [])

        # ══════════════════════════════════════════════════════════
        # DISPLAY RESULTS
        # ══════════════════════════════════════════════════════════

        # — Match Score ————————————————————————————————————————————
        st.markdown("### 🎯 Match Score")

        # Colour-code the score based on threshold
        if match_score >= 70:
            score_color = "green"
            score_label = "Strong Match"
        elif match_score >= 40:
            score_color = "orange"
            score_label = "Partial Match"
        else:
            score_color = "red"
            score_label = "Weak Match"

        st.markdown(
            f"<h2 style='color:{score_color};'>{match_score}% — {score_label}</h2>",
            unsafe_allow_html=True,
        )
        st.progress(match_score / 100)
        st.caption(
            f"Your resume matched **{match_score}%** of the skills required by this job."
        )

        st.divider()

        # — Skills side-by-side ————————————————————————————————————
        skill_col1, skill_col2 = st.columns(2)

        with skill_col1:
            st.markdown("### ✅ Your Resume Skills")
            if resume_skills:
                for skill in resume_skills:
                    st.markdown(f"- {skill}")
            else:
                st.info("No skills detected in resume.")

        with skill_col2:
            st.markdown("### 🎯 Job Description Skills")
            if jd_skills:
                for skill in jd_skills:
                    st.markdown(f"- {skill}")
            else:
                st.info("No skills detected in job description.")

        st.divider()

        # — Missing Skills ————————————————————————————————————————
        st.markdown("### ❌ Missing Skills")
        if missing_skills:
            st.warning(
                f"Your resume is missing **{len(missing_skills)}** skill(s) "
                "mentioned in the job description:"
            )
            for skill in missing_skills:
                st.markdown(f"- `{skill}`")
        else:
            st.success("Your resume covers all skills mentioned in the job description!")

        st.divider()

        # — Suggestions ───────────────────────────────────────────
        st.markdown("### 💡 Improvement Suggestions")
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                st.markdown(f"**{i}.** {suggestion}")
        else:
            st.info("No suggestions generated.")

        st.divider()

        # — Raw resume text expander ──────────────────────────────
        with st.expander("📃 View Extracted Resume Text"):
            st.text(resume_text[:3000] + ("…" if len(resume_text) > 3000 else ""))

    else:
        # Placeholder when no analysis has been run yet
        st.info(
            "👈 Upload your resume and paste a job description, "
            "then click **Analyze Resume** to see results here."
        )

        st.markdown("**What you'll get:**")
        st.markdown(
            """
            - ✅ Skills extracted from your resume
            - 🎯 Skills required by the job
            - 📊 Match score (0–100%)
            - ❌ Skills you're missing
            - 💡 Personalised improvement tips
            """
        )
