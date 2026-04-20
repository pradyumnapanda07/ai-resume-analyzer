# 📄 AI-Powered Resume Analyzer

An intelligent resume analysis tool that uses **GPT-4o-mini** to extract skills from your resume and a job description, computes a match score, identifies skill gaps, and generates personalised improvement suggestions.

> Upload resume → Paste job description → Get an instant, AI-powered gap analysis.

---

## 📸 Project Overview

Manually comparing a resume to a job description is tedious and often subjective. This tool automates the process using LLM-based NLP — the same way a human recruiter would read both documents — and delivers structured, actionable output in seconds.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM | OpenAI GPT-4o-mini |
| UI | Streamlit |
| PDF Parsing | pypdf |
| Skill Matching | Python set operations (normalised keyword matching) |
| Environment | python-dotenv |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                     USER INPUTS                      │
│   Resume (PDF/TXT)    +    Job Description (text)    │
└───────────────────┬──────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│                 STEP 1: PARSING                      │
│   parser.py — Extract raw text from PDF or TXT       │
└───────────────────┬──────────────────────────────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
┌──────────────────┐ ┌──────────────────────────────┐
│  Resume Skills   │ │    JD Skills Extraction       │
│  llm_utils.py    │ │    llm_utils.py               │
│  GPT extracts    │ │    GPT extracts required       │
│  candidate skills│ │    skills from JD             │
└────────┬─────────┘ └──────────────┬───────────────┘
         │                          │
         └──────────┬───────────────┘
                    ▼
┌──────────────────────────────────────────────────────┐
│               STEP 3: ANALYSIS                       │
│   analyzer.py — Match score + Missing skills         │
│   (Pure Python set operations — no LLM needed here)  │
└───────────────────┬──────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│           STEP 4: SUGGESTION GENERATION              │
│   llm_utils.py — GPT generates personalised tips     │
│   based on gaps + job context                        │
└───────────────────┬──────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│                STREAMLIT UI OUTPUT                   │
│   Match Score │ Skill Lists │ Gaps │ Suggestions     │
└──────────────────────────────────────────────────────┘
```

---

## ⚙️ How It Works — Step by Step

**Step 1 — Resume Parsing** (`utils/parser.py`)
The uploaded file is read using `pypdf` (for PDFs) or decoded directly (for TXT). Raw text is extracted and passed to the next stage.

**Step 2 — Skill Extraction** (`utils/llm_utils.py`)
Two separate GPT calls are made — one for the resume, one for the job description. Each call sends the text with a structured prompt instructing GPT to return skills as a JSON array. Using an LLM here (instead of a keyword list) allows it to understand context, normalise names ("k8s" → "Kubernetes"), and catch implied skills.

**Step 3 — Match Scoring** (`utils/analyzer.py`)
Skill lists are compared using normalised set intersection. The score formula is:
```
match_score = (skills in resume ∩ skills in JD) / (total JD skills) × 100
```
This is pure Python — fast, transparent, and easy to explain.

**Step 4 — Suggestion Generation** (`utils/llm_utils.py`)
A third GPT call receives the candidate's existing skills, the missing skills, and the job description. GPT generates 4–5 specific, actionable tips targeted at closing the gap for this exact role.

**Step 5 — Display** (`app.py`)
Results are shown in Streamlit: a colour-coded match score, side-by-side skill comparison, missing skills list, and numbered improvement suggestions.

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.10 or higher
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 1. Clone the repository

```bash
git clone [https://github.com/your-username/resume-analyzer.git](https://github.com/pradyumnapanda07/ai-resume-analyzer.git)
cd resume-analyzer
```

### 2. Create and activate a virtual environment


# Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with your real key:
```
OPENAI_API_KEY=sk-...
```

### 5. Run the app

```bash
streamlit run app.py
```

App opens at `http://localhost:8501`

### 6. Test it

Upload `data/sample_resume.txt` and paste this sample job description:

## 📁 Project Structure

```
resume-analyzer/
│
├── app.py                   # Streamlit UI + pipeline orchestration
│
├── utils/
│   ├── __init__.py
│   ├── parser.py            # PDF/TXT text extraction
│   ├── llm_utils.py         # All OpenAI API calls + prompt engineering
│   └── analyzer.py          # Match score + missing skills (pure Python)
│
├── data/
│   └── sample_resume.txt    # Sample resume for testing
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚠️ Limitations

- **Keyword matching only**: The match score uses normalised string comparison. "k8s" and "Kubernetes" would count as different skills unless the LLM normalises them (which it usually does).
- **Text PDFs only**: Scanned or image-based PDFs are not supported. pypdf extracts text; it cannot OCR images.
- **No persistence**: Results are not saved between sessions.
- **API cost**: Each analysis makes 3 OpenAI API calls (resume extraction, JD extraction, suggestions). With gpt-4o-mini, this is typically under $0.01 per run.

---

## 🔮 Future Improvements

- Add **embedding-based similarity** to catch semantic skill matches ("deep learning" matching "neural networks")
- Add **RAGAS-style evaluation** to measure extraction quality
- Support **multiple resume formats** (DOCX, LinkedIn PDF)
- Add **section-level parsing** (extract skills specifically from the Skills section vs. Projects section)
- Add **job description scraping** via URL input

---

## 📄 License

MIT License — free to use and modify.
