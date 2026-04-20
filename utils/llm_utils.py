"""
llm_utils.py
============
All OpenAI API calls live here. This module handles:
  1. Skill extraction — from resume or job description text
  2. Suggestion generation — personalised improvement tips

WHY LLM FOR NLP INSTEAD OF TRADITIONAL NLP?
  Traditional NLP (spaCy, NLTK) uses rule-based or statistical methods.
  They are good at standard patterns but struggle with:
    - Non-standard resume formats
    - Abbreviated or informal skill names ("k8s" vs "Kubernetes")
    - Context-aware extraction ("experience with leading teams" → "leadership")

  LLMs understand context, synonyms, and implicit meaning — they extract
  skills the same way a human recruiter would read the document.

PROMPT DESIGN:
  Every prompt in this file:
    - Gives the LLM a clear role ("You are a skill extraction expert")
    - Specifies EXACTLY what format to return (JSON schema)
    - Instructs it to return ONLY JSON — no extra text, no markdown fences
    - Uses temperature=0 for deterministic, consistent outputs
"""

import json
import re
from typing import Dict, List
from openai import OpenAI


# ── Model configuration ────────────────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"    # Fast and affordable — good for structured extraction
TEMPERATURE = 0               # 0 = fully deterministic output (important for JSON)
MAX_TOKENS = 1000             # Enough for a list of skills + suggestions


def extract_skills_from_text(
    text: str,
    source_label: str,
    api_key: str,
) -> Dict:
    """
    Use GPT to extract professional skills from a given text (resume or JD).

    Why not use regex or keyword lists?
      A keyword list would miss "k8s" (Kubernetes), "Pandas" used in a different
      context, or soft skills described in sentences. GPT reads like a human.

    Args:
        text: Raw text of resume or job description
        source_label: "resume" or "job description" — used in the prompt
        api_key: OpenAI API key

    Returns:
        Dict with key "skills" containing a list of skill strings.
        Example: {"skills": ["Python", "LangChain", "FastAPI", "RAG"]}
    """
    client = OpenAI(api_key=api_key)

    # Truncate very long texts to stay within token limits
    # 4000 chars ≈ ~1000 tokens — safe for gpt-4o-mini's context window
    truncated_text = text[:4000]

    # ── Skill Extraction Prompt ────────────────────────────────────────────────
    # Key prompt engineering decisions:
    #   1. Role assignment: "You are an expert..."
    #   2. Explicit JSON schema in the prompt
    #   3. "Return ONLY valid JSON" — prevents markdown code blocks or extra text
    #   4. Specific examples to guide extraction style
    #   5. Include both technical AND soft skills
    prompt = f"""You are an expert technical recruiter and skill extraction specialist.

Your task is to extract ALL professional skills from the {source_label} text below.

Include:
- Technical skills (programming languages, frameworks, tools, platforms)
- Domain knowledge (machine learning, NLP, RAG, cloud computing, etc.)
- Soft skills only if explicitly mentioned (leadership, communication, etc.)
- Normalise skill names: use "Kubernetes" not "k8s", "LangChain" not "langchain"

Return your response as ONLY valid JSON in this exact format — no extra text, no markdown:
{{
  "skills": ["skill1", "skill2", "skill3"]
}}

{source_label.upper()} TEXT:
{truncated_text}"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {
                    "role": "system",
                    "content": "You are a skill extraction engine. Always respond with valid JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        raw_output = response.choices[0].message.content.strip()
        return _parse_json_response(raw_output, fallback_key="skills")

    except Exception as e:
        print(f"[ERROR] Skill extraction failed: {e}")
        return {"skills": []}


def generate_suggestions(
    resume_skills: List[str],
    missing_skills: List[str],
    job_description: str,
    api_key: str,
) -> Dict:
    """
    Use GPT to generate personalised, actionable improvement suggestions.

    The suggestions are tailored based on:
      - What skills the candidate already has (build on strengths)
      - What skills are missing (target gaps)
      - The job description context (relevance to the role)

    Args:
        resume_skills: Skills found in the candidate's resume
        missing_skills: Skills in the JD but not in the resume
        job_description: Full job description text (for context)
        api_key: OpenAI API key

    Returns:
        Dict with key "suggestions" containing a list of tip strings.
        Example: {"suggestions": ["Add LangChain to your resume by building...", ...]}
    """
    client = OpenAI(api_key=api_key)

    # Truncate JD to avoid token overflow
    truncated_jd = job_description[:2000]

    # ── Suggestion Generation Prompt ──────────────────────────────────────────
    # Prompt engineering decisions:
    #   1. Provide full context: existing skills + gaps + job context
    #   2. Constrain output: exactly 4-5 actionable tips
    #   3. Specify tone: "actionable", "specific", not generic advice
    #   4. JSON-only output requirement
    prompt = f"""You are a career coach specialising in helping software engineers and AI professionals improve their resumes.

A candidate is applying for a role. Here is their situation:

SKILLS THEY HAVE:
{", ".join(resume_skills) if resume_skills else "Not specified"}

SKILLS THEY ARE MISSING (required by the job):
{", ".join(missing_skills) if missing_skills else "None — great coverage!"}

JOB DESCRIPTION CONTEXT:
{truncated_jd}

Your task: Generate 4 to 5 specific, actionable improvement suggestions to help this candidate strengthen their application for THIS specific role.

Rules:
- Be specific — mention actual skill names, not vague advice
- Suggest concrete actions: "Build a project using X", "Add Y to your skills section", "Take the Z certification"
- Do NOT repeat what they already have
- Focus on the gaps and how to address them realistically
- Keep each suggestion to 1-2 sentences

Return ONLY valid JSON in this exact format — no extra text:
{{
  "suggestions": [
    "suggestion 1",
    "suggestion 2",
    "suggestion 3",
    "suggestion 4"
  ]
}}"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.3,   # Slight creativity allowed for suggestions (unlike extraction)
            max_tokens=MAX_TOKENS,
            messages=[
                {
                    "role": "system",
                    "content": "You are a career coaching engine. Always respond with valid JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        raw_output = response.choices[0].message.content.strip()
        return _parse_json_response(raw_output, fallback_key="suggestions")

    except Exception as e:
        print(f"[ERROR] Suggestion generation failed: {e}")
        return {"suggestions": ["Could not generate suggestions. Please try again."]}


def _parse_json_response(raw_text: str, fallback_key: str) -> Dict:
    """
    Safely parse the LLM's JSON response.

    Even with "return ONLY JSON" in the prompt, LLMs occasionally wrap output
    in markdown code fences (```json ... ```). This function strips those before
    parsing — making the output robust to minor LLM formatting variations.

    Args:
        raw_text: Raw string from the LLM response
        fallback_key: Key to use in fallback dict if parsing fails

    Returns:
        Parsed dict, or a safe empty fallback dict
    """
    try:
        # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
        cleaned = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()

        parsed = json.loads(cleaned)
        return parsed

    except json.JSONDecodeError as e:
        print(f"[WARNING] JSON parse failed: {e}")
        print(f"[WARNING] Raw LLM output was: {raw_text[:200]}")
        # Return safe empty fallback so the app doesn't crash
        return {fallback_key: []}
