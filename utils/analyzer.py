"""
analyzer.py
===========
Pure Python logic for comparing skills and computing match scores.

This module has NO dependency on OpenAI — it works on plain Python lists.
Keeping this logic separate from llm_utils.py makes it:
  - Easy to test independently
  - Easy to explain in interviews (no magic, just set operations)
  - Easy to swap the algorithm later (e.g., use embeddings instead of keywords)

MATCH SCORE ALGORITHM:
  We use normalised string matching (lowercase + strip):
    match_score = (matched skills / total JD skills) × 100

  This is intentionally simple. In a production system, you might:
    - Use embedding similarity to catch "k8s" matching "Kubernetes"
    - Weight skills by importance (e.g., required vs nice-to-have)
    - Use fuzzy matching (rapidfuzz library)

  For a 1-year experience project, keyword matching is honest and explainable.
"""

from typing import List, Set


def _normalise(skill: str) -> str:
    """
    Normalise a skill string for comparison.
    Lowercases and strips whitespace to handle minor formatting differences.

    Examples:
      "Python " → "python"
      "LangChain" → "langchain"
      " REST API" → "rest api"
    """
    return skill.lower().strip()


def compute_match_score(
    resume_skills: List[str],
    jd_skills: List[str],
) -> int:
    """
    Compute what percentage of the job description's skills
    are covered by the candidate's resume.

    Formula:
      score = (number of JD skills found in resume / total JD skills) × 100

    Edge cases:
      - If JD has no skills detected → return 0
      - If resume has no skills → return 0
      - Score is rounded to the nearest integer

    Args:
        resume_skills: List of skills extracted from the resume
        jd_skills: List of skills extracted from the job description

    Returns:
        Integer match percentage (0–100)

    Example:
        resume_skills = ["Python", "FastAPI", "Docker"]
        jd_skills     = ["Python", "FastAPI", "Kubernetes", "Docker", "AWS"]
        → matched = {"python", "fastapi", "docker"} = 3
        → score = (3 / 5) × 100 = 60
    """
    if not jd_skills:
        return 0
    if not resume_skills:
        return 0

    # Normalise both lists for case-insensitive comparison
    resume_set: Set[str] = {_normalise(s) for s in resume_skills}
    jd_set: Set[str] = {_normalise(s) for s in jd_skills}

    # Count how many JD skills appear in the resume
    matched = resume_set.intersection(jd_set)
    matched_count = len(matched)

    score = (matched_count / len(jd_set)) * 100
    return round(score)


def find_missing_skills(
    resume_skills: List[str],
    jd_skills: List[str],
) -> List[str]:
    """
    Identify skills in the job description that are NOT in the resume.

    Uses set difference after normalisation.

    Args:
        resume_skills: Skills extracted from the candidate's resume
        jd_skills: Skills extracted from the job description

    Returns:
        List of missing skills (in their original casing from the JD)

    Example:
        resume_skills = ["Python", "FastAPI"]
        jd_skills     = ["Python", "FastAPI", "Kubernetes", "AWS"]
        → missing = ["Kubernetes", "AWS"]
    """
    if not jd_skills:
        return []

    # Build a normalised set of resume skills for fast lookup
    resume_normalised: Set[str] = {_normalise(s) for s in resume_skills}

    # Keep JD skills whose normalised form is NOT in the resume
    # We return the original (non-normalised) JD skill name for readability
    missing = [
        skill for skill in jd_skills
        if _normalise(skill) not in resume_normalised
    ]

    return missing


def get_matched_skills(
    resume_skills: List[str],
    jd_skills: List[str],
) -> List[str]:
    """
    Return skills that appear in BOTH the resume and the job description.
    Useful for highlighting strengths in the UI.

    Args:
        resume_skills: Skills from the resume
        jd_skills: Skills from the job description

    Returns:
        List of matched skill names (original casing from JD)
    """
    resume_normalised: Set[str] = {_normalise(s) for s in resume_skills}

    matched = [
        skill for skill in jd_skills
        if _normalise(skill) in resume_normalised
    ]

    return matched
