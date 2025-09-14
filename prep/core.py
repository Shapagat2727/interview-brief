import os
import re
from pathlib import Path
from typing import List, Optional

import pdfplumber
import requests
from bs4 import BeautifulSoup


def fetch_jd_text(url: Optional[str], fallback_text: Optional[str]) -> str:
    if fallback_text and fallback_text.strip():
        return fallback_text.strip()
    if not url:
        raise ValueError("Provide either JD URL or JD text.")

    resp = requests.get(url, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for t in soup(["nav", "script", "style", "footer", "header", "aside", "form"]):
        t.decompose()

    main = soup.find("main") or soup.find("article")
    text = main.get_text("\n") if main else soup.get_text("\n")
    text = re.sub(r"\n{2,}", "\n", text).strip()

    if not text or len(text) < 200:
        soup2 = BeautifulSoup(resp.text, "html.parser")
        for t in soup2(["script", "style"]):
            t.decompose()
        text = re.sub(r"\n{2,}", "\n", soup2.get_text("\n")).strip()

    return text


def read_cv_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".pdf":
        texts: List[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
        out = "\n".join(texts).strip()
        if not out:
            raise ValueError(
                "CV PDF text could not be extracted. Export a text-based PDF."
            )
        return out
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def build_prompt(jd_text: str, cv_text: str) -> str:
    return f"""
You are an elite Interview Prep Assistant. You receive a Job Description (JD) and a Candidate CV.
Your task: produce a concise, *actionable* prep brief tailored to the candidate.

Return JSON with these keys only:
- "role_summary": 2–3 sentences on what this role really needs
- "top_required_skills": list of 5–8 skills (ranked, short labels)
- "strong_overlaps": list of bullet points mapping CV strengths to JD needs
- "gaps_and_risks": list of gaps likely to be probed in interviews
- "likely_tech_questions": list of 6–10 questions (mix of conceptual and hands-on)
- "behavioral_questions": list of 4–6 questions tied to JD themes
- "talking_points": list of 5–8 high-impact points the candidate should emphasize
- "quick_upskilling_plan": 3–5 concrete mini-tasks (≤2 hours each) to cover gaps before interview

Rules:
- Keep answers tight and skimmable.
- Prefer the candidate's actual experience; do not invent.
- If a JD skill is missing in the CV, call it out in "gaps_and_risks" and propose how to handle it.

Return ONLY a valid JSON object with exactly those keys. Do not add markdown, commentary, or code fences.

=== JOB_DESCRIPTION ===
{jd_text}

=== CANDIDATE_CV ===
{cv_text}
""".strip()
