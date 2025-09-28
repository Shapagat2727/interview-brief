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


def build_prompt(jd_text: str, cv_text: str) -> dict:
    system_prompt = (
        "You are an elite Interview Prep Assistant. You receive a Job Description (JD) and a Candidate CV. "
        "Your task: produce a concise, actionable prep brief tailored to the candidate.\n\n"
        "Return JSON as in this example:\n"
        "```json\n"
        "{\n"
        '  "top_required_skills": ["Skill1", "Skill2"],\n'
        '  "strong_overlaps": ["Overlap1", "Overlap2"],\n'
        '  "gaps_and_risks": ["Gap1", "Risk1"],\n'
        '  "likely_tech_questions": ["Tech Question1", "Tech Question2"],\n'
        '  "behavioral_questions": ["Behavioral Question1", "Behavioral Question2"],\n'
        '  "talking_points": ["Point1", "Point2"],\n'
        '  "quick_upskilling_plan": ["Task1", "Task2"],\n'
        '  "role_summary": "Concise summary of the role."\n'
        "}\n"
        "```\n\n"
        "Guidelines:\n"
        "- Extract top required skills from the JD.\n"
        "- Identify strong overlaps between the CV and JD.\n"
        "- List gaps and risks based on the CV vs JD.\n"
        "- Generate likely technical and behavioral questions.\n"
        "- Suggest high-impact talking points for the interview.\n"
        "- Propose a quick upskilling plan (tasks ≤2h).\n"
        "- Ensure the JSON is valid and parsable.\n"
    )
    user_prompt = (
        f"Job Description:\n{jd_text}\n\nCandidate CV:\n{cv_text}\n\n"
        "Now, produce the prep brief as specified in JSON format."
    )
    return {"system": system_prompt, "user": user_prompt}
