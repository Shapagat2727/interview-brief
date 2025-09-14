import json
import re
from typing import Any, Dict, List

from openai import OpenAI


def call_openai(prompt: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    client = OpenAI()
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    # Collect text robustly
    text_chunks: List[str] = []
    if hasattr(resp, "output"):
        for item in resp.output:
            if getattr(item, "type", None) == "message":
                parts = getattr(
                    getattr(item, "message", None), "content", []
                ) or getattr(item, "content", [])
                for p in parts:
                    if getattr(p, "type", None) in ("output_text", "text"):
                        text_chunks.append(
                            getattr(p, "text", "") or getattr(p, "value", "")
                        )
    text = "".join(text_chunks).strip()

    # Parse JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = extract_json_block(text)
        return json.loads(cleaned)  # let it raise with raw text if fails


def extract_json_block(text: str) -> str:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if m:
        return m.group(0).strip()
    return text.strip()
