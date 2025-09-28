import json
import re
from typing import Any, Dict

from openai import OpenAI


def call_openai(system_prompt: str, user_propmt: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_propmt}
            ],
        response_format={"type": "json_object"}
    )
    completion = resp.choices[0].message.content
    return json.loads(completion)
