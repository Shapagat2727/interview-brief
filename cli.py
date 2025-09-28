import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from prep.core import build_prompt, fetch_jd_text, read_cv_text
from prep.llm import call_openai
from prep.render import render_markdown


def main():
    parser = argparse.ArgumentParser(
        description="Generate interview prep brief from JD and CV."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--jd-url", type=str, help="URL to job description")
    group.add_argument("--jd-text", type=str, help="Raw job description text")
    parser.add_argument(
        "--cv", type=str, required=True, help="Path to candidate CV (PDF or TXT)"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="OpenAI model name"
    )
    parser.add_argument(
        "--out", type=str, default="prep_brief.json", help="Output JSON file path"
    )
    parser.add_argument(
        "--no-stdout", action="store_true", help="Suppress Markdown output to stdout"
    )

    args = parser.parse_args()

    try:
        result = generate_prep_brief(
            jd_url=args.jd_url,
            jd_text=args.jd_text,
            cv_path=args.cv,
            model=args.model,
            out_path=args.out,
        )
        if not args.no_stdout and "markdown" in result:
            print(result["markdown"])
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def generate_prep_brief(
    jd_url: Optional[str],
    jd_text: Optional[str],
    cv_path: str,
    model: str,
    out_path: str,
) -> dict:
    # Load .env and check key
    load_dotenv()

    if not jd_url and not jd_text:
        raise ValueError("Provide either JD URL or JD text.")

    if jd_url:
        jd_text = fetch_jd_text(jd_url or None, jd_text or None)
    if not jd_text or len(jd_text) < 100:
        raise ValueError("JD text is too short or empty.")

    cv_text = read_cv_text(Path(cv_path))
    if not cv_text or len(cv_text) < 100:
        raise ValueError("CV text is too short or empty.")

    prompt = build_prompt(jd_text, cv_text)
    user_propmt = prompt.get("user")
    system_prompt = prompt.get("system")

    assert user_propmt and system_prompt, "Prompts cannot be empty."

    response = call_openai(system_prompt, user_propmt, model=model)
    
    if not isinstance(response, dict):
        raise ValueError("LLM response is not a valid JSON object.")

    markdown = render_markdown(response)

    out_json_path = Path(out_path)
    out_md_path = out_json_path.with_suffix(".md")

    with out_json_path.open("w", encoding="utf-8") as f_json:
        json.dump(response, f_json, indent=2, ensure_ascii=False)

    with out_md_path.open("w", encoding="utf-8") as f_md:
        f_md.write(markdown)

    return {
        "json_path": str(out_json_path),
        "markdown_path": str(out_md_path),
        "markdown": markdown,
    }


if __name__ == "__main__":
    main()
