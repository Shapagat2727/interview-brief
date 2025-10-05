import json
import os
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
from dotenv import load_dotenv


from prep.core import build_prompt, fetch_jd_text, read_cv_text
from prep.llm import call_openai
from prep.render import render_markdown

APP_TITLE = "Interview Prep Assistant (JD + CV → JSON + Markdown)"
DEFAULT_MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-5.1-mini"]


def _validate_inputs(
    jd_url: str,
    jd_text: str,
    cv_path: Optional[str],
    out_base: str,
) -> None:
    jd_url = (jd_url or "").strip()
    jd_text = (jd_text or "").strip()

    if not jd_url and not jd_text:
        raise gr.Error("Provide either a **JD URL** or **JD text**.")
    if jd_url and jd_text:
        raise gr.Error("Use **JD URL OR JD text** — not both.")
    if not cv_path:
        raise gr.Error("Please upload your **CV file (PDF/TXT)**.")
    if not out_base:
        raise gr.Error("Set **Output Base** (path without extension).")


def _ensure_outpaths(base: Path) -> Tuple[Path, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    return base.with_suffix(".json"), base.with_suffix(".md")


def generate(
    jd_url: str,
    jd_text: str,
    cv_file: Optional[str],
    model_text: str,
    out_base_text: str,
    progress=gr.Progress(track_tqdm=False),
):
    """
    Main callback that replicates the PySide worker behavior.
    Returns: (markdown_preview, json_view, status, json_file, md_file)
    """
    try:
        progress(0, desc="Validating inputs…")
        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            # Warn — but proceed in case the LLM wrapper uses other auth
            gr.Warning(
                "OPENAI_API_KEY is not set. Put it in a .env or the environment."
            )

        _validate_inputs(jd_url, jd_text, cv_file, out_base_text)

        progress(0.15, desc="Reading CV…")
        cv_path = Path(cv_file)
        cv_text = read_cv_text(cv_path)

        progress(0.35, desc="Fetching / parsing JD…")
        jd_full_text = fetch_jd_text((jd_url or None), (jd_text or None))

        progress(0.5, desc="Building prompts…")
        prompt = build_prompt(jd_full_text, cv_text)
        user_prompt = prompt.get("user")
        system_prompt = prompt.get("system")
        if not user_prompt or not system_prompt:
            raise gr.Error("Internal error: prompts are empty.")

        model = (model_text or "").strip() or "gpt-4o-mini"

        progress(0.7, desc=f"Calling LLM ({model})…")
        result = call_openai(system_prompt, user_prompt, model=model)

        progress(0.85, desc="Rendering markdown…")
        md_text_rendered = render_markdown(result)

        progress(0.9, desc="Saving files…")
        out_base = Path(out_base_text).expanduser().resolve()
        json_path, md_path = _ensure_outpaths(out_base)
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        md_path.write_text(md_text_rendered, encoding="utf-8")

        # Minimal schema check like your PySide version
        required = {
            "role_summary",
            "top_required_skills",
            "strong_overlaps",
            "gaps_and_risks",
            "likely_tech_questions",
            "behavioral_questions",
            "talking_points",
            "quick_upskilling_plan",
        }
        missing = sorted(list(required - set(result)))
        status = f"✅ Done. Saved:\n{json_path}\n{md_path}"
        if missing:
            status += f"\n\n⚠️ Missing keys: {missing}"

        progress(1.0, desc="Complete")
        return (
            md_text_rendered,
            result,  # JSON viewer in Gradio
            status,
            str(json_path),  # for gr.File
            str(md_path),  # for gr.File
        )
    except gr.Error:
        # validation or user-facing errors already formatted
        raise
    except Exception as e:
        raise gr.Error(f"{type(e).__name__}: {e}")


with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"# {APP_TITLE}\nUpload a CV, add a JD (URL or text), pick a model, and generate a focused prep brief."
    )

    with gr.Row():
        with gr.Column():
            with gr.Group():
                jd_url = gr.Textbox(
                    label="JD URL", placeholder="https://company.com/job/123"
                )
                jd_text = gr.Textbox(
                    label="JD Text",
                    placeholder="Paste JD text here (leave empty if using URL).",
                    lines=8,
                )
                cv_file = gr.File(
                    label="CV File (PDF or TXT)",
                    file_types=[".pdf", ".txt"],
                    type="filepath",
                )

            with gr.Group():
                model = gr.Textbox(
                    label="Model",
                    value="gpt-4o-mini",
                    info=f"Common: {', '.join(DEFAULT_MODELS)}",
                )
                out_base = gr.Textbox(
                    label="Output Base (without extension)",
                    value=str(Path.cwd() / "prep_brief"),
                    info="Example: /path/to/prep_brief → saves prep_brief.json & prep_brief.md",
                )

            generate_btn = gr.Button("⚡ Generate", variant="primary")

        with gr.Column():
            md_preview = gr.Markdown(label="Markdown Preview")
            json_view = gr.JSON(label="JSON Result")
            status_box = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                json_file = gr.File(label="Download JSON")
                md_file = gr.File(label="Download Markdown")

    generate_btn.click(
        fn=generate,
        inputs=[jd_url, jd_text, cv_file, model, out_base],
        outputs=[md_preview, json_view, status_box, json_file, md_file],
        show_progress=True,
        api_name="generate",
    )

if __name__ == "__main__":
    demo.launch()
