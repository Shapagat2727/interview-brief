from typing import Any, Dict


def render_markdown(data: Dict[str, Any]) -> str:
    def bullets(key: str) -> str:
        items = data.get(key) or []
        if isinstance(items, str):
            items = [items]
        return "\n".join(f"- {x}" for x in items)

    md = []
    md.append("# Interview Prep Brief\n")
    if "role_summary" in data:
        md.append("## Role Summary\n")
        rs = data["role_summary"]
        md.append("\n".join(rs) + "\n" if isinstance(rs, list) else f"{rs}\n")

    for section, title in [
        ("top_required_skills", "Top Required Skills"),
        ("strong_overlaps", "Strong Overlaps (CV → JD)"),
        ("gaps_and_risks", "Gaps & Risks"),
        ("likely_tech_questions", "Likely Technical Questions"),
        ("behavioral_questions", "Behavioral Questions"),
        ("talking_points", "High-Impact Talking Points"),
        ("quick_upskilling_plan", "Quick Upskilling Plan (≤2h Tasks)"),
    ]:
        if data.get(section):
            md.append(f"## {title}\n")
            md.append(bullets(section) + "\n")

    return "\n".join(md).strip() + "\n"
