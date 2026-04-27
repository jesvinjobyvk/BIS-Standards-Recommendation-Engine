"""
LLM Rationale Generator
Calls Anthropic Claude API to generate concise rationale for why each retrieved
BIS standard applies to the given product description.
Falls back gracefully if API is unavailable.
"""

import os
import json
import time
from typing import List, Dict, Optional


def generate_rationale(
    query: str,
    retrieved_standards: List[Dict],
    api_key: Optional[str] = None,
    timeout: int = 10,
) -> List[Dict]:
    """
    Generate LLM rationale for each retrieved standard.
    Adds a 'rationale' key to each standard dict.
    Falls back to description-based rationale if API unavailable.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    if not key:
        # Fallback: use description excerpt as rationale
        return _fallback_rationale(query, retrieved_standards)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)

        standards_summary = "\n".join([
            f"- {s['code']}: {s['title']} (Category: {s['category']})"
            for s in retrieved_standards
        ])

        prompt = f"""You are a BIS (Bureau of Indian Standards) compliance expert for Building Materials (BIS SP 21).

Product Description: {query}

Retrieved Standards:
{standards_summary}

For each standard listed above, write a single concise sentence (max 25 words) explaining exactly why it applies to this product description. Be specific about the product attributes that trigger this standard.

Respond ONLY with a JSON array in this exact format (no markdown, no preamble):
[
  {{"code": "IS XXXX", "rationale": "One sentence explaining relevance."}},
  ...
]"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        rationale_list = json.loads(raw)

        # Map rationales back to standards by code
        rationale_map = {r["code"]: r["rationale"] for r in rationale_list}
        for std in retrieved_standards:
            std["rationale"] = rationale_map.get(
                std["code"],
                _description_snippet(std)
            )

        return retrieved_standards

    except Exception:
        # Fallback on any error
        return _fallback_rationale(query, retrieved_standards)


def _description_snippet(std: Dict) -> str:
    """Extract a short rationale from the standard description."""
    desc = std.get("description", "")
    sentences = desc.split(". ")
    return sentences[0][:200] if sentences else desc[:200]


def _fallback_rationale(query: str, standards: List[Dict]) -> List[Dict]:
    """Assign rationales from description when API is unavailable."""
    for std in standards:
        std["rationale"] = _description_snippet(std)
    return standards
