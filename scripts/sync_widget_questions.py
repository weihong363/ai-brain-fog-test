from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_brain_fog.catalog import QUESTION_OPTION_EVENTS, QUESTION_SLOTS  # noqa: E402

WIDGET = ROOT / "assets" / "assessment-widget.html"
QUESTIONS = ROOT / "data" / "questions.json"
RULES = ROOT / "rules" / "scoring.json"
PERSONAS = ROOT / "data" / "personas.json"
START = "    const questionPools = "
END = "    const state = {"


def main() -> None:
    html = WIDGET.read_text(encoding="utf-8")
    questions = _load_json(QUESTIONS)["scenarios"]
    rules = _load_json(RULES)
    personas = _load_json(PERSONAS)
    values = {
        "questionPools": questions,
        "questionPhases": [phase for phase, _, _ in QUESTION_SLOTS],
        "questionDimensions": [dimension for _, dimension, _ in QUESTION_SLOTS],
        "questionOptionEvents": QUESTION_OPTION_EVENTS,
        "scoringRules": rules,
        "personaProfiles": personas,
        "personaAssets": _persona_assets(personas),
    }
    generated = "\n\n".join(
        f"    const {name} = {_compact_json(value)};"
        for name, value in values.items()
    )
    generated += "\n\n"
    start = html.index(START)
    end = html.index(END, start)
    WIDGET.write_text(html[:start] + generated + html[end:], encoding="utf-8")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _persona_assets(personas: dict[str, object]) -> dict[str, str]:
    return {
        persona_id: _data_uri(ROOT / str(profile["asset"]))  # type: ignore[index]
        for persona_id, profile in personas.items()
    }


def _data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/webp;base64,{encoded}"


if __name__ == "__main__":
    main()
