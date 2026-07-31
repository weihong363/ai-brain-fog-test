import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_brain_fog import assess, load_question_bank
from ai_brain_fog.session import format_result, load_assessment

questions = load_question_bank()
assessment = load_assessment(ROOT / "data" / "responses.demo.json")
print(format_result(assess(assessment, questions=questions)))
