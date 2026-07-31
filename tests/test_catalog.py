import json
from pathlib import Path

import pytest

from ai_brain_fog import load_personas, load_question_bank, load_question_pool, load_rules
from ai_brain_fog.catalog import EXPECTED_PHASE_COUNTS, POOL_PHASE_COUNTS
from ai_brain_fog.models import SCENARIOS

ROOT = Path(__file__).resolve().parents[1]


def test_question_bank_has_required_structure() -> None:
    pool = load_question_pool()

    assert len(pool) == 120
    assert all(question.correct_answer in question.options for question in pool)
    assert all(question.explanation for question in pool)
    for scenario in SCENARIOS:
        questions = load_question_pool(scenario)
        counts = {phase: sum(question.phase == phase for question in questions) for phase in POOL_PHASE_COUNTS}
        assert len(questions) == 20
        assert counts == POOL_PHASE_COUNTS
        assert len({question.question_type for question in questions}) >= 8
    embedded_dimensions = {
        dimension
        for question in pool
        for dimension in question.dimension_points
    }
    assert {
        "verification_laziness",
        "confidence_miscalibration",
        "understanding_illusion",
    } <= embedded_dimensions
    assert all(
        set(points) == set(question.options)
        for question in pool
        for points in question.dimension_points.values()
    )
    assert {
        "causal_reasoning",
        "credibility_assessment",
        "argument_audit",
        "citation_audit",
        "sampling_bias",
    } <= {question.question_type for question in pool}


def test_interference_questions_define_incorrect_ai_suggestions() -> None:
    interference = [question for question in load_question_pool() if question.phase == "interference"]

    assert all(question.ai_answer in question.options for question in interference)
    assert all(question.ai_answer != question.correct_answer for question in interference)
    assert all(question.ai_message for question in interference)


def test_interactive_widget_tracks_the_versioned_question_bank() -> None:
    html = (ROOT / "assets" / "assessment-widget.html").read_text(encoding="utf-8")

    assert all(question.question_id in html for question in load_question_pool())
    assert "是否认为自己已经理解" not in html
    assert "看到解释后的反应" not in html
    assert "assessLocally" in html
    assert "renderComplete" not in html
    assert "直接查看人格结果" not in html
    assert "question.explanation" not in html
    assert "pendingFinal" not in html
    assert "questionOptionEvents" in html
    assert 'id="bf-restart"' in html
    assert 'id="bf-share"' in html
    assert "createShareCardBlob" in html
    assert "#brain-fog-assessment [hidden]" in html
    assert "sendFollowUpMessage" not in html
    assert "ASSESSMENT_BASE64" not in html
    assert html.count("data:image/webp;base64,") == 8
    assert len(html.encode("utf-8")) < 2_000_000


def test_selection_draws_twelve_questions_with_phase_balance() -> None:
    first = load_question_bank("programming", seed=1)
    second = load_question_bank("programming", seed=2)

    assert len(first) == 12
    assert {phase: sum(question.phase == phase for question in first) for phase in EXPECTED_PHASE_COUNTS} == EXPECTED_PHASE_COUNTS
    assert {question.question_id for question in first} != {question.question_id for question in second}


def test_persona_and_rule_catalogs_are_complete() -> None:
    personas = load_personas()
    rules = load_rules()
    classified = {rule.persona_id for rule in rules.classification} | {rules.fallback_persona}

    assert len(personas) == 8
    assert classified == set(personas)
    assert all(len(persona.tags) == 3 for persona in personas.values())


def test_rule_weights_must_sum_to_one(tmp_path: Path) -> None:
    raw = json.loads((ROOT / "rules" / "scoring.json").read_text(encoding="utf-8"))
    raw["total_weights"]["authority_dependence"] = 0.31
    path = tmp_path / "invalid-rules.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="sum to 1"):
        load_rules(path)
