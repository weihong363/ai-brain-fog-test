from dataclasses import replace
from pathlib import Path

import pytest

from ai_brain_fog import (
    AnswerRecord,
    AssessmentInput,
    EventCounts,
    assess,
    classify,
    derive_events,
    load_question_bank,
    load_question_pool,
    load_rules,
)
from ai_brain_fog.session import load_assessment

ROOT = Path(__file__).resolve().parents[1]
CLICK_ONLY_QUESTION_IDS = (
    "mixed_b01",
    "mixed_b02",
    "mixed_b04",
    "mixed_b06",
    "mixed_b07",
    "mixed_b09",
    "mixed_i01",
    "mixed_i02",
    "mixed_i03",
    "mixed_i05",
    "mixed_t01",
    "mixed_t02",
)
HIDDEN_TAG_QUESTION_IDS = (
    "mixed_b01",
    "mixed_b02",
    "mixed_b03",
    "mixed_b04",
    "mixed_b06",
    "mixed_b09",
    "mixed_i01",
    "mixed_i02",
    "mixed_i03",
    "mixed_i05",
    "mixed_t01",
    "mixed_t02",
)


def _scores(
    *,
    authority: int = 20,
    verification: int = 20,
    confidence: int = 20,
    understanding: int = 20,
) -> dict[str, int]:
    return {
        "authority_dependence": authority,
        "verification_laziness": verification,
        "confidence_miscalibration": confidence,
        "understanding_illusion": understanding,
    }


@pytest.mark.parametrize(
    ("expected", "scores", "events"),
    [
        (
            "ai_authorized_know_it_all",
            _scores(authority=75, confidence=75),
            EventCounts(
                changed_correct_to_wrong_after_ai=2,
            ),
        ),
        ("multi_model_voter", _scores(), EventCounts(cross_model_only_checks=2)),
        ("silicon_spokesperson", _scores(authority=75), EventCounts()),
        (
            "fog_operator",
            _scores(understanding=75),
            EventCounts(claimed_understanding_failed_transfer=2),
        ),
        ("high_speed_repeater", _scores(verification=55), EventCounts(restate_without_support=2)),
        (
            "half_understood_master",
            _scores(understanding=60),
            EventCounts(claimed_understanding_failed_transfer=1),
        ),
        ("clear_tool_user", _scores(), EventCounts()),
        ("cautious_cyber_apprentice", _scores(authority=35), EventCounts()),
    ],
)
def test_all_personas_are_reachable(
    expected: str,
    scores: dict[str, int],
    events: EventCounts,
) -> None:
    assert classify(scores, events, load_rules()) == expected


def test_persona_priority_prefers_ai_authorized_know_it_all() -> None:
    events = EventCounts(
        changed_correct_to_wrong_after_ai=2,
        cross_model_only_checks=3,
    )
    assert classify(_scores(authority=80, confidence=80), events, load_rules()) == "ai_authorized_know_it_all"


def test_demo_assessment_derives_events_and_complete_result() -> None:
    questions = load_question_pool("mixed")
    assessment = load_assessment(ROOT / "data" / "responses.demo.json")
    indexed = {question.question_id: question for question in questions}

    events = derive_events(assessment, indexed)
    result = assess(assessment, questions=questions)

    assert events.changed_correct_to_wrong_after_ai == 2
    assert events.accepted_without_evidence == 2
    assert events.independent_source_checks == 5
    assert events.rejected_valid_evidence == 0
    assert result.hidden_tag == "AI反向改选择"
    assert len(result.fixed_tags) == 3
    assert result.recommendation
    assert result.verification_laziness is not None
    assert result.confidence_miscalibration is not None
    assert result.understanding_illusion is not None


@pytest.mark.parametrize(
    ("expected_persona", "selections", "expected_event", "minimum_count"),
    [
        (
            "multi_model_voter",
            ("B", "B", "B", "C", "D", "D", "B", "D", "B", "D", "B", "A"),
            "cross_model_only_checks",
            2,
        ),
        (
            "high_speed_repeater",
            ("D", "B", "D", "B", "B", "A", "B", "B", "B", "B", "A", "C"),
            "restate_without_support",
            2,
        ),
    ],
)
def test_click_only_answers_reach_behavior_personas(
    expected_persona: str,
    selections: tuple[str, ...],
    expected_event: str,
    minimum_count: int,
) -> None:
    pool = load_question_pool("mixed")
    indexed = {question.question_id: question for question in pool}
    questions = tuple(indexed[question_id] for question_id in CLICK_ONLY_QUESTION_IDS)
    answers = tuple(
        AnswerRecord(
            question_id=question_id,
            initial_answer=selection,
            final_answer=selection if "_i" in question_id else None,
        )
        for question_id, selection in zip(CLICK_ONLY_QUESTION_IDS, selections, strict=True)
    )
    assessment = AssessmentInput(scenario="mixed", answers=answers)
    events = derive_events(assessment, indexed)

    assert getattr(events, expected_event) >= minimum_count
    assert assess(assessment, questions=questions).persona_id == expected_persona


@pytest.mark.parametrize(
    ("expected_persona", "encoded_answers"),
    [
        ("ai_authorized_know_it_all", "C A D A D C BD AC BA CA C B"),
        ("cautious_cyber_apprentice", "D B C B B C DD CB BD AA A B"),
        ("clear_tool_user", "C B C A C C CB AD BB BC A C"),
        ("fog_operator", "D B D C A B DC CB BC AA D A"),
        ("half_understood_master", "C A B C C D AB CB DA BB C B"),
        ("high_speed_repeater", "A A C B B B AA DA AA BB A B"),
        ("multi_model_voter", "B D A C D D DB DA BC AD A D"),
        ("silicon_spokesperson", "A C B A C B BC BA CD AA C C"),
    ],
)
def test_click_only_answers_can_reach_all_personas(
    expected_persona: str,
    encoded_answers: str,
) -> None:
    pool = load_question_pool("mixed")
    indexed = {question.question_id: question for question in pool}
    questions = tuple(indexed[question_id] for question_id in CLICK_ONLY_QUESTION_IDS)
    answers = tuple(
        AnswerRecord(
            question_id=question_id,
            initial_answer=encoded[0],
            final_answer=encoded[1] if len(encoded) == 2 else None,
        )
        for question_id, encoded in zip(
            CLICK_ONLY_QUESTION_IDS,
            encoded_answers.split(),
            strict=True,
        )
    )

    result = assess(AssessmentInput(scenario="mixed", answers=answers), questions=questions)

    assert result.persona_id == expected_persona


@pytest.mark.parametrize(
    ("expected_tag", "encoded_answers"),
    [
        ("AI会议召集人", "B C B C B D BD BD AD CC C C"),
        ("AI反向改选择", "D C A D A D BA AD BC CA D D"),
        ("免验证通行", "C A B A D C CB BD AB BA B C"),
        ("理解感超载", "C D A D A A BC BC DB BA D B"),
        ("解释复读", "C B B A B A CB DA DC BC B B"),
        ("证据免疫", "D C D D A B BD AC CC BC B A"),
    ],
)
def test_click_only_answers_can_reach_all_hidden_tags(
    expected_tag: str,
    encoded_answers: str,
) -> None:
    pool = load_question_pool("mixed")
    indexed = {question.question_id: question for question in pool}
    questions = tuple(indexed[question_id] for question_id in HIDDEN_TAG_QUESTION_IDS)
    answers = tuple(
        AnswerRecord(
            question_id=question_id,
            initial_answer=encoded[0],
            final_answer=encoded[1] if len(encoded) == 2 else None,
        )
        for question_id, encoded in zip(
            HIDDEN_TAG_QUESTION_IDS,
            encoded_answers.split(),
            strict=True,
        )
    )

    result = assess(AssessmentInput(scenario="mixed", answers=answers), questions=questions)

    assert result.hidden_tag == expected_tag


def test_embedded_option_signals_score_all_three_dimensions() -> None:
    pool = load_question_pool("mixed")
    assessment = load_assessment(ROOT / "data" / "responses.demo.json")
    indexed = {question.question_id: question for question in pool}
    questions = tuple(indexed[answer.question_id] for answer in assessment.answers)

    def scored_assessment(use_highest: bool) -> AssessmentInput:
        answers = []
        for answer, question in zip(assessment.answers, questions, strict=True):
            if use_highest and question.dimension_points:
                selected = max(
                    question.options,
                    key=lambda option: sum(
                        points[option] for points in question.dimension_points.values()
                    ),
                )
            else:
                selected = question.correct_answer
            answers.append(
                replace(
                    answer,
                    initial_answer=selected,
                    final_answer=selected if question.phase == "interference" else None,
                    evidence_response="not_applicable",
                    claimed_understanding=None,
                )
            )
        return replace(assessment, answers=tuple(answers))

    low = assess(scored_assessment(False), questions=questions)
    high = assess(scored_assessment(True), questions=questions)

    assert low.verification_laziness < high.verification_laziness
    assert low.confidence_miscalibration < high.confidence_miscalibration
    assert low.understanding_illusion < high.understanding_illusion


def test_assessment_requires_complete_unique_question_set() -> None:
    questions = load_question_pool("mixed")
    assessment = load_assessment(ROOT / "data" / "responses.demo.json")

    with pytest.raises(ValueError, match="must contain 6 baseline"):
        assess(replace(assessment, answers=assessment.answers[:-1]), questions=questions)

    duplicated = assessment.answers[:-1] + (assessment.answers[0],)
    with pytest.raises(ValueError, match="duplicate question ids"):
        assess(replace(assessment, answers=duplicated), questions=questions)


def test_interference_requires_final_answer() -> None:
    questions = load_question_pool("mixed")
    assessment = load_assessment(ROOT / "data" / "responses.demo.json")
    target = next(
        index
        for index, answer in enumerate(assessment.answers)
        if "_i" in answer.question_id
    )
    invalid = replace(assessment.answers[target], final_answer=None, final_confidence=None)
    answers = assessment.answers[:target] + (invalid,) + assessment.answers[target + 1 :]

    with pytest.raises(ValueError, match="after AI interference"):
        assess(replace(assessment, answers=answers), questions=questions)


def test_assessment_rejects_unknown_answer_options() -> None:
    questions = load_question_pool("mixed")
    assessment = load_assessment(ROOT / "data" / "responses.demo.json")
    invalid = replace(assessment.answers[0], initial_answer="Z")
    answers = (invalid,) + assessment.answers[1:]

    with pytest.raises(ValueError, match="unknown answer option"):
        assess(replace(assessment, answers=answers), questions=questions)


def test_answer_record_validates_runtime_values() -> None:
    with pytest.raises(ValueError, match="requires final_answer"):
        AnswerRecord("q", "A", final_confidence=50)
    with pytest.raises(ValueError, match="unsupported verification_type"):
        AnswerRecord("q", "A", verification_type="web_search")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer between"):
        AnswerRecord("q", "A", True)  # type: ignore[arg-type]


def test_event_counts_reject_negative_values() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        EventCounts(accepted_without_evidence=-1)
