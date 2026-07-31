from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from .models import ANSWER_EVENTS, SCENARIOS, AnswerEvent, PersonaProfile, QuestionSpec

EXPECTED_PHASE_COUNTS = {"baseline": 6, "interference": 4, "transfer": 2}
POOL_PHASE_COUNTS = {"baseline": 10, "interference": 6, "transfer": 4}
DIMENSIONS = {
    "authority_dependence",
    "verification_laziness",
    "confidence_miscalibration",
    "understanding_illusion",
}
QUESTION_TYPES = {
    "source_evaluation",
    "causal_reasoning",
    "credibility_assessment",
    "authority_resistance",
    "formal_logic",
    "uncertainty_handling",
    "argument_audit",
    "citation_audit",
    "code_reasoning",
    "sampling_bias",
    "condition_transfer",
    "evidence_update",
}
QUESTION_SLOTS = (
    ("baseline", "verification_laziness", "source_evaluation"),
    ("baseline", "confidence_miscalibration", "causal_reasoning"),
    ("baseline", "verification_laziness", "credibility_assessment"),
    ("baseline", "authority_dependence", "authority_resistance"),
    ("baseline", "confidence_miscalibration", "formal_logic"),
    ("baseline", "authority_dependence", "uncertainty_handling"),
    ("baseline", "verification_laziness", "citation_audit"),
    ("baseline", "confidence_miscalibration", "sampling_bias"),
    ("baseline", "verification_laziness", "source_evaluation"),
    ("baseline", "confidence_miscalibration", "argument_audit"),
    ("interference", "authority_dependence", "argument_audit"),
    ("interference", "verification_laziness", "citation_audit"),
    ("interference", "confidence_miscalibration", "formal_logic"),
    ("interference", "authority_dependence", "causal_reasoning"),
    ("interference", "verification_laziness", "credibility_assessment"),
    ("interference", "confidence_miscalibration", "sampling_bias"),
    ("transfer", "understanding_illusion", "condition_transfer"),
    ("transfer", "understanding_illusion", "evidence_update"),
    ("transfer", "verification_laziness", "source_evaluation"),
    ("transfer", "confidence_miscalibration", "formal_logic"),
)
QUESTION_OPTION_EVENTS: tuple[dict[str, tuple[AnswerEvent, ...]], ...] = (
    {
        "A": ("accepted_without_evidence",),
        "B": ("cross_model_only_checks",),
        "C": ("independent_source_checks",),
    },
    {},
    {
        "A": ("accepted_without_evidence",),
        "B": ("accepted_without_evidence",),
        "C": ("independent_source_checks",),
        "D": ("rejected_valid_evidence",),
    },
    {
        "A": ("accepted_without_evidence",),
        "B": ("cross_model_only_checks",),
        "C": ("independent_source_checks",),
    },
    {},
    {
        "A": ("accepted_without_evidence",),
        "B": ("restate_without_support",),
        "D": ("independent_source_checks",),
    },
    {
        "A": ("accepted_without_evidence",),
        "C": ("independent_source_checks",),
        "D": ("cross_model_only_checks",),
    },
    {},
    {
        "A": ("accepted_without_evidence", "restate_without_support"),
        "C": ("independent_source_checks",),
    },
    {},
    {},
    {
        "A": ("accepted_without_evidence",),
        "B": ("rejected_valid_evidence",),
        "C": ("independent_source_checks",),
        "D": ("cross_model_only_checks",),
    },
    {},
    {},
    {
        "A": ("accepted_without_evidence",),
        "B": ("restate_without_support",),
        "C": ("independent_source_checks",),
    },
    {},
    {},
    {"C": ("independent_source_checks",)},
    {"C": ("independent_source_checks",)},
    {},
)
POINTS_BY_CORRECT = {
    "A": (0, 80, 70, 95),
    "B": (90, 0, 75, 85),
    "C": (95, 75, 0, 85),
    "D": (85, 70, 95, 0),
}


def default_questions_path() -> Path:
    return _data_path("questions.json")


def default_personas_path() -> Path:
    return _data_path("personas.json")


def load_question_pool(
    scenario: str | None = None,
    path: Path | None = None,
) -> tuple[QuestionSpec, ...]:
    raw = json.loads((path or default_questions_path()).read_text(encoding="utf-8"))
    questions = tuple(
        _question(scenario_key, item, index)
        for scenario_key, items in raw["scenarios"].items()
        for index, item in enumerate(items)
        if scenario is None or scenario_key == scenario
    )
    _validate_question_pool(questions, scenario)
    return questions


def load_question_bank(
    scenario: str = "mixed",
    seed: int | None = 0,
    path: Path | None = None,
) -> tuple[QuestionSpec, ...]:
    pool = load_question_pool(scenario, path)
    generator = random.Random(seed)
    return tuple(
        question
        for phase in EXPECTED_PHASE_COUNTS
        for question in generator.sample(
            [item for item in pool if item.phase == phase],
            EXPECTED_PHASE_COUNTS[phase],
        )
    )


def load_personas(path: Path | None = None) -> dict[str, PersonaProfile]:
    raw = json.loads((path or default_personas_path()).read_text(encoding="utf-8"))
    return {persona_id: _persona(persona_id, item) for persona_id, item in raw.items()}


def _question(scenario: str, raw: dict[str, object], index: int) -> QuestionSpec:
    phase, dimension, question_type = QUESTION_SLOTS[index]
    options = {
        key: str(value)
        for key, value in zip(("A", "B", "C", "D"), raw["options"], strict=True)  # type: ignore[arg-type]
    }
    points = {
        key: int(value)
        for key, value in zip(
            ("A", "B", "C", "D"),
            POINTS_BY_CORRECT[str(raw["correct_answer"])],
            strict=True,
        )
    }
    return QuestionSpec(
        question_id=str(raw["id"]),
        scenario=scenario,  # type: ignore[arg-type]
        phase=phase,  # type: ignore[arg-type]
        dimension=dimension,  # type: ignore[arg-type]
        question_type=question_type,  # type: ignore[arg-type]
        prompt_template=str(raw["prompt"]),
        options=options,
        correct_answer=str(raw["correct_answer"]),
        explanation=str(raw["explanation"]),
        authority_points=points if dimension == "authority_dependence" else {},
        dimension_points={} if dimension == "authority_dependence" else {dimension: points},  # type: ignore[dict-item]
        option_events=QUESTION_OPTION_EVENTS[index],
        requires_verification=dimension == "verification_laziness",
        ai_answer=str(raw["ai_answer"]) if raw.get("ai_answer") is not None else None,
        ai_message=str(raw["ai_message"]) if raw.get("ai_message") is not None else None,
    )


def _persona(persona_id: str, raw: dict[str, object]) -> PersonaProfile:
    tags = tuple(str(value) for value in raw["tags"])  # type: ignore[union-attr]
    if len(tags) != 3:
        raise ValueError(f"persona {persona_id} must define exactly three tags")
    return PersonaProfile(
        persona_id=persona_id,
        name=str(raw["name"]),
        subtitle=str(raw["subtitle"]),
        tags=tags,  # type: ignore[arg-type]
        share_line=str(raw["share_line"]),
        asset=str(raw["asset"]),
    )


def _validate_question_pool(
    questions: tuple[QuestionSpec, ...],
    scenario: str | None,
) -> None:
    ids = [question.question_id for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("question bank contains duplicate ids")
    scenarios = SCENARIOS if scenario is None else {scenario}
    for scenario_key in scenarios:
        subset = [question for question in questions if question.scenario == scenario_key]
        counts = {
            phase: sum(question.phase == phase for question in subset)
            for phase in POOL_PHASE_COUNTS
        }
        if counts != POOL_PHASE_COUNTS:
            raise ValueError(f"{scenario_key} pool must contain 10 baseline, 6 interference and 4 transfer: {counts}")
        if len({question.question_type for question in subset}) < 8:
            raise ValueError(f"{scenario_key} pool must contain at least eight distinct question types")
    for question in questions:
        _validate_question(question)


def _validate_question(question: QuestionSpec) -> None:
    if question.dimension not in DIMENSIONS:
        raise ValueError(f"{question.question_id} has an invalid dimension")
    if question.question_type not in QUESTION_TYPES:
        raise ValueError(f"{question.question_id} has an invalid question_type")
    if not question.options:
        raise ValueError(f"{question.question_id} must define answer options")
    if question.correct_answer not in question.options:
        raise ValueError(f"{question.question_id} has an invalid correct_answer")
    if set(question.authority_points) - set(question.options):
        raise ValueError(f"{question.question_id} has authority scores for unknown options")
    if any(not 0 <= score <= 100 for score in question.authority_points.values()):
        raise ValueError(f"{question.question_id} authority scores must be between 0 and 100")
    for dimension, points in question.dimension_points.items():
        if dimension not in DIMENSIONS:
            raise ValueError(f"{question.question_id} has scores for an invalid dimension")
        if set(points) != set(question.options):
            raise ValueError(f"{question.question_id} must score every option for {dimension}")
        if any(not 0 <= score <= 100 for score in points.values()):
            raise ValueError(f"{question.question_id} {dimension} scores must be between 0 and 100")
    if set(question.option_events) - set(question.options):
        raise ValueError(f"{question.question_id} has events for unknown options")
    if any(event not in ANSWER_EVENTS for events in question.option_events.values() for event in events):
        raise ValueError(f"{question.question_id} has an unsupported option event")
    if question.phase == "interference":
        if question.ai_answer not in question.options or question.ai_answer == question.correct_answer:
            raise ValueError(f"{question.question_id} must define an incorrect AI answer")
        if not question.ai_message:
            raise ValueError(f"{question.question_id} must define an AI message")


def _data_path(filename: str) -> Path:
    source_path = Path(__file__).resolve().parents[2] / "data" / filename
    return source_path if source_path.exists() else Path(sys.prefix) / "share" / "ai-brain-fog-test" / filename
