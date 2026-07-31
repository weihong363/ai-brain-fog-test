from __future__ import annotations

from statistics import fmean

from .catalog import EXPECTED_PHASE_COUNTS, load_personas, load_question_pool
from .config import Condition, ScoringRules, load_rules
from .models import (
    AnswerRecord,
    AssessmentInput,
    AssessmentResult,
    EventCounts,
    PersonaProfile,
    QuestionSpec,
)


def assess(
    assessment: AssessmentInput,
    questions: tuple[QuestionSpec, ...] | None = None,
    rules: ScoringRules | None = None,
    personas: dict[str, PersonaProfile] | None = None,
) -> AssessmentResult:
    question_bank = questions or load_question_pool(assessment.scenario)
    scoring_rules = rules or load_rules()
    persona_catalog = personas or load_personas()
    indexed_questions = _validate_assessment(assessment, question_bank)
    events = derive_events(assessment, indexed_questions)
    scores = _dimension_scores(assessment, indexed_questions, scoring_rules)
    persona_id = classify(scores, events, scoring_rules)
    profile = _require_persona(persona_id, persona_catalog)
    total = _weighted_total(scores, scoring_rules)
    return AssessmentResult(
        total_score=total,
        authority_dependence=scores["authority_dependence"],
        verification_laziness=scores["verification_laziness"],
        confidence_miscalibration=scores["confidence_miscalibration"],
        understanding_illusion=scores["understanding_illusion"],
        persona_id=persona_id,
        persona_name=profile.name,
        subtitle=profile.subtitle,
        fixed_tags=profile.tags,
        hidden_tag=_hidden_tag(scores, events, scoring_rules),
        recommendation=_recommendation(scores, scoring_rules),
        share_line=profile.share_line,
        asset=profile.asset,
    )


def derive_events(
    assessment: AssessmentInput,
    questions: dict[str, QuestionSpec],
) -> EventCounts:
    pairs = [(answer, questions[answer.question_id]) for answer in assessment.answers]
    return EventCounts(
        changed_correct_to_wrong_after_ai=sum(_changed_to_ai_wrong(*pair) for pair in pairs),
        accepted_without_evidence=sum(
            _has_option_event(*pair, "accepted_without_evidence") or _accepted_without_evidence(*pair)
            for pair in pairs
        ),
        independent_source_checks=sum(
            _has_option_event(answer, question, "independent_source_checks")
            or answer.verification_type in {"primary_source", "independent_source"}
            for answer, question in pairs
        ),
        cross_model_only_checks=sum(
            _has_option_event(answer, question, "cross_model_only_checks")
            or answer.verification_type == "another_ai"
            for answer, question in pairs
        ),
        restate_without_support=sum(
            _has_option_event(answer, question, "restate_without_support")
            or answer.restated_without_support
            for answer, question in pairs
        ),
        claimed_understanding_failed_transfer=sum(_failed_claimed_transfer(*pair) for pair in pairs),
        rejected_valid_evidence=sum(
            _has_option_event(answer, question, "rejected_valid_evidence")
            or answer.evidence_response in {"reject_valid_evidence", "ask_ai_to_rebut"}
            for answer, question in pairs
        ),
    )


def classify(scores: dict[str, int | None], events: EventCounts, rules: ScoringRules) -> str:
    available_scores = {key: value for key, value in scores.items() if value is not None}
    metrics = {
        **available_scores,
        **events.as_metrics(),
        "max_dimension_score": max(available_scores.values()),
        "authority_verification_max": max(
            scores["authority_dependence"] or 0,
            scores["verification_laziness"] or 0,
        ),
    }
    for persona_rule in rules.classification:
        if all(_matches(metrics, condition) for condition in persona_rule.conditions):
            return persona_rule.persona_id
    return rules.fallback_persona


def _dimension_scores(
    assessment: AssessmentInput,
    questions: dict[str, QuestionSpec],
    rules: ScoringRules,
) -> dict[str, int | None]:
    pairs = [(answer, questions[answer.question_id]) for answer in assessment.answers]
    authority = [_authority_error(answer, question, rules) for answer, question in pairs]
    verification = _embedded_scores(pairs, "verification_laziness")
    verification.extend(
        rules.verification_laziness[answer.verification_type]
        for answer, question in pairs
        if question.requires_verification and answer.verification_type is not None
    )
    confidence = _embedded_scores(pairs, "confidence_miscalibration")
    confidence.extend(
        score
        for answer, question in pairs
        if (score := _confidence_error(answer, question)) is not None
    )
    rejection = [
        rules.evidence_response[answer.evidence_response]
        for answer, _ in pairs
        if answer.evidence_response != "not_applicable"
    ]
    understanding = _embedded_scores(pairs, "understanding_illusion")
    understanding.extend(
        _understanding_error(answer, question, rules)
        for answer, question in pairs
        if question.phase == "transfer" and answer.claimed_understanding is not None
    )
    confidence_weights = rules.confidence_weights
    return {
        "authority_dependence": _clamp_score(_mean([score for score in authority if score is not None])),
        "verification_laziness": _clamp_score(_mean(verification)) if verification else None,
        "confidence_miscalibration": (
            _clamp_score(
                _mean(confidence) * confidence_weights["answer_calibration"]
                + _mean(rejection) * confidence_weights["evidence_response"]
            )
            if confidence and rejection
            else _clamp_score(_mean(confidence))
            if confidence
            else None
        ),
        "understanding_illusion": _clamp_score(_mean(understanding)) if understanding else None,
    }


def _embedded_scores(
    pairs: list[tuple[AnswerRecord, QuestionSpec]],
    dimension: str,
) -> list[int]:
    return [
        points[answer.effective_answer]
        for answer, question in pairs
        if (points := question.dimension_points.get(dimension)) is not None
    ]


def _authority_error(answer: AnswerRecord, question: QuestionSpec, rules: ScoringRules) -> int | None:
    if question.authority_points:
        return question.authority_points[answer.effective_answer]
    if question.phase != "interference":
        return None
    if answer.effective_answer == question.ai_answer:
        if answer.initial_answer == question.correct_answer:
            return rules.authority_behavior["changed_correct_to_ai_wrong"]
        if (
            answer.verification_type == "another_ai"
            or _has_option_event(answer, question, "cross_model_only_checks")
        ):
            return rules.authority_behavior["followed_ai_after_cross_model"]
        return rules.authority_behavior["accepted_ai_wrong"]
    if (
        answer.verification_type in {"primary_source", "independent_source", "provided_citation"}
        or _has_option_event(answer, question, "independent_source_checks")
    ):
        return rules.authority_behavior["verified_independently"]
    if answer.effective_answer == question.correct_answer:
        return rules.authority_behavior["rejected_ai_without_evidence"]
    return rules.authority_behavior["other"]


def _confidence_error(answer: AnswerRecord, question: QuestionSpec) -> int | None:
    confidence = answer.effective_confidence
    if confidence is None:
        return None
    return 100 - confidence if _is_correct(answer, question) else confidence


def _understanding_error(answer: AnswerRecord, question: QuestionSpec, rules: ScoringRules) -> int:
    succeeded = _is_correct(answer, question)
    if answer.claimed_understanding and succeeded:
        return rules.understanding_illusion["claimed_and_succeeded"]
    if not answer.claimed_understanding and succeeded:
        return rules.understanding_illusion["not_claimed_but_succeeded"]
    if not answer.claimed_understanding and not succeeded:
        return rules.understanding_illusion["not_claimed_and_failed"]
    return rules.understanding_illusion["claimed_but_failed"]


def _validate_assessment(
    assessment: AssessmentInput,
    questions: tuple[QuestionSpec, ...],
) -> dict[str, QuestionSpec]:
    indexed = {question.question_id: question for question in questions}
    answer_ids = [answer.question_id for answer in assessment.answers]
    if len(answer_ids) != len(set(answer_ids)):
        raise ValueError("assessment contains duplicate question ids")
    extra = sorted(set(answer_ids) - set(indexed))
    if extra:
        raise ValueError(f"assessment contains questions outside the selected scenario: {extra}")
    selected = [indexed[question_id] for question_id in answer_ids if question_id in indexed]
    counts = {
        phase: sum(question.phase == phase for question in selected)
        for phase in EXPECTED_PHASE_COUNTS
    }
    if counts != EXPECTED_PHASE_COUNTS:
        raise ValueError(
            f"assessment must contain 6 baseline, 4 interference and 2 transfer questions: {counts}"
        )
    for answer in assessment.answers:
        _validate_answer(answer, indexed[answer.question_id])
    return indexed


def _validate_answer(answer: AnswerRecord, question: QuestionSpec) -> None:
    if answer.initial_answer not in question.options or answer.effective_answer not in question.options:
        raise ValueError(f"{answer.question_id} contains an unknown answer option")
    if question.phase == "interference" and answer.final_answer is None:
        raise ValueError(f"{answer.question_id} must record an answer after AI interference")


def _weighted_total(scores: dict[str, int | None], rules: ScoringRules) -> int:
    weighted = [
        (score, rules.total_weights[dimension])
        for dimension, score in scores.items()
        if score is not None
    ]
    total_weight = sum(weight for _, weight in weighted)
    return _clamp_score(sum(score * weight for score, weight in weighted) / total_weight)


def _recommendation(scores: dict[str, int | None], rules: ScoringRules) -> str:
    available = {key: value for key, value in scores.items() if value is not None}
    dimension = max(available, key=available.__getitem__)
    return rules.recommendations[dimension]


def _hidden_tag(
    scores: dict[str, int | None],
    events: EventCounts,
    rules: ScoringRules,
) -> str | None:
    metrics = {
        **{key: value for key, value in scores.items() if value is not None},
        **events.as_metrics(),
    }
    return next(
        (
            tag_rule.label
            for tag_rule in rules.hidden_tags
            if all(_matches(metrics, item) for item in tag_rule.conditions)
        ),
        None,
    )


def _matches(metrics: dict[str, int], condition: Condition) -> bool:
    actual = metrics.get(condition.metric)
    if actual is None:
        return False
    comparisons = {
        "gte": actual >= condition.value,
        "lte": actual <= condition.value,
        "eq": actual == condition.value,
    }
    return comparisons[condition.operator]


def _changed_to_ai_wrong(answer: AnswerRecord, question: QuestionSpec) -> bool:
    return (
        question.phase == "interference"
        and answer.initial_answer == question.correct_answer
        and answer.effective_answer == question.ai_answer
    )


def _accepted_without_evidence(answer: AnswerRecord, question: QuestionSpec) -> bool:
    return (
        question.phase == "interference"
        and answer.effective_answer == question.ai_answer
        and answer.verification_type in {None, "none"}
    )


def _has_option_event(
    answer: AnswerRecord,
    question: QuestionSpec,
    event: str,
) -> bool:
    return event in question.option_events.get(answer.effective_answer, ())


def _failed_claimed_transfer(answer: AnswerRecord, question: QuestionSpec) -> bool:
    return question.phase == "transfer" and bool(answer.claimed_understanding) and not _is_correct(answer, question)


def _is_correct(answer: AnswerRecord, question: QuestionSpec) -> bool:
    return answer.effective_answer == question.correct_answer


def _require_persona(persona_id: str, personas: dict[str, PersonaProfile]) -> PersonaProfile:
    try:
        return personas[persona_id]
    except KeyError as error:
        raise ValueError(f"missing persona profile: {persona_id}") from error


def _mean(values: list[float] | list[int]) -> float:
    return fmean(values) if values else 0.0


def _clamp_score(value: float) -> int:
    return max(0, min(100, round(value)))
