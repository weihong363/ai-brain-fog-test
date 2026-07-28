from __future__ import annotations

from statistics import fmean

from .models import AnswerRecord, AssessmentInput, AssessmentResult

VERIFICATION_LAZINESS = {
    "primary_source": 0,
    "independent_source": 10,
    "provided_citation": 25,
    "another_ai": 70,
    "same_ai_again": 85,
    "none": 100,
}

EVIDENCE_REJECTION = {
    "change_after_evidence": 0,
    "remain_uncertain": 20,
    "reject_valid_evidence": 75,
    "ask_ai_to_rebut": 100,
    "not_applicable": 0,
}


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _clamp_score(value: float) -> int:
    return max(0, min(100, round(value)))


def confidence_error(answer: AnswerRecord) -> int:
    return 100 - answer.confidence if answer.is_correct else answer.confidence


def understanding_error(answer: AnswerRecord) -> int | None:
    if answer.transfer_succeeded is None:
        return None
    if answer.claimed_understanding and answer.transfer_succeeded:
        return 0
    if not answer.claimed_understanding and answer.transfer_succeeded:
        return 10
    if not answer.claimed_understanding and not answer.transfer_succeeded:
        return 30
    return 100


def classify(
    authority: int,
    verification: int,
    confidence: int,
    understanding: int,
    assessment: AssessmentInput,
) -> str:
    events = assessment.events

    if authority >= 75 and confidence >= 75 and events.changed_correct_to_wrong_after_ai >= 2:
        return "ai_authorized_know_it_all"

    if events.cross_model_only_checks >= 2 and events.independent_source_checks == 0:
        return "multi_model_voter"

    if authority >= 75:
        return "silicon_spokesperson"

    if understanding >= 75 and events.claimed_understanding_failed_transfer >= 2:
        return "fog_operator"

    if events.restate_without_support >= 2 and (authority >= 55 or verification >= 55):
        return "high_speed_repeater"

    if understanding >= 60 and events.claimed_understanding_failed_transfer >= 1:
        return "half_understood_master"

    if max(authority, verification, confidence, understanding) <= 30 and events.independent_source_checks >= 2:
        return "clear_tool_user"

    return "cautious_cyber_apprentice"


def build_evidence(assessment: AssessmentInput) -> tuple[str, ...]:
    events = assessment.events
    candidates: list[tuple[int, str]] = [
        (events.changed_correct_to_wrong_after_ai * 5, f"{events.changed_correct_to_wrong_after_ai}次把正确答案改成AI的错误答案"),
        (events.accepted_without_evidence * 4, f"{events.accepted_without_evidence}次未看依据就接受结论"),
        (events.cross_model_only_checks * 3, f"{events.cross_model_only_checks}次只让另一个AI复核"),
        (events.restate_without_support * 3, f"{events.restate_without_support}次直接复述AI结论但无法补充证据"),
        (events.claimed_understanding_failed_transfer * 4, f"{events.claimed_understanding_failed_transfer}次声称掌握但迁移题失败"),
        (events.independent_source_checks * 2, f"{events.independent_source_checks}次主动检查独立原始材料"),
    ]
    nonzero = sorted((item for item in candidates if item[0] > 0), reverse=True)
    evidence = [text for _, text in nonzero[:2]]

    wrong_confidences = [a.confidence for a in assessment.answers if not a.is_correct]
    if wrong_confidences:
        evidence.append(f"错题平均置信度 {round(_mean(wrong_confidences))}%")

    return tuple(evidence[:3])


def assess(assessment: AssessmentInput) -> AssessmentResult:
    answers = list(assessment.answers)
    if not answers:
        raise ValueError("assessment must contain at least one answer")

    authority = _clamp_score(_mean([a.authority_points for a in answers]))
    verification = _clamp_score(_mean([VERIFICATION_LAZINESS[a.verification_type] for a in answers]))

    base_confidence = _mean([confidence_error(a) for a in answers])
    evidence_rejection = _mean([EVIDENCE_REJECTION[a.evidence_response] for a in answers])
    confidence = _clamp_score(base_confidence * 0.75 + evidence_rejection * 0.25)

    understanding_values = [score for answer in answers if (score := understanding_error(answer)) is not None]
    understanding = _clamp_score(_mean(understanding_values))

    total = _clamp_score(
        authority * 0.30
        + verification * 0.25
        + confidence * 0.30
        + understanding * 0.15
    )

    persona = classify(
        authority=authority,
        verification=verification,
        confidence=confidence,
        understanding=understanding,
        assessment=assessment,
    )

    return AssessmentResult(
        total_score=total,
        authority_dependence=authority,
        verification_laziness=verification,
        confidence_miscalibration=confidence,
        understanding_illusion=understanding,
        persona_id=persona,
        evidence=build_evidence(assessment),
    )
