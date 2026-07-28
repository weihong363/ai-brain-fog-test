from ai_brain_fog import AnswerRecord, AssessmentInput, EventCounts, assess


def test_clear_tool_user() -> None:
    data = AssessmentInput(
        answers=(
            AnswerRecord(
                question_id="q1",
                is_correct=True,
                confidence=90,
                authority_points=10,
                verification_type="primary_source",
            ),
            AnswerRecord(
                question_id="q2",
                is_correct=True,
                confidence=85,
                authority_points=20,
                verification_type="independent_source",
                claimed_understanding=True,
                transfer_succeeded=True,
            ),
        ),
        events=EventCounts(independent_source_checks=3),
    )
    result = assess(data)
    assert result.persona_id == "clear_tool_user"
    assert result.total_score <= 30


def test_ai_authorized_know_it_all() -> None:
    data = AssessmentInput(
        answers=(
            AnswerRecord(
                question_id="q1",
                is_correct=False,
                confidence=95,
                authority_points=95,
                verification_type="none",
                evidence_response="ask_ai_to_rebut",
            ),
            AnswerRecord(
                question_id="q2",
                is_correct=False,
                confidence=90,
                authority_points=90,
                verification_type="same_ai_again",
                evidence_response="reject_valid_evidence",
            ),
        ),
        events=EventCounts(
            changed_correct_to_wrong_after_ai=2,
            accepted_without_evidence=2,
        ),
    )
    result = assess(data)
    assert result.persona_id == "ai_authorized_know_it_all"
    assert result.authority_dependence >= 75
    assert result.confidence_miscalibration >= 75


def test_multi_model_voter_priority() -> None:
    data = AssessmentInput(
        answers=(
            AnswerRecord(
                question_id="q1",
                is_correct=True,
                confidence=70,
                authority_points=70,
                verification_type="another_ai",
            ),
        ),
        events=EventCounts(
            cross_model_only_checks=3,
            independent_source_checks=0,
        ),
    )
    result = assess(data)
    assert result.persona_id == "multi_model_voter"
