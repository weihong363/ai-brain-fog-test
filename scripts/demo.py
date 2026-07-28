from ai_brain_fog import AnswerRecord, AssessmentInput, EventCounts, assess

assessment = AssessmentInput(
    answers=(
        AnswerRecord(
            question_id="authority_01",
            is_correct=False,
            confidence=91,
            authority_points=95,
            verification_type="none",
            evidence_response="reject_valid_evidence",
        ),
        AnswerRecord(
            question_id="authority_02",
            is_correct=False,
            confidence=85,
            authority_points=90,
            verification_type="another_ai",
        ),
        AnswerRecord(
            question_id="transfer_01",
            is_correct=False,
            confidence=88,
            authority_points=80,
            verification_type="none",
            claimed_understanding=True,
            transfer_succeeded=False,
        ),
    ),
    events=EventCounts(
        changed_correct_to_wrong_after_ai=2,
        accepted_without_evidence=3,
        independent_source_checks=0,
        cross_model_only_checks=1,
        claimed_understanding_failed_transfer=1,
    ),
)

result = assess(assessment)
print(result)
