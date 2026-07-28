from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

VerificationType = Literal[
    "primary_source",
    "independent_source",
    "provided_citation",
    "another_ai",
    "same_ai_again",
    "none",
]
EvidenceResponse = Literal[
    "change_after_evidence",
    "remain_uncertain",
    "reject_valid_evidence",
    "ask_ai_to_rebut",
    "not_applicable",
]


@dataclass(frozen=True)
class AnswerRecord:
    question_id: str
    is_correct: bool
    confidence: int
    authority_points: int = 0
    verification_type: VerificationType = "none"
    evidence_response: EvidenceResponse = "not_applicable"
    claimed_understanding: bool = False
    transfer_succeeded: bool | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 100:
            raise ValueError("confidence must be between 0 and 100")
        if not 0 <= self.authority_points <= 100:
            raise ValueError("authority_points must be between 0 and 100")


@dataclass(frozen=True)
class EventCounts:
    changed_correct_to_wrong_after_ai: int = 0
    accepted_without_evidence: int = 0
    independent_source_checks: int = 0
    cross_model_only_checks: int = 0
    restate_without_support: int = 0
    claimed_understanding_failed_transfer: int = 0
    rejected_valid_evidence: int = 0


@dataclass(frozen=True)
class AssessmentInput:
    answers: tuple[AnswerRecord, ...]
    events: EventCounts = field(default_factory=EventCounts)


@dataclass(frozen=True)
class AssessmentResult:
    total_score: int
    authority_dependence: int
    verification_laziness: int
    confidence_miscalibration: int
    understanding_illusion: int
    persona_id: str
    evidence: tuple[str, ...]
