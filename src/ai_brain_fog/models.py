from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Scenario = Literal["programming", "learning", "work", "business", "news", "mixed"]
QuestionPhase = Literal["baseline", "interference", "transfer"]
Dimension = Literal[
    "authority_dependence",
    "verification_laziness",
    "confidence_miscalibration",
    "understanding_illusion",
]
QuestionType = Literal[
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
]
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
AnswerEvent = Literal[
    "accepted_without_evidence",
    "independent_source_checks",
    "cross_model_only_checks",
    "restate_without_support",
    "rejected_valid_evidence",
]

SCENARIOS = frozenset({"programming", "learning", "work", "business", "news", "mixed"})
VERIFICATION_TYPES = frozenset(
    {"primary_source", "independent_source", "provided_citation", "another_ai", "same_ai_again", "none"}
)
EVIDENCE_RESPONSES = frozenset(
    {
        "change_after_evidence",
        "remain_uncertain",
        "reject_valid_evidence",
        "ask_ai_to_rebut",
        "not_applicable",
    }
)
ANSWER_EVENTS = frozenset(
    {
        "accepted_without_evidence",
        "independent_source_checks",
        "cross_model_only_checks",
        "restate_without_support",
        "rejected_valid_evidence",
    }
)


@dataclass(frozen=True)
class QuestionSpec:
    question_id: str
    scenario: Scenario
    phase: QuestionPhase
    dimension: Dimension
    question_type: QuestionType
    prompt_template: str
    options: dict[str, str]
    correct_answer: str
    explanation: str
    authority_points: dict[str, int]
    dimension_points: dict[Dimension, dict[str, int]]
    option_events: dict[str, tuple[AnswerEvent, ...]]
    requires_verification: bool = False
    ai_answer: str | None = None
    ai_message: str | None = None

    def prompt_for(self, scenario: Scenario) -> str:
        if scenario != self.scenario:
            raise ValueError(f"{self.question_id} does not belong to scenario {scenario}")
        return self.prompt_template


@dataclass(frozen=True)
class AnswerRecord:
    question_id: str
    initial_answer: str
    initial_confidence: int | None = None
    final_answer: str | None = None
    final_confidence: int | None = None
    verification_type: VerificationType | None = None
    evidence_response: EvidenceResponse = "not_applicable"
    claimed_understanding: bool | None = None
    restated_without_support: bool = False

    def __post_init__(self) -> None:
        if self.initial_confidence is not None:
            _validate_percentage("initial_confidence", self.initial_confidence)
        if self.final_confidence is not None:
            _validate_percentage("final_confidence", self.final_confidence)
        if self.final_answer is None and self.final_confidence is not None:
            raise ValueError("final_confidence requires final_answer")
        if self.verification_type is not None and self.verification_type not in VERIFICATION_TYPES:
            raise ValueError(f"unsupported verification_type: {self.verification_type}")
        if self.evidence_response not in EVIDENCE_RESPONSES:
            raise ValueError(f"unsupported evidence_response: {self.evidence_response}")

    @property
    def effective_answer(self) -> str:
        return self.final_answer or self.initial_answer

    @property
    def effective_confidence(self) -> int | None:
        return self.final_confidence if self.final_confidence is not None else self.initial_confidence


@dataclass(frozen=True)
class EventCounts:
    changed_correct_to_wrong_after_ai: int = 0
    accepted_without_evidence: int = 0
    independent_source_checks: int = 0
    cross_model_only_checks: int = 0
    restate_without_support: int = 0
    claimed_understanding_failed_transfer: int = 0
    rejected_valid_evidence: int = 0

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.as_metrics().values()):
            raise ValueError("event counts cannot be negative")

    def as_metrics(self) -> dict[str, int]:
        return {
            "changed_correct_to_wrong_after_ai": self.changed_correct_to_wrong_after_ai,
            "accepted_without_evidence": self.accepted_without_evidence,
            "independent_source_checks": self.independent_source_checks,
            "cross_model_only_checks": self.cross_model_only_checks,
            "restate_without_support": self.restate_without_support,
            "claimed_understanding_failed_transfer": self.claimed_understanding_failed_transfer,
            "rejected_valid_evidence": self.rejected_valid_evidence,
        }


@dataclass(frozen=True)
class AssessmentInput:
    scenario: Scenario
    answers: tuple[AnswerRecord, ...]

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"unsupported scenario: {self.scenario}")


@dataclass(frozen=True)
class PersonaProfile:
    persona_id: str
    name: str
    subtitle: str
    tags: tuple[str, str, str]
    share_line: str
    asset: str


@dataclass(frozen=True)
class AssessmentResult:
    total_score: int
    authority_dependence: int
    verification_laziness: int | None
    confidence_miscalibration: int | None
    understanding_illusion: int | None
    persona_id: str
    persona_name: str
    subtitle: str
    fixed_tags: tuple[str, str, str]
    hidden_tag: str | None
    recommendation: str
    share_line: str
    asset: str


def _validate_percentage(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise ValueError(f"{name} must be an integer between 0 and 100")
