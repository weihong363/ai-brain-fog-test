from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .models import EVIDENCE_RESPONSES, VERIFICATION_TYPES

Operator = Literal["gte", "lte", "eq"]
DIMENSIONS = {
    "authority_dependence",
    "verification_laziness",
    "confidence_miscalibration",
    "understanding_illusion",
}
AUTHORITY_BEHAVIORS = {
    "changed_correct_to_ai_wrong",
    "accepted_ai_wrong",
    "followed_ai_after_cross_model",
    "rejected_ai_without_evidence",
    "verified_independently",
    "other",
}
UNDERSTANDING_OUTCOMES = {
    "claimed_and_succeeded",
    "not_claimed_but_succeeded",
    "not_claimed_and_failed",
    "claimed_but_failed",
}


@dataclass(frozen=True)
class Condition:
    metric: str
    operator: Operator
    value: int


@dataclass(frozen=True)
class PersonaRule:
    persona_id: str
    conditions: tuple[Condition, ...]


@dataclass(frozen=True)
class HiddenTagRule:
    label: str
    conditions: tuple[Condition, ...]


@dataclass(frozen=True)
class ScoringRules:
    total_weights: dict[str, float]
    confidence_weights: dict[str, float]
    verification_laziness: dict[str, int]
    evidence_response: dict[str, int]
    understanding_illusion: dict[str, int]
    authority_behavior: dict[str, int]
    classification: tuple[PersonaRule, ...]
    fallback_persona: str
    hidden_tags: tuple[HiddenTagRule, ...]
    recommendations: dict[str, str]


def default_rules_path() -> Path:
    source_path = Path(__file__).resolve().parents[2] / "rules" / "scoring.json"
    return source_path if source_path.exists() else Path(sys.prefix) / "rules" / "scoring.json"


def load_rules(path: Path | None = None) -> ScoringRules:
    raw = json.loads((path or default_rules_path()).read_text(encoding="utf-8"))
    rules = ScoringRules(
        total_weights=_number_map(raw["total_weights"], float),
        confidence_weights=_number_map(raw["confidence_weights"], float),
        verification_laziness=_number_map(raw["verification_laziness"], int),
        evidence_response=_number_map(raw["evidence_response"], int),
        understanding_illusion=_number_map(raw["understanding_illusion"], int),
        authority_behavior=_number_map(raw["authority_behavior"], int),
        classification=tuple(_persona_rule(item) for item in raw["classification"]),
        fallback_persona=str(raw["fallback_persona"]),
        hidden_tags=tuple(_hidden_tag_rule(item) for item in raw["hidden_tags"]),
        recommendations={str(key): str(value) for key, value in raw["recommendations"].items()},
    )
    _validate_rules(rules)
    return rules


def _persona_rule(raw: dict[str, object]) -> PersonaRule:
    return PersonaRule(persona_id=str(raw["persona_id"]), conditions=_conditions(raw))


def _hidden_tag_rule(raw: dict[str, object]) -> HiddenTagRule:
    return HiddenTagRule(label=str(raw["label"]), conditions=_conditions(raw))


def _conditions(raw: dict[str, object]) -> tuple[Condition, ...]:
    return tuple(
        Condition(str(item["metric"]), str(item["operator"]), int(item["value"]))  # type: ignore[arg-type]
        for item in raw["conditions"]  # type: ignore[union-attr]
    )


def _number_map(raw: dict[str, object], number_type: type[float] | type[int]) -> dict[str, float] | dict[str, int]:
    return {str(key): number_type(value) for key, value in raw.items()}


def _validate_rules(rules: ScoringRules) -> None:
    if set(rules.total_weights) != DIMENSIONS or set(rules.recommendations) != DIMENSIONS:
        raise ValueError("weights and recommendations must cover all four dimensions")
    if abs(sum(rules.total_weights.values()) - 1.0) > 1e-9:
        raise ValueError("total_weights must sum to 1")
    if set(rules.confidence_weights) != {"answer_calibration", "evidence_response"}:
        raise ValueError("confidence_weights must define both confidence components")
    if abs(sum(rules.confidence_weights.values()) - 1.0) > 1e-9:
        raise ValueError("confidence_weights must sum to 1")
    if set(rules.verification_laziness) != VERIFICATION_TYPES:
        raise ValueError("verification_laziness must cover every verification type")
    if set(rules.evidence_response) != EVIDENCE_RESPONSES:
        raise ValueError("evidence_response must cover every evidence response")
    if set(rules.understanding_illusion) != UNDERSTANDING_OUTCOMES:
        raise ValueError("understanding_illusion must cover every transfer outcome")
    if set(rules.authority_behavior) != AUTHORITY_BEHAVIORS:
        raise ValueError("authority_behavior must define every behavior score")
    score_maps = (
        rules.verification_laziness,
        rules.evidence_response,
        rules.understanding_illusion,
        rules.authority_behavior,
    )
    if any(not 0 <= score <= 100 for score_map in score_maps for score in score_map.values()):
        raise ValueError("rule scores must be between 0 and 100")
    conditional_rules = (*rules.classification, *rules.hidden_tags)
    invalid_operator = any(
        condition.operator not in {"gte", "lte", "eq"}
        for rule in conditional_rules
        for condition in rule.conditions
    )
    if invalid_operator:
        raise ValueError("classification contains an unsupported operator")
    persona_ids = [rule.persona_id for rule in rules.classification]
    if len(persona_ids) != len(set(persona_ids)):
        raise ValueError("classification contains duplicate persona ids")
