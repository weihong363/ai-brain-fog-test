from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from .catalog import load_question_bank
from .models import (
    AnswerRecord,
    AssessmentInput,
    AssessmentResult,
    QuestionSpec,
    Scenario,
)

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]
Choice = TypeVar("Choice")
ShuffleFn = Callable[[list[str]], None]

SCENARIO_LABELS: dict[Scenario, str] = {
    "programming": "编程",
    "learning": "学习",
    "work": "职场",
    "business": "商业",
    "news": "新闻",
    "mixed": "综合",
}
def collect_assessment(
    questions: tuple[QuestionSpec, ...] | None = None,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
    shuffle_fn: ShuffleFn = random.shuffle,
) -> AssessmentInput:
    scenario = _choose_scenario(input_fn, output_fn)
    output_fn("没有标准答案，请选择最符合你真实反应的选项。")
    selected_questions = questions or load_question_bank(scenario, seed=None)
    answers = tuple(
        _collect_answer(index, question, scenario, input_fn, output_fn, shuffle_fn)
        for index, question in enumerate(selected_questions, start=1)
    )
    return AssessmentInput(scenario=scenario, answers=answers)


def load_assessment(path: Path) -> AssessmentInput:
    raw = json.loads(path.read_text(encoding="utf-8"))
    answers = tuple(AnswerRecord(**item) for item in raw["answers"])
    return AssessmentInput(scenario=raw["scenario"], answers=answers)


def format_result(result: AssessmentResult) -> str:
    hidden = f"；隐藏标签：{result.hidden_tag}" if result.hidden_tag else ""
    confidence = result.confidence_miscalibration
    confidence_text = str(confidence) if confidence is not None else "暂未评估"
    verification = result.verification_laziness
    verification_text = str(verification) if verification is not None else "暂未评估"
    return (
        f"\n{result.persona_name}｜{result.subtitle}\n"
        f"标签：{' / '.join(result.fixed_tags)}{hidden}\n\n"
        f"AI脑雾指数：{result.total_score}\n"
        f"- AI权威依赖：{result.authority_dependence}\n"
        f"- 验证惰性：{verification_text}\n"
        f"- 自信失准：{confidence_text}\n"
        f"- 理解幻觉：{result.understanding_illusion}\n\n"
        f"最关键建议：{result.recommendation}\n"
        f"分享文案：{result.share_line}\n"
        "说明：这是娱乐化AI素养测试，不构成智力、心理或临床评估。"
    )


def _choose_scenario(input_fn: InputFn, output_fn: OutputFn) -> Scenario:
    choices = list(SCENARIO_LABELS)
    labels = "，".join(f"{index + 1}.{SCENARIO_LABELS[key]}" for index, key in enumerate(choices))
    output_fn("选择常用场景：" + labels)
    selected = _ask_choice("请输入编号：", {str(index + 1): key for index, key in enumerate(choices)}, input_fn)
    return selected


def _collect_answer(
    index: int,
    question: QuestionSpec,
    scenario: Scenario,
    input_fn: InputFn,
    output_fn: OutputFn,
    shuffle_fn: ShuffleFn,
) -> AnswerRecord:
    displayed_options = _randomized_option_mapping(question, shuffle_fn)
    output_fn(f"\n[{index}/12] {question.prompt_for(scenario)}")
    output_fn(
        "\n".join(
            f"{display_key}. {question.options[value]}"
            for display_key, value in displayed_options.items()
        )
    )
    initial_answer = _ask_choice("最符合你真实反应的选项：", displayed_options, input_fn)
    final_answer = _collect_interference(question, displayed_options, input_fn, output_fn)
    if question.phase == "interference":
        output_fn(f"补充信息：{question.explanation}")
    return AnswerRecord(
        question_id=question.question_id,
        initial_answer=initial_answer,
        final_answer=final_answer,
    )


def _collect_interference(
    question: QuestionSpec,
    displayed_options: dict[str, str],
    input_fn: InputFn,
    output_fn: OutputFn,
) -> str | None:
    if question.phase != "interference":
        return None
    output_fn(f"\nAI提示：{_remap_ai_message(question.ai_message, displayed_options)}")
    final_answer = _ask_choice(
        "看过AI提示后，最符合你此刻反应的选项：",
        displayed_options,
        input_fn,
    )
    return final_answer


def _randomized_option_mapping(
    question: QuestionSpec,
    shuffle_fn: ShuffleFn,
) -> dict[str, str]:
    values = list(question.options)
    shuffle_fn(values)
    return dict(zip(question.options, values, strict=True))


def _remap_ai_message(message: str | None, displayed_options: dict[str, str]) -> str:
    display_by_value = {value: key for key, value in displayed_options.items()}
    return re.sub(
        r"答案([A-D])",
        lambda match: f"答案{display_by_value.get(match.group(1), match.group(1))}",
        message or "",
    )


def _ask_choice(prompt: str, choices: dict[str, Choice], input_fn: InputFn) -> Choice:
    while True:
        raw = input_fn(prompt).strip().upper()
        if raw in choices:
            return choices[raw]
