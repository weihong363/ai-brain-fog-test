from pathlib import Path

from ai_brain_fog import assess, load_question_pool
from ai_brain_fog.session import collect_assessment, format_result, load_assessment

ROOT = Path(__file__).resolve().parents[1]


def test_demo_response_runs_end_to_end() -> None:
    questions = load_question_pool("mixed")
    assessment = load_assessment(ROOT / "data" / "responses.demo.json")

    output = format_result(assess(assessment, questions=questions))

    assert "AI脑雾指数" in output
    assert "行为证据：" not in output
    assert "判断正确" not in output
    assert "答对" not in output
    assert "答错" not in output
    assert "最关键建议：" in output
    assert "分享文案：" in output
    assert "验证惰性：暂未评估" not in output
    assert "自信失准：暂未评估" not in output
    assert "不构成智力、心理或临床评估" in output


def test_interactive_flow_collects_all_twelve_answers() -> None:
    responses = iter(
        [
            "6",
            "C",
            "C",
            "C",
            "C",
            "B",
            "D",
            "B", "B",
            "C", "C",
            "B", "B",
            "B", "B",
            "A",
            "C",
        ]
    )

    assessment = collect_assessment(
        input_fn=lambda _: next(responses),
        output_fn=lambda _: None,
    )

    assert assessment.scenario == "mixed"
    assert len(assessment.answers) == 12
    assert all(answer.question_id.startswith("mixed_") for answer in assessment.answers)
    assert all(answer.effective_confidence is None for answer in assessment.answers)
    assert all(answer.verification_type is None for answer in assessment.answers)
    assert all(answer.claimed_understanding is None for answer in assessment.answers)
    assert all(answer.evidence_response == "not_applicable" for answer in assessment.answers)


def test_interactive_options_are_randomized_without_changing_scoring_ids() -> None:
    questions = load_question_pool("mixed")[:12]
    responses = iter(["6"] + ["A"] * 16)
    output: list[str] = []

    assessment = collect_assessment(
        questions=questions,
        input_fn=lambda _: next(responses),
        output_fn=output.append,
        shuffle_fn=lambda values: values.reverse(),
    )

    first = questions[0]
    assert any(f"A. {first.options['D']}" in line for line in output)
    assert assessment.answers[0].initial_answer == "D"
    assert any("AI提示：答案D" in line for line in output)
