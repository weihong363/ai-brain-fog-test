from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_brain_fog import AnswerRecord, AssessmentInput, assess
from ai_brain_fog.session import format_result


def main() -> None:
    parser = argparse.ArgumentParser(description="计算交互组件提交的 AI 脑雾测试结果")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("payload", nargs="?", help="UTF-8 JSON 的 Base64 编码")
    source.add_argument("--json-file", type=Path, help="用于本地验证的原始回答 JSON")
    args = parser.parse_args()
    raw = _load_payload(args.payload, args.json_file)
    assessment = AssessmentInput(
        scenario=raw["scenario"],
        answers=tuple(AnswerRecord(**item) for item in raw["answers"]),
    )
    print(format_result(assess(assessment)))


def _load_payload(payload: str | None, json_file: Path | None) -> dict[str, object]:
    if json_file is not None:
        return json.loads(json_file.read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError("payload is required")
    return json.loads(base64.b64decode(payload, validate=True).decode("utf-8"))


if __name__ == "__main__":
    main()
