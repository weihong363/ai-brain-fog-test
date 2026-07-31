from __future__ import annotations

import argparse
from pathlib import Path

from .scoring import assess
from .session import collect_assessment, format_result, load_assessment


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 AI 脑雾人格测试")
    parser.add_argument("--responses", type=Path, help="读取结构化回答 JSON；不提供时进入交互模式")
    args = parser.parse_args()
    assessment = load_assessment(args.responses) if args.responses else collect_assessment()
    print(format_result(assess(assessment)))
