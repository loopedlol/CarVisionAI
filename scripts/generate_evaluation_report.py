#!/usr/bin/env python3
"""Regenerate a concise Markdown report from machine-readable metrics."""

import argparse
import json
from pathlib import Path

from carvision.evaluation import render_comparison_markdown, render_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    rendered = (render_comparison_markdown(data) if "configurations" in data
                else render_markdown_report(data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

