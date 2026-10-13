"""Run with .venv/bin/python -m scripts.replay_deep_think (no retrieval)."""

import argparse
import json
from pathlib import Path

from core.thinking_replay import (
    DEFAULT_FIXTURE,
    ExtractiveReplayModel,
    benchmark_configurations,
    replay,
)


def render_replay_report(result):
    from core.thinking_evidence import render_card_references

    report = result["report"]
    if "## References" not in report:
        checkpoint = result["checkpoint"]
        references = render_card_references(checkpoint["fact_cards"], checkpoint["report_bindings"])
        if references:
            report = report.rstrip() + "\n\n" + references + "\n"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--reextract", action="store_true")
    parser.add_argument("--empty", action="store_true", help="Exercise empty-generation fallback")
    parser.add_argument("--context", type=int, default=12_288)
    parser.add_argument("--base-url", help="Optional deployed OpenAI-compatible model; retrieval stays offline")
    parser.add_argument("--model", default="default")
    parser.add_argument("--container", help="Use its loopback llama-server without restarting the application")
    parser.add_argument("--production", action="store_true", help="Replay the production chapter path, with no tools/retrieval")
    parser.add_argument("--chapter-output", type=int, default=3072)
    parser.add_argument("--audit-output", type=int, default=1024)
    parser.add_argument("--section-output", type=int, choices=(2048, 4096, 6144), default=2048)
    parser.add_argument("--non-thinking", action="store_true")
    parser.add_argument("--repeat", type=int, choices=range(1, 6), default=1)
    parser.add_argument(
        "--matrix", action="store_true",
        help="Compare 2k thinking, 4k non-thinking, and permitted 4k/6k thinking policies",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fixture.resolve() in {args.output.resolve(), args.output.with_suffix(".md").resolve()}:
        parser.error("Outputs must not overwrite the input evidence")
    if args.output.resolve() == args.fixture.resolve():
        parser.error("Output must not overwrite the fixture")
    model = ExtractiveReplayModel(empty=args.empty)
    if args.container:
        from scripts.generate_thinking_report import DockerModel

        model = DockerModel(args.container, args.base_url or "http://127.0.0.1:8081/v1", args.model)
    elif args.base_url:
        from core.llm_openai import load_openai

        _, model = load_openai(model=args.model, base_url=args.base_url)
    configurations = benchmark_configurations(args.context) if args.matrix else [{
        "label": ("non-thinking" if args.non_thinking else "thinking") + f"-{args.section_output}",
        "section": {"output_tokens": args.section_output, "thinking_mode": not args.non_thinking},
    }]
    results = [
        replay(
            args.fixture,
            model=model,
            reextract=args.reextract,
            context_tokens=args.context,
            stage_policies={"section": configuration["section"],
                            "chapter_analysis": {"output_tokens": args.chapter_output},
                            "audit": {"output_tokens": args.audit_output}},
            configuration_label=configuration["label"],
            production=args.production,
        )
        for configuration in configurations
        for _ in range(args.repeat)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    for index, result in enumerate(results, 1):
        report_path = args.output.with_suffix(".md") if len(results) == 1 else args.output.with_name(f"{args.output.stem}-{index}.md")
        if report_path.resolve() == args.fixture.resolve():
            raise ValueError("Report must not overwrite the fixture")
        report = render_replay_report(result)
        report_path.write_text(report, encoding="utf-8")
    print(json.dumps([{
        "configuration": result["configuration"]["label"],
        **{key: result[key] for key in (
            "fixture_sha256", "mode", "cards", "eligible_cards", "bound_cards",
            "elapsed_seconds", "error",
        )},
    } for result in results], indent=2))


if __name__ == "__main__":
    main()
