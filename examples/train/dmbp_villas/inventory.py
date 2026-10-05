"""Write corpus inventory tables for an exported DMBP Villas dataset.

``build_inventory`` reads only the exported dataset (``dataset.json``, the split
files, exported contexts, labels, and teacher records) and writes:

- ``inventory/submissions.csv``: one row per submission with duplicates, split,
  and context size and file counts;
- ``inventory/episodes.csv``: one row per episode with teacher trace statistics,
  label counts, and how its leaf spec compares with the teacher's prompt;
- ``inventory/labels.csv``: one row per (episode, rule key) teacher label;
- ``inventory/rules.csv``: training-split verdict counts per rule key with
  sparse-rule flags;
- ``inventory/README.md``: summary tables.
"""

from __future__ import annotations

import csv
import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VERDICTS = ("pass", "fail", "needs_human", "not_applicable")
MIN_DECISIVE_TRAIN_SUBMISSIONS = 5


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tree_size(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines + [""]


def _teacher_instructions(invocation: dict[str, Any]) -> str:
    """The subagent prompt from a teacher system prompt, without the SDK's appended notes."""
    return invocation["system"][-1]["text"].split("\n\nNotes:\n", 1)[0]


def _task_values(task: str) -> list[str]:
    return [line.split(": ", 1)[-1] for line in task.splitlines()]


def _compare_spec(spec: dict[str, Any] | None, teacher: dict[str, Any] | None) -> dict[str, Any]:
    """Compare a leaf spec with the first teacher invocation; empty cells when either is missing."""
    if spec is None or teacher is None:
        return {
            "spec_instructions_match": "",
            "spec_instruction_diff_lines": "",
            "spec_tools_match": "",
            "spec_mcp_schemas_match": "",
            "spec_task_values_match": "",
        }
    invocation = teacher["invocations"][0]
    instructions = _teacher_instructions(invocation)
    diff = difflib.unified_diff(spec["instructions"].splitlines(), instructions.splitlines(), lineterm="", n=0)
    teacher_mcp = [tool for tool in invocation["tools"] if tool["name"].startswith("mcp__")]
    return {
        "spec_instructions_match": instructions == spec["instructions"],
        "spec_instruction_diff_lines": sum(
            1 for line in diff if line[:1] in "+-" and not line.startswith(("+++", "---"))
        ),
        "spec_tools_match": [tool["name"] for tool in invocation["tools"]] == spec["tools"],
        "spec_mcp_schemas_match": teacher_mcp == spec["mcp_tools"],
        "spec_task_values_match": _task_values(invocation["agent_tool_arguments"]["prompt"])
        == _task_values(spec["task_prompt"]),
    }


def build_inventory(dataset_dir: Path) -> Path:
    """Write the inventory directory and return the path of its README."""
    manifest = _read_json(dataset_dir / "dataset.json")
    rows = _read_jsonl(dataset_dir / "train.jsonl") + _read_jsonl(dataset_dir / "validation.jsonl")
    output = dataset_dir / "inventory"
    output.mkdir(exist_ok=True)

    submission_rows = []
    for submission in manifest["submissions"]:
        context = dataset_dir / submission["agent_context_path"]
        app_files, app_bytes = _tree_size(context / "application")
        policy_files, policy_bytes = _tree_size(context / "policies")
        drawings = context / "application" / "drawings"
        extensions = Counter(item.suffix.lstrip(".") for item in (context / "application").rglob("*") if item.is_file())
        submission_rows.append(
            {
                "submission_id": submission["submission_id"],
                "run_id": submission["run_id"],
                "duplicate_run_ids": " ".join(submission["duplicate_run_ids"]),
                "split": submission["split"],
                "status": submission["status"],
                "has_trace": submission["has_trace"],
                "episodes": submission["episode_count"],
                "drawings": sum(1 for item in drawings.iterdir() if item.is_dir()) if drawings.is_dir() else 0,
                "application_files": app_files,
                "application_mb": round(app_bytes / 1e6, 1),
                "application_file_types": " ".join(f"{ext}:{n}" for ext, n in sorted(extensions.items())),
                "policy_files": policy_files,
                "policy_mb": round(policy_bytes / 1e6, 1),
            }
        )

    episode_rows, label_rows = [], []
    for row in sorted(rows, key=lambda item: (item["submission_id"], item["position"])):
        label = _read_json(dataset_dir / row["ground_truth_path"])
        verdicts = Counter(rule.get("verdict") for rule in label["rules"])
        tooling_notes = sum(1 for rule in label["rules"] if rule.get("tooling_note"))
        for rule in label["rules"]:
            label_rows.append(
                {
                    "submission_id": row["submission_id"],
                    "split": row["split"],
                    "category_id": row["category_id"],
                    "rule_key": rule["rule_key"],
                    "verdict": rule.get("verdict"),
                    "evidence_level": rule.get("evidence_level"),
                    "has_tooling_note": bool(rule.get("tooling_note")),
                }
            )
        teacher = _read_json(dataset_dir / row["teacher_path"]) if row.get("teacher_path") else None
        spec = _read_json(dataset_dir / row["leaf_spec_path"]) if row.get("leaf_spec_path") else None
        stats = teacher["stats"] if teacher else {}
        episode_rows.append(
            {
                "episode_id": row["episode_id"],
                "split": row["split"],
                "category_id": row["category_id"],
                "rules": len(label["rules"]),
                **{f"label_{verdict}": verdicts.get(verdict, 0) for verdict in VERDICTS},
                "labels_with_tooling_note": tooling_notes,
                "has_teacher": bool(stats),
                "teacher_models": " ".join(stats.get("models", [])),
                "invocations": stats.get("invocations", ""),
                "model_calls": stats.get("model_calls", ""),
                "tool_calls": stats.get("tool_calls", ""),
                "tool_errors": stats.get("tool_errors", ""),
                "tool_result_span_mismatches": stats.get("tool_result_span_mismatches", ""),
                "output_tokens": stats.get("output_tokens", ""),
                "max_prompt_tokens": stats.get("max_final_prompt_tokens", ""),
                "finish_reason": stats.get("finish_reason", ""),
                "wall_seconds": stats.get("wall_seconds", ""),
                "has_leaf_spec": spec is not None,
                **_compare_spec(spec, teacher),
            }
        )

    rule_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    decisive_submissions: dict[tuple[str, str], set[str]] = defaultdict(set)
    for label in label_rows:
        if label["split"] != "train":
            continue
        key = (label["category_id"], label["rule_key"])
        rule_counts[key][label["verdict"]] += 1
        if label["verdict"] in ("pass", "fail"):
            decisive_submissions[key].add(label["submission_id"])
    rule_rows = []
    for (category_id, rule_key), counts in sorted(rule_counts.items()):
        decisive = len(decisive_submissions[(category_id, rule_key)])
        rule_rows.append(
            {
                "category_id": category_id,
                "rule_key": rule_key,
                **{verdict: counts.get(verdict, 0) for verdict in VERDICTS},
                "decisive_train_submissions": decisive,
                "no_pass": counts.get("pass", 0) == 0,
                "no_fail": counts.get("fail", 0) == 0,
                "few_decisive": decisive < MIN_DECISIVE_TRAIN_SUBMISSIONS,
            }
        )

    _write_csv(output / "submissions.csv", submission_rows)
    _write_csv(output / "episodes.csv", episode_rows)
    _write_csv(output / "labels.csv", label_rows)
    _write_csv(output / "rules.csv", rule_rows)
    readme = output / "README.md"
    readme.write_text(_render(manifest, submission_rows, episode_rows, label_rows, rule_rows), encoding="utf-8")
    return readme


def _render(
    manifest: dict[str, Any],
    submissions: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    rules: list[dict[str, Any]],
) -> str:
    split_counts = Counter(episode["split"] for episode in episodes)
    duplicates = sum(len(row["duplicate_run_ids"].split()) for row in submissions)
    lines = [
        "# DMBP Villas dataset inventory",
        "",
        "Generated by `prepare_dataset.py`; regenerate with `prepare_dataset.py inventory`.",
        "",
        f"- Submissions: {len(submissions)} ({sum(row['status'] == 'ready' for row in submissions)} ready); "
        f"duplicate runs dropped: {duplicates}",
        f"- Episodes: {len(episodes)} ({split_counts['train']} train / {split_counts['validation']} validation); "
        f"with teacher trace: {sum(episode['has_teacher'] for episode in episodes)}",
        f"- Labels: {len(labels)} rule verdicts",
        "",
        "## Submissions",
        "",
    ]
    lines += _table(
        ["Submission", "Split", "Status", "Trace", "Episodes", "Drawings", "App files", "App MB", "Duplicates"],
        [
            [
                r["submission_id"],
                r["split"],
                r["status"],
                r["has_trace"],
                r["episodes"],
                r["drawings"],
                r["application_files"],
                r["application_mb"],
                r["duplicate_run_ids"] or "-",
            ]
            for r in submissions
        ],
    )

    traced = [episode for episode in episodes if episode["has_teacher"]]
    prompt_tokens = [episode["max_prompt_tokens"] for episode in traced]
    lines += ["## Teacher traces", ""]
    lines += [
        f"Max prompt tokens per episode: p50 {_percentile(prompt_tokens, 0.5)}, "
        f"p95 {_percentile(prompt_tokens, 0.95)}, max {max(prompt_tokens, default=0)}.",
        "",
    ]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in traced:
        by_category[episode["category_id"]].append(episode)
    lines += _table(
        ["Category", "Episodes", "Models", "Mean model calls", "Mean tool calls", "Tool errors", "Max prompt tokens"],
        [
            [
                category,
                len(items),
                " ".join(sorted({model.split("/")[-1] for item in items for model in item["teacher_models"].split()})),
                round(sum(item["model_calls"] for item in items) / len(items), 1),
                round(sum(item["tool_calls"] for item in items) / len(items), 1),
                sum(item["tool_errors"] for item in items),
                max(item["max_prompt_tokens"] for item in items),
            ]
            for category, items in sorted(by_category.items())
        ],
    )
    unusual = [
        episode
        for episode in traced
        if episode["invocations"] != 1
        or episode["finish_reason"] != "end_turn"
        or episode["tool_result_span_mismatches"]
    ]
    if unusual:
        lines += [
            "Episodes needing review. Finish `tool_use` means the trace ends after a tool call, so the final "
            "model call is missing from the trace.",
            "",
        ]
        lines += _table(
            ["Episode", "Invocations", "Finish", "Span mismatches"],
            [
                [e["episode_id"], e["invocations"], e["finish_reason"], e["tool_result_span_mismatches"]]
                for e in unusual
            ],
        )
    excluded = [
        [f"{submission['submission_id']}:{item['subagent_name']}", item["reason"]]
        for submission in manifest["submissions"]
        for item in submission.get("excluded_episodes", [])
    ]
    if excluded:
        lines += ["Episodes excluded from the dataset:", ""]
        lines += _table(["Episode", "Reason"], excluded)
    missing = [episode["episode_id"] for episode in episodes if not episode["has_teacher"]]
    if missing:
        lines += [f"Episodes without a teacher trace: {len(missing)}.", ""]

    lines += _render_leaf_specs(manifest, episodes)

    lines += ["## Labels by category", ""]
    category_verdicts: dict[str, Counter[str]] = defaultdict(Counter)
    tooling_notes: Counter[str] = Counter()
    for label in labels:
        category_verdicts[label["category_id"]][label["verdict"]] += 1
        tooling_notes[label["category_id"]] += label["has_tooling_note"]
    lines += _table(
        ["Category", *VERDICTS, "With tooling note"],
        [
            [category, *(counts.get(verdict, 0) for verdict in VERDICTS), tooling_notes[category]]
            for category, counts in sorted(category_verdicts.items())
        ],
    )
    evidence: dict[str, Counter[str]] = defaultdict(Counter)
    for label in labels:
        evidence[label["evidence_level"]][label["verdict"]] += 1
    lines += ["## Labels by evidence level", ""]
    lines += _table(
        ["Evidence level", *VERDICTS],
        [[level, *(counts.get(verdict, 0) for verdict in VERDICTS)] for level, counts in sorted(evidence.items())],
    )

    lines += [
        "## Sparse rules (training split)",
        "",
        f"A rule is decisive in a submission when its verdict is pass or fail. `few_decisive` means fewer than "
        f"{MIN_DECISIVE_TRAIN_SUBMISSIONS} decisive training submissions. Per-rule detail is in `rules.csv`.",
        "",
    ]
    by_rule_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rule in rules:
        by_rule_category[rule["category_id"]].append(rule)
    lines += _table(
        ["Category", "Rules", "No pass", "No fail", "Few decisive", "Never decisive"],
        [
            [
                category,
                len(items),
                sum(item["no_pass"] for item in items),
                sum(item["no_fail"] for item in items),
                sum(item["few_decisive"] for item in items),
                sum(item["decisive_train_submissions"] == 0 for item in items),
            ]
            for category, items in sorted(by_rule_category.items())
        ],
    )
    return "\n".join(lines)


def _render_leaf_specs(manifest: dict[str, Any], episodes: list[dict[str, Any]]) -> list[str]:
    with_spec = [episode for episode in episodes if episode["has_leaf_spec"]]
    lines = ["## Leaf specs", ""]
    if not with_spec:
        return lines + ["No leaf specs rendered; run `prepare_dataset.py leaf-specs`.", ""]
    for source in manifest.get("leaf_spec_sources", []):
        lines.append(
            "- Rendered from "
            + ", ".join(
                f"{part} `{state['path']}` at `{state['git_commit'][:10]}`"
                + (" with uncommitted changes" if state["dirty"] else "")
                for part, state in source.items()
            )
        )
    compared = [episode for episode in with_spec if episode["spec_instructions_match"] != ""]
    lines += [
        "",
        f"{len(with_spec)} episodes have a leaf spec; {len(compared)} also have a teacher trace and are compared "
        "with the teacher's subagent prompt (system prompt without the SDK's notes), tool list, MCP tool schemas, "
        "and the category, findings path, and intake path in its task message.",
        "",
    ]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in compared:
        by_category[episode["category_id"]].append(episode)
    lines += _table(
        [
            "Category",
            "Episodes",
            "Instructions identical",
            "Max differing lines",
            "Tools",
            "MCP schemas",
            "Task values",
        ],
        [
            [
                category,
                len(items),
                sum(item["spec_instructions_match"] for item in items),
                max(item["spec_instruction_diff_lines"] for item in items),
                sum(item["spec_tools_match"] for item in items),
                sum(item["spec_mcp_schemas_match"] for item in items),
                sum(item["spec_task_values_match"] for item in items),
            ]
            for category, items in sorted(by_category.items())
        ],
    )
    return lines
