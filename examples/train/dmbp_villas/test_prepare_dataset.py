import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from prepare_dataset import DatasetError, export_dataset, materialize_runs
from teacher_trace import extract_subagents

WORKSPACE_INPUTS = ["intake_check.md", "policy_constraints.md", "policy_dispatch.json"]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _span(span_id, parent_id, name, start, attributes, events=()):
    return {
        "name": name,
        "context": {"span_id": span_id},
        "parent_id": parent_id,
        "start_time": f"2026-10-01T00:00:{start:02d}Z",
        "end_time": f"2026-10-01T00:00:{start + 1:02d}Z",
        "status": {"status_code": "UNSET"},
        "attributes": attributes,
        "events": list(events),
    }


def _chat(span_id, parent_id, start, response_id, offset, total, delta, output, finish=None, first=False):
    attributes = {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "openai/teacher",
        "gen_ai.response.id": response_id,
        "gen_ai.input.messages.offset": offset,
        "gen_ai.input.messages.total_count": total,
        "gen_ai.input.messages": json.dumps(delta),
        "gen_ai.output.messages": json.dumps(output),
        "gen_ai.usage.input_tokens": 10 if finish else 0,
        "gen_ai.usage.output_tokens": 5 if finish else 0,
    }
    if finish:
        attributes["gen_ai.response.finish_reasons"] = [finish]
    if first:
        system = [{"type": "text", "text": "x-anthropic-billing-header: cch=1;"}, {"type": "text", "text": "leaf"}]
        attributes["gen_ai.system_instructions"] = json.dumps(system)
        attributes["gen_ai.tool.definitions"] = json.dumps([{"name": "Read"}])
    return _span(span_id, parent_id, "chat openai/teacher", start, attributes)


def _subagent_spans(run_id: str, subagent: str, prefix: str, start: int) -> list[dict]:
    root = f"/data/runs/{run_id}/agent_context"
    user = {"role": "user", "content": [{"type": "text", "text": subagent, "cache_control": {"type": "ephemeral"}}]}
    call = {"type": "tool_use", "id": f"{prefix}-t", "name": "Read", "input": {"file_path": f"{root}/x.md"}}
    echoed = {"role": "assistant", "content": [{**call, "input": {**call["input"], "limit": None}}]}
    result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"{prefix}-t", "content": "text"}]}
    plain_user = {"role": "user", "content": [{"type": "text", "text": subagent}]}
    first_real = _chat(f"{prefix}-r1", f"{prefix}-i", start + 1, f"{prefix}-c1", 0, 1, [user], [call], "tool_use", True)
    first_real["attributes"]["gen_ai.input.messages"] = "[{... [truncated 10 chars]"
    first_real["events"] = [
        {"name": "gen_ai.input.messages", "attributes": {"payload": json.dumps({"gen_ai.input.messages": [user]})}}
    ]
    return [
        _span(
            f"{prefix}-a",
            "orchestrator",
            "execute_tool Agent",
            start,
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "Agent",
                "gen_ai.tool.call.id": f"{prefix}-a",
                "gen_ai.tool.call.arguments": json.dumps({"subagent_type": subagent, "prompt": subagent}),
            },
        ),
        _span(
            f"{prefix}-i",
            f"{prefix}-a",
            f"invoke_agent {subagent}",
            start,
            {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": subagent},
        ),
        _chat(f"{prefix}-p1", f"{prefix}-i", start, "msg_1", 0, 1, [user], [{"type": "text", "text": ""}], first=True),
        first_real,
        _chat(f"{prefix}-r1b", f"{prefix}-i", start + 1, f"{prefix}-c1", 1, 1, [], [call], "tool_use"),
        _span(
            f"{prefix}-t",
            f"{prefix}-i",
            "execute_tool Read",
            start + 2,
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "Read",
                "gen_ai.tool.call.id": f"{prefix}-t",
                "gen_ai.tool.call.result": "text",
            },
        ),
        # Only the placeholder carries the full delta for the last call; cache_control has moved.
        _chat(
            f"{prefix}-p2",
            f"{prefix}-i",
            start + 3,
            "msg_2",
            0,
            3,
            [plain_user, echoed, result],
            [{"type": "text", "text": ""}],
        ),
        _chat(
            f"{prefix}-r2",
            f"{prefix}-i",
            start + 4,
            f"{prefix}-c2",
            3,
            3,
            [],
            [{"type": "text", "text": "Wrote"}],
            "end_turn",
        ),
    ]


def _write_trace(run_dir: Path, spans: list[dict]) -> None:
    path = run_dir / "agent_context" / "logs" / "trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(span) + "\n" for span in spans), encoding="utf-8")


def _make_run(
    runs_dir: Path,
    run_id: str,
    *,
    complete: bool = True,
    input_dir: str | None = None,
    sha: str | None = None,
    traced_subagents: tuple[str, ...] = (),
) -> None:
    run_dir = runs_dir / run_id
    context = run_dir / "agent_context"
    _write_json(
        run_dir / "metadata.json",
        {"input_dir": input_dir or f"data/datasets/new_dmbp_data/{run_id}", "input_set_sha256": sha or f"sha-{run_id}"},
    )
    _write_json(context / "application" / "document_index.json", {"drawings": []})
    _write_json(context / "policies" / "document_index.json", {"documents": []})
    (context / ".agents").mkdir()
    (context / ".claude").mkdir()
    (context / "logs").mkdir()
    (context / "logs" / "agent.log").write_text("ignored", encoding="utf-8")

    categories = []
    for position, category_id in enumerate(("site-massing", "room-geometry"), start=1):
        categories.append(
            {
                "position": position,
                "category_id": category_id,
                "name": category_id,
                "extraction_agent": category_id,
                "rule_keys": [f"{category_id}.rule"],
                "findings_path": f"workspace/findings/{category_id}.md",
                "scratch_path": f"workspace/scratch/{category_id}",
            }
        )
    _write_json(
        context / "workspace" / "policy_dispatch.json",
        {"schema_version": 1, "rulebook_id": "rulebook", "categories": categories},
    )
    (context / "workspace" / "policy_constraints.md").write_text("# Constraints\n", encoding="utf-8")
    (context / "workspace" / "findings").mkdir()
    (context / "workspace" / "intake_check.md").write_text("# Intake\n", encoding="utf-8")
    (context / "workspace" / "pre_approval_report.md").write_text("ignored", encoding="utf-8")

    if complete:
        for category in categories:
            _write_json(
                context / "workspace" / "structured_findings" / f"{category['position']}_label.json",
                {
                    "schema_version": 1,
                    "position": category["position"],
                    "category_id": category["category_id"],
                    "category_name": category["name"],
                    "findings_path": category["findings_path"],
                    "summary": "summary",
                    "rules": [
                        {
                            "rule_key": category["rule_keys"][0],
                            "verdict": "pass",
                            "evidence_level": "Direct",
                            "tooling_note": "",
                        }
                    ],
                },
            )
    if traced_subagents:
        spans = []
        for index, subagent in enumerate(traced_subagents):
            spans += _subagent_spans(run_id, subagent, f"s{index}", 10 * index)
        _write_trace(run_dir, spans)


def test_extract_subagents_rebuilds_model_visible_conversation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    _write_trace(
        run_dir, _subagent_spans("run-a", "site-massing", "s0", 0) + _subagent_spans("run-a", "site-massing", "s1", 20)
    )

    record = extract_subagents(run_dir / "agent_context" / "logs" / "trace.jsonl", "run-a")["site-massing"]

    assert record["stats"]["invocations"] == 2
    invocation = record["invocations"][0]
    assert invocation["system"] == [{"type": "text", "text": "leaf"}]
    assert [message["role"] for message in invocation["messages"]] == ["user", "assistant", "user", "assistant"]
    assert invocation["messages"][1]["content"][0]["input"]["limit"] is None
    assert invocation["outputs"][0][0]["input"] == {"file_path": "{AGENT_CONTEXT}/x.md"}
    assert invocation["stats"]["model_calls"] == 2
    assert invocation["stats"]["finish_reason"] == "end_turn"
    assert "/data/runs" not in json.dumps(record)
    assert "cache_control" not in json.dumps(record)


def test_export_and_materialize_keep_ground_truth_hidden(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "run-a", traced_subagents=("site-massing",))
    _make_run(runs_dir, "run-b")
    dataset_dir = tmp_path / "dataset"

    manifest = export_dataset(runs_dir, dataset_dir, validation_submission_ids=["run-b"])

    assert manifest["episode_count"] == 4
    assert manifest["splits"] == {"train": ["run-a"], "validation": ["run-b"]}
    submission_dir = dataset_dir / "submissions" / "run-a"
    exported_context = submission_dir / "agent_context"
    assert sorted(path.name for path in exported_context.iterdir()) == ["application", "policies", "workspace"]
    assert sorted(path.name for path in (exported_context / "workspace").iterdir()) == WORKSPACE_INPUTS
    assert (submission_dir / "ground_truth" / "site-massing.json").is_file()
    assert (submission_dir / "teacher" / "site-massing.json").is_file()
    run_a = next(item for item in manifest["submissions"] if item["submission_id"] == "run-a")
    assert run_a["missing_teacher_subagents"] == ["room-geometry"]
    assert (dataset_dir / "inventory" / "README.md").is_file()

    materialized_dir = tmp_path / "materialized"
    materialize_runs(dataset_dir, materialized_dir)
    materialized_workspace = materialized_dir / "run-a" / "agent_context" / "workspace"
    assert sorted(path.name for path in materialized_workspace.iterdir()) == WORKSPACE_INPUTS

    restored_dir = tmp_path / "restored"
    materialize_runs(dataset_dir, restored_dir, restore_ground_truth=True)
    restored = restored_dir / "run-a" / "agent_context" / "workspace" / "structured_findings" / "site-massing.json"
    assert restored.is_file()


def test_relaunched_subagent_is_excluded(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "run-a", traced_subagents=("site-massing", "site-massing"))
    _make_run(runs_dir, "run-b")

    manifest = export_dataset(runs_dir, tmp_path / "dataset", validation_submission_ids=["run-b"])

    run_a = next(item for item in manifest["submissions"] if item["submission_id"] == "run-a")
    assert run_a["episode_count"] == 1
    assert run_a["excluded_episodes"] == [{"subagent_name": "site-massing", "reason": "teacher invoked 2 times"}]
    assert not (tmp_path / "dataset" / "submissions" / "run-a" / "ground_truth" / "site-massing.json").exists()


def test_duplicate_snapshots_export_once(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    shared = {"input_dir": "data/datasets/new_dmbp_data/sub-1", "sha": "same"}
    _make_run(runs_dir, "run-a", complete=False, **shared)
    _make_run(runs_dir, "run-b", **shared)
    _make_run(runs_dir, "run-c")

    manifest = export_dataset(runs_dir, tmp_path / "dataset", validation_submission_ids=["run-c"])

    submission = next(item for item in manifest["submissions"] if item["submission_id"] == "sub-1")
    assert submission["run_id"] == "run-b"
    assert submission["duplicate_run_ids"] == ["run-a"]
    assert manifest["submission_count"] == 2


def test_conflicting_snapshots_of_one_submission_fail(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "run-a", input_dir="data/sub-1", sha="one")
    _make_run(runs_dir, "run-b", input_dir="data/sub-1", sha="two")

    with pytest.raises(DatasetError, match="different input_set_sha256"):
        export_dataset(runs_dir, tmp_path / "dataset")


def test_incomplete_run_requires_explicit_opt_in(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "run-a")
    _make_run(runs_dir, "run-b", complete=False)

    with pytest.raises(DatasetError, match="Missing ground truth"):
        export_dataset(runs_dir, tmp_path / "strict", validation_submission_ids=["run-a"])

    manifest = export_dataset(
        runs_dir,
        tmp_path / "allowed",
        validation_submission_ids=["run-a"],
        allow_incomplete=True,
    )
    incomplete = next(item for item in manifest["submissions"] if item["submission_id"] == "run-b")
    assert incomplete["status"] == "incomplete"
    assert incomplete["episode_count"] == 0
    assert incomplete["missing_ground_truth_categories"] == ["site-massing", "room-geometry"]
