import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from prepare_dataset import DatasetError, export_dataset, materialize_runs


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_run(runs_dir: Path, run_id: str, *, complete: bool = True) -> None:
    context = runs_dir / run_id / "agent_context"
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
    (context / "workspace" / "intake_check.md").write_text("ignored", encoding="utf-8")

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
                        }
                    ],
                },
            )


def test_export_and_materialize_keep_ground_truth_hidden(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "run-a")
    _make_run(runs_dir, "run-b")
    dataset_dir = tmp_path / "dataset"

    manifest = export_dataset(
        runs_dir,
        dataset_dir,
        validation_run_ids=["run-b"],
    )

    assert manifest["episode_count"] == 4
    assert manifest["splits"] == {"train": ["run-a"], "validation": ["run-b"]}
    exported_context = dataset_dir / "submissions" / "run-a" / "agent_context"
    assert sorted(path.name for path in exported_context.iterdir()) == ["application", "policies", "workspace"]
    assert sorted(path.name for path in (exported_context / "workspace").iterdir()) == [
        "policy_constraints.md",
        "policy_dispatch.json",
    ]
    assert (dataset_dir / "submissions" / "run-a" / "ground_truth" / "site-massing.json").is_file()

    materialized_dir = tmp_path / "materialized"
    materialize_runs(dataset_dir, materialized_dir)
    materialized_workspace = materialized_dir / "run-a" / "agent_context" / "workspace"
    assert sorted(path.name for path in materialized_workspace.iterdir()) == [
        "policy_constraints.md",
        "policy_dispatch.json",
    ]

    restored_dir = tmp_path / "restored"
    materialize_runs(dataset_dir, restored_dir, restore_ground_truth=True)
    assert (
        restored_dir
        / "run-a"
        / "agent_context"
        / "workspace"
        / "structured_findings"
        / "site-massing.json"
    ).is_file()


def test_incomplete_run_requires_explicit_opt_in(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "run-a")
    _make_run(runs_dir, "run-b", complete=False)

    with pytest.raises(DatasetError, match="Missing ground truth"):
        export_dataset(runs_dir, tmp_path / "strict", validation_run_ids=["run-a"])

    manifest = export_dataset(
        runs_dir,
        tmp_path / "allowed",
        validation_run_ids=["run-a"],
        allow_incomplete=True,
    )
    incomplete = next(item for item in manifest["submissions"] if item["submission_id"] == "run-b")
    assert incomplete["status"] == "incomplete"
    assert incomplete["episode_count"] == 0
    assert incomplete["missing_ground_truth_categories"] == ["site-massing", "room-geometry"]
