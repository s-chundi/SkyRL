"""Convert DMBP Villas run directories to and from a training dataset.

The exported dataset keeps model-visible inputs under ``submissions/`` and
ground truth beside, rather than inside, the agent context. The workspace keeps
only the read-only files staged before extraction starts. Each split file has
one JSON object per submission and extraction subagent.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
DATA_SOURCE = "dmbp_villas"
DEFAULT_VALIDATION_FRACTION = 0.2
WORKSPACE_INPUT_FILES = ("policy_constraints.md", "policy_dispatch.json")


class DatasetError(ValueError):
    """Raised when a source run or exported dataset is invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        stream = path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DatasetError(f"Missing split file: {path}") from exc
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"Expected an object in {path}:{line_number}")
            rows.append(row)
    return rows


def _require_new_directory(path: Path) -> None:
    if path.exists():
        raise DatasetError(f"Output path already exists: {path}")
    path.mkdir(parents=True)


def _safe_name(value: str, field: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise DatasetError(f"Invalid {field}: {value!r}")
    return value


def _discover_contexts(runs_dir: Path, run_ids: list[str] | None) -> list[Path]:
    if run_ids:
        contexts = [runs_dir / _safe_name(run_id, "run id") / "agent_context" for run_id in run_ids]
    else:
        contexts = sorted(runs_dir.glob("*/agent_context"))
    if not contexts:
        raise DatasetError(f"No */agent_context directories found under {runs_dir}")
    missing = [path for path in contexts if not path.is_dir()]
    if missing:
        raise DatasetError(f"Missing agent context: {missing[0]}")
    return sorted(contexts, key=lambda path: path.parent.name)


def _choose_validation_ids(run_ids: list[str], requested: list[str] | None) -> set[str]:
    if requested:
        validation_ids = {_safe_name(run_id, "validation run id") for run_id in requested}
        unknown = validation_ids.difference(run_ids)
        if unknown:
            raise DatasetError(f"Validation run ids were not selected for export: {sorted(unknown)}")
        if len(validation_ids) == len(run_ids):
            raise DatasetError("The training split would be empty")
        return validation_ids

    validation_count = max(1, round(len(run_ids) * DEFAULT_VALIDATION_FRACTION))
    if validation_count == len(run_ids):
        raise DatasetError("At least two runs are required to create train and validation splits")
    return set(sorted(run_ids)[-validation_count:])


def _copy_workspace_inputs(source_workspace: Path, output_workspace: Path) -> None:
    output_workspace.mkdir(parents=True)
    for file_name in WORKSPACE_INPUT_FILES:
        source = source_workspace / file_name
        if not source.is_file():
            raise DatasetError(f"Missing workspace input file: {source}")
        shutil.copy2(source, output_workspace / file_name)


def _load_categories(context: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dispatch_path = context / "workspace" / "policy_dispatch.json"
    dispatch = _read_json(dispatch_path)
    categories = dispatch.get("categories")
    if not isinstance(categories, list) or not categories:
        raise DatasetError(f"No categories found in {dispatch_path}")

    seen_ids: set[str] = set()
    seen_agents: set[str] = set()
    normalized = []
    for category in categories:
        if not isinstance(category, dict):
            raise DatasetError(f"Invalid category in {dispatch_path}")
        category_id = _safe_name(str(category.get("category_id", "")), "category id")
        subagent_name = _safe_name(str(category.get("extraction_agent", "")), "subagent name")
        if category_id in seen_ids:
            raise DatasetError(f"Duplicate category id {category_id!r} in {dispatch_path}")
        if subagent_name in seen_agents:
            raise DatasetError(f"Duplicate extraction agent {subagent_name!r} in {dispatch_path}")
        if not isinstance(category.get("rule_keys"), list):
            raise DatasetError(f"Category {category_id!r} has no rule_keys list")
        seen_ids.add(category_id)
        seen_agents.add(subagent_name)
        normalized.append(category)
    normalized.sort(key=lambda category: (category.get("position", 0), category["category_id"]))
    return dispatch, normalized


def _load_ground_truth(context: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    ground_truth_dir = context / "workspace" / "structured_findings"
    if not ground_truth_dir.is_dir():
        return {}

    ground_truth = {}
    for path in sorted(ground_truth_dir.glob("*.json")):
        value = _read_json(path)
        category_id = _safe_name(str(value.get("category_id", "")), "ground-truth category id")
        if category_id in ground_truth:
            raise DatasetError(f"Duplicate ground truth for category {category_id!r} in {ground_truth_dir}")
        ground_truth[category_id] = (path, value)
    return ground_truth


def _validate_ground_truth(
    category: dict[str, Any], source_path: Path, ground_truth: dict[str, Any]
) -> None:
    category_id = category["category_id"]
    if ground_truth.get("position") != category.get("position"):
        raise DatasetError(f"Position mismatch for {category_id!r} in {source_path}")
    if ground_truth.get("findings_path") != category.get("findings_path"):
        raise DatasetError(f"Findings path mismatch for {category_id!r} in {source_path}")

    rules = ground_truth.get("rules")
    if not isinstance(rules, list):
        raise DatasetError(f"Ground truth has no rules list: {source_path}")
    actual_rule_keys = [rule.get("rule_key") for rule in rules if isinstance(rule, dict)]
    expected_rule_keys = category["rule_keys"]
    if len(actual_rule_keys) != len(rules) or set(actual_rule_keys) != set(expected_rule_keys):
        missing = sorted(set(expected_rule_keys).difference(actual_rule_keys))
        extra = sorted(set(actual_rule_keys).difference(expected_rule_keys))
        raise DatasetError(
            f"Rule mismatch for {category_id!r} in {source_path}; missing={missing}, extra={extra}"
        )


def export_dataset(
    runs_dir: Path,
    output_dir: Path,
    *,
    run_ids: list[str] | None = None,
    validation_run_ids: list[str] | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Export run directories into clean contexts, hidden labels, and JSONL splits."""
    contexts = _discover_contexts(runs_dir, run_ids)
    selected_ids = [context.parent.name for context in contexts]
    validation_ids = _choose_validation_ids(selected_ids, validation_run_ids)
    _require_new_directory(output_dir)

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    submissions = []

    try:
        for context in contexts:
            submission_id = _safe_name(context.parent.name, "submission id")
            split = "validation" if submission_id in validation_ids else "train"
            dispatch, categories = _load_categories(context)
            ground_truth = _load_ground_truth(context)

            expected_category_ids = {category["category_id"] for category in categories}
            unexpected = sorted(set(ground_truth).difference(expected_category_ids))
            if unexpected:
                raise DatasetError(f"Unexpected ground-truth categories for {submission_id}: {unexpected}")

            missing_categories = [
                category["category_id"] for category in categories if category["category_id"] not in ground_truth
            ]
            if missing_categories and not allow_incomplete:
                raise DatasetError(
                    f"Missing ground truth for {submission_id}: {missing_categories}. "
                    "Use --allow-incomplete to inventory the run without training rows."
                )

            submission_dir = output_dir / "submissions" / submission_id
            exported_context = submission_dir / "agent_context"
            for input_name in ("application", "policies"):
                source = context / input_name
                if not source.is_dir():
                    raise DatasetError(f"Missing model input directory: {source}")
                shutil.copytree(source, exported_context / input_name)
            _copy_workspace_inputs(context / "workspace", exported_context / "workspace")

            ground_truth_dir = submission_dir / "ground_truth"
            ground_truth_dir.mkdir()
            exported_count = 0
            for category in categories:
                category_id = category["category_id"]
                if category_id not in ground_truth:
                    continue
                source_path, label = ground_truth[category_id]
                _validate_ground_truth(category, source_path, label)

                subagent_name = category["extraction_agent"]
                label_path = ground_truth_dir / f"{subagent_name}.json"
                _write_json(label_path, label)
                relative_context = exported_context.relative_to(output_dir).as_posix()
                relative_label = label_path.relative_to(output_dir).as_posix()
                rows_by_split[split].append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "data_source": DATA_SOURCE,
                        "env_class": DATA_SOURCE,
                        "episode_id": f"{submission_id}:{subagent_name}",
                        "submission_id": submission_id,
                        "split": split,
                        "subagent_name": subagent_name,
                        "category_id": category_id,
                        "category_name": category.get("name"),
                        "position": category.get("position"),
                        "rulebook_id": dispatch.get("rulebook_id"),
                        "rule_keys": category["rule_keys"],
                        "agent_context_path": relative_context,
                        "ground_truth_path": relative_label,
                        "findings_path": category.get("findings_path"),
                        "scratch_path": category.get("scratch_path"),
                    }
                )
                exported_count += 1

            submissions.append(
                {
                    "submission_id": submission_id,
                    "split": split,
                    "status": "ready" if not missing_categories else "incomplete",
                    "agent_context_path": exported_context.relative_to(output_dir).as_posix(),
                    "ground_truth_dir": ground_truth_dir.relative_to(output_dir).as_posix(),
                    "episode_count": exported_count,
                    "missing_ground_truth_categories": missing_categories,
                }
            )

        for split, rows in rows_by_split.items():
            rows.sort(key=lambda row: (row["submission_id"], row["position"], row["subagent_name"]))
            _write_jsonl(output_dir / f"{split}.jsonl", rows)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset_name": DATA_SOURCE,
            "source_runs_dir": str(runs_dir.resolve()),
            "splits": {
                "train": sorted(set(selected_ids).difference(validation_ids)),
                "validation": sorted(validation_ids),
            },
            "submission_count": len(submissions),
            "episode_count": sum(len(rows) for rows in rows_by_split.values()),
            "submissions": submissions,
        }
        _write_json(output_dir / "dataset.json", manifest)
        validate_dataset(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def validate_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Validate dataset structure and model/ground-truth isolation."""
    manifest = _read_json(dataset_dir / "dataset.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DatasetError(f"Unsupported dataset schema version: {manifest.get('schema_version')}")

    submissions = manifest.get("submissions")
    if not isinstance(submissions, list):
        raise DatasetError("dataset.json has no submissions list")
    submission_by_id = {submission["submission_id"]: submission for submission in submissions}
    if len(submission_by_id) != len(submissions):
        raise DatasetError("dataset.json contains duplicate submission ids")

    all_rows = []
    for split in ("train", "validation"):
        rows = _read_jsonl(dataset_dir / f"{split}.jsonl")
        for row in rows:
            if row.get("split") != split:
                raise DatasetError(f"Row {row.get('episode_id')} is in the wrong split file")
        all_rows.extend(rows)

    episode_ids = [row.get("episode_id") for row in all_rows]
    if len(set(episode_ids)) != len(episode_ids):
        raise DatasetError("Split files contain duplicate episode ids")

    for submission_id, submission in submission_by_id.items():
        context = dataset_dir / submission["agent_context_path"]
        for input_name in ("application", "policies"):
            if not (context / input_name).is_dir():
                raise DatasetError(f"Missing {input_name} directory for {submission_id}")
        workspace = context / "workspace"
        workspace_names = sorted(path.name for path in workspace.iterdir()) if workspace.is_dir() else None
        if workspace_names != sorted(WORKSPACE_INPUT_FILES):
            raise DatasetError(
                f"Model-visible workspace for {submission_id} must contain only "
                f"{list(WORKSPACE_INPUT_FILES)}; found {workspace_names}"
            )
        for excluded_name in ("logs", ".agents", ".claude"):
            if (context / excluded_name).exists():
                raise DatasetError(f"Excluded path present for {submission_id}: {excluded_name}")

    for row in all_rows:
        submission = submission_by_id.get(row.get("submission_id"))
        if submission is None:
            raise DatasetError(f"Unknown submission in row {row.get('episode_id')}")
        if submission["split"] != row["split"]:
            raise DatasetError(f"Submission split mismatch in row {row.get('episode_id')}")
        context = dataset_dir / row["agent_context_path"]
        if context != dataset_dir / submission["agent_context_path"]:
            raise DatasetError(f"Context mismatch in row {row.get('episode_id')}")
        label_path = dataset_dir / row["ground_truth_path"]
        label = _read_json(label_path)
        if label.get("category_id") != row.get("category_id"):
            raise DatasetError(f"Ground-truth category mismatch in row {row.get('episode_id')}")

    if manifest.get("episode_count") != len(all_rows):
        raise DatasetError("dataset.json episode_count does not match split files")
    return manifest


def materialize_runs(
    dataset_dir: Path,
    output_runs_dir: Path,
    *,
    restore_ground_truth: bool = False,
) -> None:
    """Materialize clean run directories from an exported dataset."""
    manifest = validate_dataset(dataset_dir)
    _require_new_directory(output_runs_dir)
    rows = _read_jsonl(dataset_dir / "train.jsonl") + _read_jsonl(dataset_dir / "validation.jsonl")
    rows_by_submission: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_submission.setdefault(row["submission_id"], []).append(row)

    try:
        for submission in manifest["submissions"]:
            submission_id = submission["submission_id"]
            source_context = dataset_dir / submission["agent_context_path"]
            output_context = output_runs_dir / submission_id / "agent_context"
            for input_name in ("application", "policies"):
                shutil.copytree(source_context / input_name, output_context / input_name)
            workspace = output_context / "workspace"
            _copy_workspace_inputs(source_context / "workspace", workspace)
            if restore_ground_truth:
                restored_dir = workspace / "structured_findings"
                for row in rows_by_submission.get(submission_id, []):
                    restored_dir.mkdir(exist_ok=True)
                    shutil.copy2(
                        dataset_dir / row["ground_truth_path"],
                        restored_dir / f"{row['subagent_name']}.json",
                    )
    except Exception:
        shutil.rmtree(output_runs_dir, ignore_errors=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Convert run directories to the dataset format")
    export_parser.add_argument("--runs-dir", type=Path, required=True)
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--run-id", action="append", dest="run_ids")
    export_parser.add_argument("--validation-run-id", action="append", dest="validation_run_ids")
    export_parser.add_argument("--allow-incomplete", action="store_true")

    materialize_parser = subparsers.add_parser(
        "materialize", help="Convert the dataset format to clean run directories"
    )
    materialize_parser.add_argument("--dataset-dir", type=Path, required=True)
    materialize_parser.add_argument("--output-runs-dir", type=Path, required=True)
    materialize_parser.add_argument("--restore-ground-truth", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate an exported dataset")
    validate_parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        if args.command == "export":
            manifest = export_dataset(
                args.runs_dir,
                args.output_dir,
                run_ids=args.run_ids,
                validation_run_ids=args.validation_run_ids,
                allow_incomplete=args.allow_incomplete,
            )
            print(
                f"Exported {manifest['episode_count']} episodes from "
                f"{manifest['submission_count']} submissions to {args.output_dir}"
            )
        elif args.command == "materialize":
            materialize_runs(
                args.dataset_dir,
                args.output_runs_dir,
                restore_ground_truth=args.restore_ground_truth,
            )
            print(f"Materialized run directories at {args.output_runs_dir}")
        else:
            manifest = validate_dataset(args.dataset_dir)
            print(
                f"Valid dataset: {manifest['episode_count']} episodes across "
                f"{manifest['submission_count']} submissions"
            )
    except DatasetError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
