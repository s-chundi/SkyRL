"""Convert DMBP Villas run directories to and from a training dataset.

The exported dataset keeps model-visible inputs under ``submissions/`` and
ground truth, teacher traces, and leaf specs beside, rather than inside, the
agent context.
The workspace keeps only the read-only files staged before extraction starts.
Each split file has one JSON object per submission and extraction subagent.

A submission is one input snapshot (``metadata.json`` ``input_set_sha256``) and
is named after its ``input_dir``, or after its run id when ``input_dir`` is null.
When several runs share a snapshot, one run is exported and the others are
recorded as duplicates.

Leaf specs (the rendered instructions, task message, tools, and path scopes of
each episode's subagent) come from the ips-applications loaders, so they are
rendered in that checkout's environment by ``render_leaf_specs.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

from inventory import build_inventory
from teacher_trace import TraceError, extract_subagents

SCHEMA_VERSION = 3
DATA_SOURCE = "dmbp_villas"
DEFAULT_VALIDATION_FRACTION = 0.2
WORKSPACE_INPUT_FILES = ("intake_check.md", "policy_constraints.md", "policy_dispatch.json")
TRACE_PATH = Path("agent_context") / "logs" / "trace.jsonl"
PERMITS_PROJECT = Path("applications") / "permits-demo"
RENDER_SCRIPT = Path(__file__).resolve().parent / "render_leaf_specs.py"


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


def _choose_validation_ids(submission_ids: list[str], requested: list[str] | None) -> set[str]:
    if requested:
        validation_ids = {_safe_name(submission_id, "validation submission id") for submission_id in requested}
        unknown = validation_ids.difference(submission_ids)
        if unknown:
            raise DatasetError(f"Validation submission ids were not selected for export: {sorted(unknown)}")
        if len(validation_ids) == len(submission_ids):
            raise DatasetError("The training split would be empty")
        return validation_ids

    validation_count = max(1, round(len(submission_ids) * DEFAULT_VALIDATION_FRACTION))
    if validation_count == len(submission_ids):
        raise DatasetError("At least two submissions are required to create train and validation splits")
    return set(sorted(submission_ids)[-validation_count:])


def _load_run(context: Path) -> dict[str, Any]:
    run_dir = context.parent
    run_id = _safe_name(run_dir.name, "run id")
    metadata = _read_json(run_dir / "metadata.json")
    if not metadata.get("input_set_sha256"):
        raise DatasetError(f"Run {run_id} has no input_set_sha256 in metadata.json; it cannot be deduplicated")
    input_dir = metadata.get("input_dir")
    submission_id = _safe_name(Path(input_dir).name, "submission id") if input_dir else run_id
    dispatch, categories = _load_categories(context)
    ground_truth = _load_ground_truth(context)
    unexpected = sorted(set(ground_truth).difference(category["category_id"] for category in categories))
    if unexpected:
        raise DatasetError(f"Unexpected ground-truth categories for run {run_id}: {unexpected}")
    trace_path = run_dir / TRACE_PATH
    return {
        "run_id": run_id,
        "context": context,
        "submission_id": submission_id,
        "input_dir": input_dir,
        "input_set_sha256": metadata.get("input_set_sha256"),
        "dispatch": dispatch,
        "categories": categories,
        "ground_truth": ground_truth,
        "missing_categories": [c["category_id"] for c in categories if c["category_id"] not in ground_truth],
        "trace_path": trace_path if trace_path.is_file() else None,
    }


def _deduplicate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one run per input snapshot: labeled, then traced, then smallest run id."""
    by_snapshot: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_snapshot.setdefault(run["input_set_sha256"], []).append(run)

    selected = []
    for group in by_snapshot.values():
        group.sort(key=lambda run: (bool(run["missing_categories"]), run["trace_path"] is None, run["run_id"]))
        chosen = {**group[0], "duplicate_run_ids": [run["run_id"] for run in group[1:]]}
        selected.append(chosen)

    by_submission: dict[str, list[str]] = {}
    for run in selected:
        by_submission.setdefault(run["submission_id"], []).append(run["run_id"])
    conflicts = {submission: ids for submission, ids in by_submission.items() if len(ids) > 1}
    if conflicts:
        raise DatasetError(
            f"Runs share an input_dir but have different input_set_sha256 values: {conflicts}. "
            "Select one snapshot per submission with --run-id."
        )
    return sorted(selected, key=lambda run: run["submission_id"])


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


def _validate_ground_truth(category: dict[str, Any], source_path: Path, ground_truth: dict[str, Any]) -> None:
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
        raise DatasetError(f"Rule mismatch for {category_id!r} in {source_path}; missing={missing}, extra={extra}")


def export_dataset(
    runs_dir: Path,
    output_dir: Path,
    *,
    run_ids: list[str] | None = None,
    validation_submission_ids: list[str] | None = None,
    allow_incomplete: bool = False,
    ips_applications: Path | None = None,
) -> dict[str, Any]:
    """Export run directories into clean contexts, hidden labels and teacher traces, and JSONL splits.

    With ``ips_applications``, also render each episode's leaf spec from that checkout.
    """
    runs = _deduplicate([_load_run(context) for context in _discover_contexts(runs_dir, run_ids)])
    submission_ids = [run["submission_id"] for run in runs]
    validation_ids = _choose_validation_ids(submission_ids, validation_submission_ids)
    incomplete = {run["submission_id"]: run["missing_categories"] for run in runs if run["missing_categories"]}
    if incomplete and not allow_incomplete:
        raise DatasetError(
            f"Missing ground truth: {incomplete}. Use --allow-incomplete to inventory these runs without training rows."
        )
    _require_new_directory(output_dir)

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    submissions = []

    try:
        for run in runs:
            submission_id = run["submission_id"]
            split = "validation" if submission_id in validation_ids else "train"
            context = run["context"]
            try:
                teacher = extract_subagents(run["trace_path"], run["run_id"]) if run["trace_path"] else {}
            except TraceError as exc:
                raise DatasetError(f"Cannot read teacher trace for run {run['run_id']}: {exc}") from exc

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
            teacher_dir = submission_dir / "teacher"
            exported_count = 0
            missing_teacher = []
            excluded = []
            for category in run["categories"]:
                category_id = category["category_id"]
                if category_id not in run["ground_truth"]:
                    continue
                source_path, label = run["ground_truth"][category_id]
                _validate_ground_truth(category, source_path, label)

                subagent_name = category["extraction_agent"]
                invocations = teacher.get(subagent_name, {}).get("stats", {}).get("invocations", 1)
                if invocations > 1:
                    # The re-launched session starts from the first session's findings file, unlike an episode.
                    excluded.append({"subagent_name": subagent_name, "reason": f"teacher invoked {invocations} times"})
                    continue
                label_path = ground_truth_dir / f"{subagent_name}.json"
                _write_json(label_path, label)
                teacher_path = None
                if subagent_name in teacher:
                    teacher_path = (teacher_dir / f"{subagent_name}.json").relative_to(output_dir).as_posix()
                    _write_json(output_dir / teacher_path, teacher[subagent_name])
                else:
                    missing_teacher.append(subagent_name)
                rows_by_split[split].append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "data_source": DATA_SOURCE,
                        "env_class": DATA_SOURCE,
                        "episode_id": f"{submission_id}:{subagent_name}",
                        "submission_id": submission_id,
                        "run_id": run["run_id"],
                        "split": split,
                        "subagent_name": subagent_name,
                        "category_id": category_id,
                        "category_name": category.get("name"),
                        "position": category.get("position"),
                        "rulebook_id": run["dispatch"].get("rulebook_id"),
                        "rule_keys": category["rule_keys"],
                        "agent_context_path": exported_context.relative_to(output_dir).as_posix(),
                        "ground_truth_path": label_path.relative_to(output_dir).as_posix(),
                        "teacher_path": teacher_path,
                        "leaf_spec_path": None,
                        "findings_path": category.get("findings_path"),
                        "scratch_path": category.get("scratch_path"),
                    }
                )
                exported_count += 1

            submissions.append(
                {
                    "submission_id": submission_id,
                    "run_id": run["run_id"],
                    "duplicate_run_ids": run["duplicate_run_ids"],
                    "input_dir": run["input_dir"],
                    "input_set_sha256": run["input_set_sha256"],
                    "split": split,
                    "status": "ready" if not run["missing_categories"] else "incomplete",
                    "agent_context_path": exported_context.relative_to(output_dir).as_posix(),
                    "ground_truth_dir": ground_truth_dir.relative_to(output_dir).as_posix(),
                    "has_trace": run["trace_path"] is not None,
                    "episode_count": exported_count,
                    "missing_ground_truth_categories": run["missing_categories"],
                    "missing_teacher_subagents": missing_teacher,
                    "excluded_episodes": excluded,
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
                "train": sorted(set(submission_ids).difference(validation_ids)),
                "validation": sorted(validation_ids),
            },
            "submission_count": len(submissions),
            "episode_count": sum(len(rows) for rows in rows_by_split.values()),
            "submissions": submissions,
        }
        _write_json(output_dir / "dataset.json", manifest)
        if ips_applications is not None:
            return add_leaf_specs(output_dir, ips_applications)
        validate_dataset(output_dir)
        build_inventory(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def add_leaf_specs(dataset_dir: Path, ips_applications: Path) -> dict[str, Any]:
    """Render every episode's leaf spec with the ips-applications loaders and link it from its row."""
    project = (ips_applications / PERMITS_PROJECT).resolve()
    if not (project / "pyproject.toml").is_file():
        raise DatasetError(f"Not an ips-applications checkout: {ips_applications}")
    env = {
        **os.environ,
        "PERMITS_USE_CASE_PACKAGE": "services.permits",
        "PYTHONPATH": str(project),
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    command = ["uv", "run", "--no-sync", "--project", str(project), "python", str(RENDER_SCRIPT)]
    result = subprocess.run(
        [*command, "--dataset-dir", str(dataset_dir.resolve())],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise DatasetError(f"Rendering leaf specs failed:\n{result.stderr[-4000:]}")

    manifest = _read_json(dataset_dir / "dataset.json")
    sources = set()
    for split in ("train", "validation"):
        rows = _read_jsonl(dataset_dir / f"{split}.jsonl")
        for row in rows:
            spec_path = Path(row["agent_context_path"]).parent / "leaf_specs" / f"{row['subagent_name']}.json"
            if not (dataset_dir / spec_path).is_file():
                raise DatasetError(f"No leaf spec was rendered for {row['episode_id']}")
            row["leaf_spec_path"] = spec_path.as_posix()
            sources.add(json.dumps(_read_json(dataset_dir / spec_path)["source"], sort_keys=True))
        _write_jsonl(dataset_dir / f"{split}.jsonl", rows)
    manifest["leaf_spec_sources"] = [json.loads(source) for source in sorted(sources)]
    _write_json(dataset_dir / "dataset.json", manifest)
    validate_dataset(dataset_dir)
    build_inventory(dataset_dir)
    return manifest


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
        if row.get("teacher_path") is not None:
            teacher = _read_json(dataset_dir / row["teacher_path"])
            if teacher.get("subagent_name") != row.get("subagent_name") or teacher.get("run_id") != row.get("run_id"):
                raise DatasetError(f"Teacher trace mismatch in row {row.get('episode_id')}")
        if row.get("leaf_spec_path") is not None:
            _validate_leaf_spec(row, _read_json(dataset_dir / row["leaf_spec_path"]))

    if manifest.get("episode_count") != len(all_rows):
        raise DatasetError("dataset.json episode_count does not match split files")
    return manifest


def _validate_leaf_spec(row: dict[str, Any], spec: dict[str, Any]) -> None:
    category = spec.get("category") or {}
    expected = {
        "name": row.get("subagent_name"),
        "rulebook_id": row.get("rulebook_id"),
        "category_id": row.get("category_id"),
        "rule_keys": row.get("rule_keys"),
        "findings_path": row.get("findings_path"),
        "scratch_path": row.get("scratch_path"),
    }
    actual = {key: spec.get(key) if key in ("name", "rulebook_id") else category.get(key) for key in expected}
    mismatched = sorted(key for key in expected if actual[key] != expected[key])
    if mismatched:
        raise DatasetError(f"Leaf spec mismatch in row {row.get('episode_id')}: {mismatched}")


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
    export_parser.add_argument("--validation-submission-id", action="append", dest="validation_submission_ids")
    export_parser.add_argument("--allow-incomplete", action="store_true")
    export_parser.add_argument(
        "--ips-applications", type=Path, help="Render leaf specs with this ips-applications checkout"
    )

    leaf_spec_parser = subparsers.add_parser("leaf-specs", help="Render leaf specs for an exported dataset")
    leaf_spec_parser.add_argument("--dataset-dir", type=Path, required=True)
    leaf_spec_parser.add_argument("--ips-applications", type=Path, required=True)

    materialize_parser = subparsers.add_parser(
        "materialize", help="Convert the dataset format to clean run directories"
    )
    materialize_parser.add_argument("--dataset-dir", type=Path, required=True)
    materialize_parser.add_argument("--output-runs-dir", type=Path, required=True)
    materialize_parser.add_argument("--restore-ground-truth", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate an exported dataset")
    validate_parser.add_argument("--dataset-dir", type=Path, required=True)

    inventory_parser = subparsers.add_parser("inventory", help="Regenerate inventory tables for an exported dataset")
    inventory_parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        if args.command == "export":
            manifest = export_dataset(
                args.runs_dir,
                args.output_dir,
                run_ids=args.run_ids,
                validation_submission_ids=args.validation_submission_ids,
                allow_incomplete=args.allow_incomplete,
                ips_applications=args.ips_applications,
            )
            print(
                f"Exported {manifest['episode_count']} episodes from "
                f"{manifest['submission_count']} submissions to {args.output_dir}"
            )
        elif args.command == "leaf-specs":
            manifest = add_leaf_specs(args.dataset_dir, args.ips_applications)
            print(f"Rendered leaf specs for {manifest['episode_count']} episodes in {args.dataset_dir}")
        elif args.command == "materialize":
            materialize_runs(
                args.dataset_dir,
                args.output_runs_dir,
                restore_ground_truth=args.restore_ground_truth,
            )
            print(f"Materialized run directories at {args.output_runs_dir}")
        elif args.command == "inventory":
            validate_dataset(args.dataset_dir)
            report = build_inventory(args.dataset_dir)
            print(f"Wrote inventory to {report.parent}")
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
