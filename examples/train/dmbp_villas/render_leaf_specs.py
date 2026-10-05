"""Render the leaf spec of every episode in an exported DMBP Villas dataset.

Runs inside the ips-applications ``permits-demo`` environment, because the spec comes
from the production loaders (``render_leaf_specs``). ``prepare_dataset.py leaf-specs``
starts it; run it directly only to debug rendering::

    PERMITS_USE_CASE_PACKAGE=services.permits PYTHONPATH=<ips>/applications/permits-demo \\
        uv run --no-sync --project <ips>/applications/permits-demo \\
        python render_leaf_specs.py --dataset-dir <dataset>

Writes ``submissions/<submission_id>/leaf_specs/<subagent_name>.json`` for every row.
Each file is the rendered spec plus ``source``: the git commit of the agent content
(prompts, subagents, config) and of the loader package, and whether either had
uncommitted changes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from permits.platform.atoms.agent._lib import _loaders
from permits.platform.atoms.agent._lib._agent_config import get_config
from permits.platform.atoms.agent._lib._loaders import render_leaf_specs

USE_CASE = "dmbp_villas"
SPEC_SCHEMA_VERSION = 1


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _git_state(path: Path) -> dict[str, object]:
    repo = Path(_git(path, "rev-parse", "--show-toplevel"))
    return {
        "path": path.resolve().relative_to(repo.resolve()).as_posix(),
        "git_commit": _git(repo, "rev-parse", "HEAD"),
        "dirty": bool(_git(repo, "status", "--porcelain", "--", str(path))),
    }


def _source(content_root: Path) -> dict[str, object]:
    return {"content": _git_state(content_root), "loaders": _git_state(Path(_loaders.__file__).parent)}


def render_dataset(dataset_dir: Path) -> int:
    config = get_config().with_use_case(USE_CASE)
    source = _source(config.content_root)
    rows = []
    for split in ("train", "validation"):
        lines = (dataset_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        rows += [json.loads(line) for line in lines if line.strip()]
    names_by_context: dict[str, list[str]] = {}
    for row in rows:
        names_by_context.setdefault(row["agent_context_path"], []).append(row["subagent_name"])

    written = 0
    for context_path, names in sorted(names_by_context.items()):
        context = dataset_dir / context_path
        specs = render_leaf_specs(config, context, names=names)
        output = context.parent / "leaf_specs"
        output.mkdir(exist_ok=True)
        for name, spec in specs.items():
            value = {"schema_version": SPEC_SCHEMA_VERSION, "source": source, **spec.model_dump(mode="json")}
            (output / f"{name}.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    print(f"Rendered {render_dataset(args.dataset_dir)} leaf specs")


if __name__ == "__main__":
    main()
