"""The notebooks are what this repository ships, so check what a reader will see in them.

Every notebook must parse, have every code cell executed in order with the current code
(outputs from an older version of a cell are worse than none), show no error output, use
gymnasium rather than the retired gym package, and import only what requirements.txt and
pyproject.toml declare. A cell that must not run inside Jupyter (A3C's multiprocessing
training, which `a3c_training.py` runs instead) carries the `skip-execution` tag.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(ROOT.glob("[0-9][0-9]_*.ipynb"))
SKIP_TAG = "skip-execution"
# Import name -> distribution name, where they differ.
DISTRIBUTION = {"gymnasium": "gymnasium", "matplotlib": "matplotlib", "numpy": "numpy", "torch": "torch"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code" and "".join(c["source"]).strip()]


def _declared() -> set[str]:
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    names = {re.split(r"[\s#<>=~!\[]", line.strip(), maxsplit=1)[0].lower() for line in req.splitlines() if line.strip()}
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    deps = re.search(r"^dependencies = \[(.*?)\]", pyproject, re.S | re.M)
    names |= {re.split(r"[<>=~!\[ ]", d.strip().strip('"'), maxsplit=1)[0].lower() for d in deps.group(1).split(",") if d.strip()}
    return {n for n in names if n}


def _third_party_imports(code: str) -> set[str]:
    tree = ast.parse(code)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return {r for r in roots if r not in sys.stdlib_module_names}


def test_there_are_notebooks() -> None:
    assert len(NOTEBOOKS) >= 18


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_every_code_cell_was_executed_in_order(path: Path) -> None:
    nb = _load(path)
    counts = []
    for i, cell in enumerate(_code_cells(nb)):
        if SKIP_TAG in cell.get("metadata", {}).get("tags", []):
            continue
        assert cell.get("execution_count"), f"{path.name}: code cell {i} was never executed"
        counts.append(cell["execution_count"])
    assert counts == sorted(counts), f"{path.name}: cells were executed out of order: {counts}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_no_error_outputs(path: Path) -> None:
    for i, cell in enumerate(_code_cells(_load(path))):
        errors = [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]
        assert not errors, f"{path.name}: code cell {i} shows {errors[0].get('ename')}: {errors[0].get('evalue')}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_markdown_math_is_not_mangled(path: Path) -> None:
    # A single backslash in the JSON source is read as an escape: \theta becomes a tab
    # followed by "heta", and \nabla a line break followed by "abla".
    for i, cell in enumerate(_load(path)["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        bad = re.search(r"[\x00-\x09\x0b-\x1f]|\n(?=abla)", "".join(cell["source"]))
        assert not bad, f"{path.name}: markdown cell {i} has {bad.group()!r}; write the backslash twice in the JSON"


@pytest.mark.parametrize("path", [*NOTEBOOKS, ROOT / "a3c_training.py"], ids=lambda p: p.stem)
def test_imports_are_declared_and_gym_is_gymnasium(path: Path) -> None:
    if path.suffix == ".ipynb":
        lines = []
        for cell in _code_cells(_load(path)):
            lines += [ln for ln in "".join(cell["source"]).splitlines() if not ln.lstrip().startswith(("%", "!"))]
        code = "\n".join(lines)
    else:
        code = path.read_text(encoding="utf-8")
    imports = _third_party_imports(code)
    assert "gym" not in imports, f"{path.name} imports the retired gym package; use gymnasium"
    undeclared = {i for i in imports if DISTRIBUTION.get(i, i).lower() not in _declared()}
    assert not undeclared, f"{path.name} imports {sorted(undeclared)}, which requirements.txt does not declare"
