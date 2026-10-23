"""Cheap checks on sacco.ipynb that don't need a Jupyter kernel."""
import ast
import json

import pytest

from conftest import ROOT

NOTEBOOK = json.loads((ROOT / "sacco.ipynb").read_text(encoding="utf-8"))
CODE_CELLS = ["".join(cell["source"]) for cell in NOTEBOOK["cells"] if cell["cell_type"] == "code"]


def python_only(source: str) -> str:
    """The cell without IPython magics (%pip, %autoreload, ...), which aren't Python."""
    return "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))


@pytest.mark.parametrize("number, source", list(enumerate(CODE_CELLS)))
def test_code_cells_are_valid_python(number, source):
    ast.parse(python_only(source), filename=f"code cell {number}")


def test_model_is_saved_in_the_models_folder():
    assert any('Path("models/pd_model.joblib")' in source for source in CODE_CELLS)


def test_every_src_import_exists():
    for source in CODE_CELLS:
        for node in ast.walk(ast.parse(python_only(source))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src"):
                module = __import__(node.module, fromlist=[alias.name for alias in node.names])
                for alias in node.names:
                    assert hasattr(module, alias.name), f"{node.module} has no {alias.name}"
