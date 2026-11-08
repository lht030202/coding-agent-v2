from __future__ import annotations

from pathlib import Path

import pytest

BUGGY_CALCULATOR = 'def subtract(a, b):\n    """Return a minus b."""\n    return a + b\n'
TEST_CALCULATOR = "from calculator import subtract\n\n\ndef test_subtract():\n    assert subtract(5, 3) == 2\n"


@pytest.fixture
def buggy_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "calculator.py").write_text(BUGGY_CALCULATOR, encoding="utf-8")
    (repo / "tests" / "test_calculator.py").write_text(TEST_CALCULATOR, encoding="utf-8")
    return repo
