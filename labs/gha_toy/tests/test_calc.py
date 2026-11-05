import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calc import add, divide  # noqa: E402


def test_add():
    # RED run: change the expected value to 5, push, watch both matrix jobs fail.
    # GREEN run: put it back to 4, push again.
    assert add(2, 2) == 4


def test_divide():
    assert divide(9, 3) == 3
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
