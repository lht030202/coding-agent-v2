import pytest

from minishop.parser import parse_order_text


def test_parses_lines_and_ignores_comments():
    text = "# weekly order\nTEA-01 x2\n\nMUG-02 x1  # gift\n"
    assert parse_order_text(text) == [("TEA-01", 2), ("MUG-02", 1)]


def test_rejects_bad_lines_with_line_number():
    with pytest.raises(ValueError, match="line 2"):
        parse_order_text("TEA-01 x2\nnot an order line\n")
