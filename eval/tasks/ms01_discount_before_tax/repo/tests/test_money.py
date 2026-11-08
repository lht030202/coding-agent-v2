from decimal import Decimal

from minishop.money import format_money, percent_of, to_money


def test_to_money_rounds_half_up():
    assert to_money("2.675") == Decimal("2.68")
    assert to_money(1.005) == Decimal("1.01")


def test_to_money_keeps_cents():
    assert to_money("19.99") == Decimal("19.99")
    assert to_money(3) == Decimal("3.00")


def test_format_money():
    assert format_money(Decimal("1234.5")) == "USD 1,234.50"


def test_percent_of():
    assert percent_of(Decimal("200.00"), 10) == Decimal("20.00")
