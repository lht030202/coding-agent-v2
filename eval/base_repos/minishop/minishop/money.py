"""Money helpers. Every amount is a Decimal rounded to cents."""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def to_money(value: int | float | str | Decimal) -> Decimal:
    """Convert a number to a Decimal rounded to cents, rounding halves up (2.675 -> 2.68)."""
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


def format_money(amount: Decimal, currency: str = "USD") -> str:
    """Format an amount like 'USD 1,234.50'."""
    return f"{currency} {to_money(amount):,.2f}"


def percent_of(amount: Decimal, percent: int | float | str | Decimal) -> Decimal:
    """Return `percent` % of `amount`, rounded to cents."""
    return to_money(Decimal(str(amount)) * Decimal(str(percent)) / Decimal(100))
