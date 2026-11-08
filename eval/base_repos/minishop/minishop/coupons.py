"""Coupon rules. Codes are case-insensitive."""
from __future__ import annotations

from decimal import Decimal

from minishop.models import Order
from minishop.money import percent_of, to_money

MIN_SPEND_SAVE10 = Decimal("50.00")


def normalize_code(code: str | None) -> str:
    return (code or "").strip().upper()


def coupon_discount(order: Order) -> Decimal:
    """Discount granted by the order's coupon, applied before tax.

    - SAVE10: 10% off the items subtotal when it is at least 50.00.
    - BOGO-<SKU>: buy one get one free on that SKU, i.e. every second unit is free.
    - Unknown or empty codes give no discount.
    """
    code = normalize_code(order.coupon)
    if not code:
        return Decimal("0.00")
    if code == "SAVE10":
        subtotal = order.items_subtotal()
        if subtotal < MIN_SPEND_SAVE10:
            return Decimal("0.00")
        return percent_of(subtotal, 10)
    if code.startswith("BOGO-"):
        sku = code[len("BOGO-"):]
        for line in order.lines:
            if line.product.sku.upper() == sku:
                free_units = line.quantity // 2
                return to_money(line.product.unit_price * free_units)
        return Decimal("0.00")
    return Decimal("0.00")
