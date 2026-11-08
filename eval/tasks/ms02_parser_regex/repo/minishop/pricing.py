"""Order pricing: discounts first, then tax on the discounted amount."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from minishop.coupons import coupon_discount
from minishop.models import Order
from minishop.money import percent_of, to_money

TIER_DISCOUNT_PERCENT = {"standard": 0, "silver": 5, "gold": 10}
BULK_THRESHOLD = 10  # units of one product
BULK_DISCOUNT_PERCENT = 3
TAX_PERCENT_BY_CATEGORY = {"food": 5, "books": 0}
DEFAULT_TAX_PERCENT = 8


@dataclass(frozen=True)
class PriceBreakdown:
    subtotal: Decimal
    discount: Decimal
    tax: Decimal
    total: Decimal


def bulk_discount(order: Order) -> Decimal:
    """3% off every line that has at least BULK_THRESHOLD units."""
    total = Decimal("0")
    for line in order.lines:
        if line.quantity >= BULK_THRESHOLD:
            total += percent_of(line.subtotal, BULK_DISCOUNT_PERCENT)
    return to_money(total)


def tier_discount(order: Order) -> Decimal:
    percent = TIER_DISCOUNT_PERCENT.get(order.customer_tier, 0)
    return percent_of(order.items_subtotal(), percent)


def tax_for(order: Order, discount: Decimal) -> Decimal:
    """Tax is charged per line category on the discounted amount; the discount is spread proportionally."""
    subtotal = order.items_subtotal()
    if subtotal == 0:
        return Decimal("0.00")
    tax = Decimal("0")
    for line in order.lines:
        share = line.subtotal / subtotal
        taxable = line.subtotal - discount * share
        percent = TAX_PERCENT_BY_CATEGORY.get(line.product.category, DEFAULT_TAX_PERCENT)
        tax += taxable * Decimal(percent) / Decimal(100)
    return to_money(tax)


def price_order(order: Order) -> PriceBreakdown:
    subtotal = order.items_subtotal()
    discount = to_money(tier_discount(order) + bulk_discount(order) + coupon_discount(order))
    discount = min(discount, subtotal)
    tax = tax_for(order, discount)
    total = to_money(subtotal - discount + tax)
    return PriceBreakdown(subtotal=subtotal, discount=discount, tax=tax, total=total)
