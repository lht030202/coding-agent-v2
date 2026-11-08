from decimal import Decimal

from minishop.models import Order, Product
from minishop.pricing import price_order

TEA = Product("TEA-01", "Green Tea", Decimal("3.50"), "food")
MUG = Product("MUG-02", "Travel Mug", Decimal("12.00"))


def make(*items, tier="standard", coupon=None):
    order = Order(customer_tier=tier, coupon=coupon)
    for product, quantity in items:
        order.add(product, quantity)
    return order


def test_standard_order_tax_by_category():
    breakdown = price_order(make((TEA, 2), (MUG, 1)))
    assert breakdown.subtotal == Decimal("19.00")
    assert breakdown.discount == Decimal("0.00")
    assert breakdown.tax == Decimal("1.31")
    assert breakdown.total == Decimal("20.31")


def test_tax_is_charged_after_tier_discount():
    breakdown = price_order(make((MUG, 5), tier="gold"))
    assert breakdown.discount == Decimal("6.00")
    assert breakdown.tax == Decimal("4.32")
    assert breakdown.total == Decimal("58.32")


def test_bulk_discount_for_large_lines():
    breakdown = price_order(make((TEA, 12)))
    assert breakdown.discount == Decimal("1.26")
    assert breakdown.total == Decimal("42.78")
