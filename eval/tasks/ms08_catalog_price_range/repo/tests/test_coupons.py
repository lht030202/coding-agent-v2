from decimal import Decimal

from minishop.coupons import coupon_discount
from minishop.models import Order, Product
from minishop.pricing import price_order

TEA = Product("TEA-01", "Green Tea", Decimal("3.50"), "food")
MUG = Product("MUG-02", "Travel Mug", Decimal("12.00"))


def order(*items, coupon=None):
    result = Order(coupon=coupon)
    for product, quantity in items:
        result.add(product, quantity)
    return result


def test_save10_needs_minimum_spend():
    assert coupon_discount(order((MUG, 4), coupon="save10")) == Decimal("0.00")  # 48.00 < 50.00
    assert coupon_discount(order((MUG, 5), coupon="SAVE10")) == Decimal("6.00")  # 60.00


def test_bogo_makes_every_second_unit_free():
    assert coupon_discount(order((TEA, 5), coupon="BOGO-TEA-01")) == Decimal("7.00")  # 2 free units


def test_bogo_applies_before_tax():
    breakdown = price_order(order((TEA, 4), coupon="bogo-tea-01"))
    assert breakdown.discount == Decimal("7.00")
    assert breakdown.total == Decimal("7.35")


def test_unknown_coupon_gives_nothing():
    assert coupon_discount(order((MUG, 1), coupon="NOPE")) == Decimal("0.00")
