from decimal import Decimal

import pytest

from minishop.models import Order, OrderLine, Product

TEA = Product("TEA-01", "Green Tea", Decimal("3.50"), "food")
MUG = Product("MUG-02", "Travel Mug", Decimal("12.00"))


def test_line_subtotal():
    assert OrderLine(TEA, 3).subtotal == Decimal("10.50")


def test_add_merges_same_sku():
    order = Order()
    order.add(TEA, 2)
    order.add(TEA, 1)
    order.add(MUG, 1)
    assert len(order.lines) == 2
    assert order.quantity_of("TEA-01") == 3
    assert order.items_subtotal() == Decimal("22.50")


def test_quantity_must_be_positive():
    with pytest.raises(ValueError):
        OrderLine(TEA, 0)
