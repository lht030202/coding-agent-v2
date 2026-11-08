"""Hidden tests: never shown to the agent, copied in only when a run is judged."""
from decimal import Decimal

import pytest

from minishop.catalog import by_category, price_range, search
from minishop.coupons import coupon_discount
from minishop.inventory import Inventory
from minishop.models import Order, Product
from minishop.money import percent_of, to_money
from minishop.parser import parse_order_text
from minishop.pricing import price_order
from minishop.report import render_invoice

TEA = Product("TEA-01", "Green Tea", Decimal("3.50"), "food")
MUG = Product("MUG-02", "Travel Mug", Decimal("12.00"))
BOOK = Product("BK-100", "Python Tricks", Decimal("20.00"), "books")
PEN = Product("PEN-03", "Gel Pen", Decimal("2.50"))


def make(*items, tier="standard", coupon=None):
    order = Order(customer_tier=tier, coupon=coupon)
    for product, quantity in items:
        order.add(product, quantity)
    return order


# money
def test_hidden_more_half_up_cases():
    assert to_money("0.125") == Decimal("0.13")
    assert percent_of(Decimal("10.05"), 50) == Decimal("5.03")


# pricing
def test_hidden_mixed_categories_gold():
    breakdown = price_order(make((TEA, 4), (BOOK, 1), (MUG, 1), tier="gold"))
    assert breakdown.discount == Decimal("4.60")
    assert breakdown.tax == Decimal("1.49")
    assert breakdown.total == Decimal("42.89")


def test_hidden_bulk_threshold_is_inclusive():
    breakdown = price_order(make((TEA, 10)))
    assert breakdown.discount == Decimal("1.05")
    assert breakdown.total == Decimal("35.65")
    assert price_order(make((TEA, 9))).discount == Decimal("0.00")


# coupons
def test_hidden_bogo_edge_cases():
    assert coupon_discount(make((TEA, 1), coupon="BOGO-TEA-01")) == Decimal("0.00")
    assert coupon_discount(make((MUG, 2), coupon="BOGO-TEA-01")) == Decimal("0.00")
    assert coupon_discount(make((TEA, 6), coupon=" bogo-tea-01 ")) == Decimal("10.50")


def test_hidden_save10_at_threshold():
    assert coupon_discount(make((PEN, 20), coupon="SAVE10")) == Decimal("5.00")


# parser
def test_hidden_parser_spacing_and_case():
    text = "COF-10 x 1\ntea-01 X3\n"
    assert parse_order_text(text) == [("COF-10", 1), ("TEA-01", 3)]
    with pytest.raises(ValueError, match="line 1"):
        parse_order_text("TEA-01 x0\n")


# inventory
def test_hidden_inventory_full_cycle():
    inventory = Inventory({"TEA-01": 10})
    inventory.reserve("TEA-01", 6)
    inventory.release("TEA-01", 1)
    inventory.fulfil("TEA-01", 3)
    assert inventory.reserved("TEA-01") == 2
    assert inventory.on_hand("TEA-01") == 7
    assert inventory.available("TEA-01") == 5
    with pytest.raises(ValueError):
        inventory.release("TEA-01", 3)
    assert inventory.available("UNKNOWN") == 0


# report
def test_hidden_invoice_quotes_and_newline():
    text = render_invoice(make((BOOK, 1)), title="RECEIPT")
    assert text.startswith("RECEIPT\n=======\n")
    assert '  1 x "Python Tricks" @ USD 20.00 = USD 20.00\n' in text
    assert text.endswith("Total: USD 20.00\n")


# catalog
def test_hidden_catalog_queries():
    cheapest_first = price_range(Decimal("0"), Decimal("1000"))
    prices = [product.unit_price for product in cheapest_first]
    assert prices == sorted(prices)
    assert all(product.category == "books" for product in by_category("books"))
    assert search("   ") == []
