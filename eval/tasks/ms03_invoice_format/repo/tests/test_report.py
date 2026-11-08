from decimal import Decimal

from minishop.models import Order, Product
from minishop.report import render_invoice

TEA = Product("TEA-01", "Green Tea", Decimal("3.50"), "food")
MUG = Product("MUG-02", "Travel Mug", Decimal("12.00"))


def test_invoice_layout():
    order = Order()
    order.add(TEA, 2)
    order.add(MUG, 1)
    expected = (
        "INVOICE\n"
        "=======\n"
        '  2 x "Green Tea" @ USD 3.50 = USD 7.00\n'
        '  1 x "Travel Mug" @ USD 12.00 = USD 12.00\n'
        "----------------------------------------\n"
        "Subtotal: USD 19.00\n"
        "Discount: -USD 0.00\n"
        "Tax: USD 1.31\n"
        "Total: USD 20.31\n"
    )
    assert render_invoice(order) == expected


def test_invoice_shows_bulk_discount_for_ten_units():
    order = Order()
    order.add(TEA, 10)
    text = render_invoice(order)
    assert "Discount: -USD 1.05" in text
    assert "Total: USD 35.65" in text
