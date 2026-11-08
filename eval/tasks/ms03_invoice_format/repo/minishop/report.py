"""Plain-text invoices."""
from __future__ import annotations

from minishop.models import Order
from minishop.money import format_money
from minishop.pricing import price_order


def render_invoice(order: Order, title: str = "INVOICE") -> str:
    """Render an invoice. Product names are wrapped in double quotes; the text ends with a newline."""
    breakdown = price_order(order)
    lines = [title, "=" * len(title)]
    for line in order.lines:
        lines.append(
            f"{line.quantity:>3} x '{line.product.name}' @ {format_money(line.product.unit_price)}"
            f" = {format_money(line.subtotal)}"
        )
    lines.append("-" * 40)
    lines.append(f"Subtotal: {format_money(breakdown.subtotal)}")
    lines.append(f"Discount: -{format_money(breakdown.discount)}")
    lines.append(f"Tax: {format_money(breakdown.tax)}")
    lines.append(f"Total: {format_money(breakdown.total)}")
    return "\n".join(lines)
