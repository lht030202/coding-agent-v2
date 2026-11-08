"""Core data types: products, order lines and orders."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from minishop.money import to_money


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    unit_price: Decimal
    category: str = "general"


@dataclass
class OrderLine:
    product: Product
    quantity: int

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")

    @property
    def subtotal(self) -> Decimal:
        return to_money(self.product.unit_price * self.quantity)


@dataclass
class Order:
    lines: list[OrderLine] = field(default_factory=list)
    customer_tier: str = "standard"
    coupon: str | None = None

    def add(self, product: Product, quantity: int) -> None:
        """Add units of a product; repeated SKUs are merged into one line."""
        for line in self.lines:
            if line.product.sku == product.sku:
                line.quantity += quantity
                return
        self.lines.append(OrderLine(product, quantity))

    def items_subtotal(self) -> Decimal:
        return to_money(sum((line.subtotal for line in self.lines), Decimal("0")))

    def quantity_of(self, sku: str) -> int:
        return sum(line.quantity for line in self.lines if line.product.sku == sku)
