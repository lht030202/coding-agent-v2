"""Stock levels and reservations."""
from __future__ import annotations


class InsufficientStock(Exception):
    pass


class Inventory:
    def __init__(self, stock: dict[str, int] | None = None):
        self._stock: dict[str, int] = dict(stock or {})
        self._reserved: dict[str, int] = {}

    def on_hand(self, sku: str) -> int:
        return self._stock.get(sku, 0)

    def reserved(self, sku: str) -> int:
        return self._reserved.get(sku, 0)

    def available(self, sku: str) -> int:
        """Units that can still be reserved: on hand minus already reserved."""
        return self.on_hand(sku) - self.reserved(sku)

    def reserve(self, sku: str, quantity: int) -> None:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if quantity > self.available(sku):
            raise InsufficientStock(f"{sku}: requested {quantity}, available {self.available(sku)}")
        self._reserved[sku] = self.reserved(sku) + quantity

    def release(self, sku: str, quantity: int) -> None:
        """Cancel part of a reservation."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if quantity > self.reserved(sku):
            raise ValueError(f"{sku}: cannot release {quantity}, only {self.reserved(sku)} reserved")
        self._reserved[sku] = self.reserved(sku) - quantity

    def fulfil(self, sku: str, quantity: int) -> None:
        """Ship reserved units: they leave both the reservation and the stock."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if quantity > self.reserved(sku):
            raise ValueError(f"{sku}: cannot fulfil {quantity}, only {self.reserved(sku)} reserved")
        self._reserved[sku] = self.reserved(sku) - quantity
        self._stock[sku] = self.on_hand(sku) - quantity
