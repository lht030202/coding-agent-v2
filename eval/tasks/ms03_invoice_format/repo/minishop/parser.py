"""Parse plain-text orders such as:

    # weekly order
    TEA-01 x2
    COF-10 x 1
"""
from __future__ import annotations

import re

LINE_RE = re.compile(r"^(?P<sku>[A-Z0-9][A-Z0-9-]*)\s+x\s*(?P<qty>\d+)$", re.IGNORECASE)


def parse_order_text(text: str) -> list[tuple[str, int]]:
    """Return (SKU, quantity) pairs. SKUs are upper-cased; comments (#) and blank lines are ignored."""
    items: list[tuple[str, int]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = LINE_RE.match(line)
        if not match:
            raise ValueError(f"line {number}: cannot parse {raw.strip()!r}")
        quantity = int(match.group("qty"))
        if quantity <= 0:
            raise ValueError(f"line {number}: quantity must be positive")
        items.append((match.group("sku").upper(), quantity))
    return items
