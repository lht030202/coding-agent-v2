from decimal import Decimal

import pytest

from minishop.catalog import find_by_sku, price_range, search


def test_find_by_sku_is_case_insensitive():
    assert find_by_sku(" tea-01 ").name == "Green Tea"
    with pytest.raises(KeyError):
        find_by_sku("NOPE-1")


def test_search_matches_names():
    names = [product.name for product in search("tea")]
    assert "Green Tea" in names
    assert names == sorted(names)


def test_price_range_includes_both_bounds():
    skus = [product.sku for product in price_range(Decimal("3.50"), Decimal("3.50"))]
    assert "TEA-01" in skus
