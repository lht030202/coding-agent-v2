import pytest

from minishop.inventory import InsufficientStock, Inventory


def test_reserve_reduces_available():
    inventory = Inventory({"TEA-01": 5})
    inventory.reserve("TEA-01", 3)
    assert inventory.available("TEA-01") == 2


def test_cannot_reserve_more_than_available():
    inventory = Inventory({"TEA-01": 5})
    inventory.reserve("TEA-01", 4)
    with pytest.raises(InsufficientStock):
        inventory.reserve("TEA-01", 2)


def test_release_returns_units():
    inventory = Inventory({"TEA-01": 5})
    inventory.reserve("TEA-01", 3)
    inventory.release("TEA-01", 2)
    assert inventory.reserved("TEA-01") == 1
    assert inventory.available("TEA-01") == 4
