"""Tests for app/core.py, one group per milestone. They run on a CPU with no trained model and no downloads."""
import pytest

from app import core
from thilimarketbasket.verbalize import parse_basket

N = 10_000  # the smallest dataset the generator makes (about 4 s, cached across tests)


# Milestone 1: Prepare mock sales transaction data as sequences of SKUs.
@pytest.fixture(scope="module")
def baskets():
    return core.make_baskets(N, seed=0)


def test_baskets_are_catalogue_skus_without_repeats(baskets):
    names = set(core.catalog_names())
    assert len(names) == 1000 and len(baskets) == N
    for basket in baskets[:500]:
        assert basket and len(basket) == len(set(basket)) and set(basket) <= names


def test_verbalize_writes_each_item_once_in_a_seeded_order(baskets):
    basket = baskets[1]
    text = core.verbalize(basket, seed=3)
    assert text.startswith("Basket: ") and "; " in text
    items = parse_basket(text)
    assert sorted(items) == sorted(basket)                          # every item once, nothing else
    assert core.verbalize(basket, seed=3) == text                     # same seed -> same text
    texts = {core.verbalize(basket, seed=s) for s in range(10)}
    assert len(texts) > 1                                             # other seeds reshuffle
    assert core.verbalize(["B", "A", "A"], seed=0) in ("Basket: A; B", "Basket: B; A")


def test_split_is_deterministic_and_about_five_percent(baskets):
    train, val = core.split(baskets, val_share=0.05, seed=0)
    assert len(train) + len(val) == N
    assert 0.04 * N < len(val) < 0.06 * N
    train2, val2 = core.split(baskets, val_share=0.05, seed=0)
    assert (len(train2), len(val2)) == (len(train), len(val)) and val2 == val


def test_categories_cover_the_catalogue():
    categories = core.catalog_categories()
    assert set(categories) == set(core.catalog_names())
    assert "Pulses" in set(categories.values())


# Milestone 2: Continue pretrain a small language model on these SKU sequences.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Implement next-item prediction logic using the trained model.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Build a Streamlit UI to input a partial basket and display predictions.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Evaluate model performance against a simple baseline.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")
