"""The forecaster's logic, one section per milestone. The Streamlit page (`app/main.py`) and the scripts in
`scripts/` only call functions from here."""
from functools import lru_cache

import numpy as np
from thilimarketbasket.dataset import generate_dataset
from thilimarketbasket.generate import make_catalog
from thilimarketbasket.verbalize import verbalize_basket

# --- Milestone 1: Prepare mock sales transaction data as sequences of SKUs.
# The shop and its baskets come from the drill's generator (same seed = same shop, same planted itemsets). A basket
# becomes one line of text, "Basket: item; item; item", each SKU written once in a shuffled order, so the model
# never learns an order that means nothing.


@lru_cache(maxsize=4)
def _dataset(n_baskets: int, seed: int):
    return generate_dataset(n_baskets=n_baskets, seed=seed)


def make_baskets(n_baskets: int = 100_000, seed: int = 0) -> list[list[str]]:
    """Mock till transactions: `n_baskets` baskets of SKU names (at least 10,000; the generator needs that many to
    plant its rarest itemset). Repeats of an SKU (quantity > 1) are kept once, in first-scanned order."""
    return [list(dict.fromkeys(basket)) for basket in _dataset(n_baskets, seed).baskets]


@lru_cache(maxsize=1)
def _catalog():
    return make_catalog(seed=0)


def catalog_names() -> list[str]:
    """The 1,000 SKU names of the shop, in catalogue order."""
    return [item.name for item in _catalog()]


def catalog_categories() -> dict[str, str]:
    """SKU name -> category, e.g. "Surya Grains Toor Dal 1 kg" -> "Pulses"."""
    return {item.name: item.category for item in _catalog()}


def verbalize(basket, seed: int = 0) -> str:
    """One basket as text, e.g. "Basket: Onion 1 kg; Surya Grains Toor Dal 1 kg". Each item once, in a random order
    that is the same for the same seed."""
    return verbalize_basket(basket, seed=seed)


def split(baskets, val_share: float = 0.05, seed: int = 0):
    """(train, val): the drill's split, where each basket is held out with probability `val_share`."""
    is_val = np.random.default_rng(seed).random(len(baskets)) < val_share
    train = [b for b, v in zip(baskets, is_val) if not v]
    val = [b for b, v in zip(baskets, is_val) if v]
    return train, val


def verbalize_all(baskets, seed: int = 0) -> list[str]:
    """Every basket as one line of text, each with its own order seed drawn from `seed`."""
    seeds = np.random.default_rng(seed).integers(2**31, size=len(baskets))
    return [verbalize(basket, seed=int(s)) for basket, s in zip(baskets, seeds)]


# --- Milestone 2: Continue pretrain a small language model on these SKU sequences.
# The training loop is in `app/train.py` (ported from the drill) and runs with `scripts/train.py`. The trained
# model lives in MODEL_DIR (default models/forecaster), which is git-ignored.


# --- Milestone 3: Implement next-item prediction logic using the trained model.
# Hint: Implement token-by-token generation, similar to text completion, for next-item prediction.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Build a Streamlit UI to input a partial basket and display predictions.
# Hint: Utilize Streamlit widgets like `st.text_input` and `st.button` for the UI.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Evaluate model performance against a simple baseline.
# Hint: Compare your model's top-N accuracy against a simple frequency-based baseline.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
