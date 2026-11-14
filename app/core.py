"""The forecaster's logic, one section per milestone. The Streamlit page (`app/main.py`) and the scripts in
`scripts/` only call functions from here."""
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from thilimarketbasket.dataset import generate_dataset
from thilimarketbasket.generate import make_catalog
from thilimarketbasket.verbalize import verbalize_basket, verbalize_prefix

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
# Like text completion: the visible basket is the prompt, "Basket: a; b;", and the model writes the next item one
# token at a time until it ends the item with ";". Doing that `n_samples` times (sampling at temperature 1.0) and
# counting which items come out gives a ranked list, with the share of samples as a confidence score.


class Prediction(NamedTuple):
    item: str
    confidence: float   # model: share of sampled continuations; baseline: share of past baskets with the item
    source: str         # "model" or "baseline"


def model_dir() -> Path:
    return Path(os.environ.get("MODEL_DIR", "models/forecaster"))


def model_available(directory=None) -> bool:
    directory = Path(directory or model_dir())
    return (directory / "config.json").exists() and any(directory.glob("*.safetensors"))


@lru_cache(maxsize=2)
def load_model(directory: str | None = None):
    """(model, tokenizer) from a saved folder, in eval mode on the CPU. Cached: loading takes a few seconds."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    directory = str(directory or model_dir())
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()
    return model, tokenizer


def model_sampler(model, tokenizer, temperature: float = 1.0):
    """A function (prompt, n_samples, max_new_tokens, seed) -> list of continuations, one per sample.

    All samples are written together, one token per step, reusing the key/value cache. A sample stops at the first
    token containing ";" (end of an item) or at end-of-text; one that runs out of tokens comes back as "" because
    its last item may be cut off."""
    import torch
    eos = tokenizer.eos_token_id
    stops = torch.zeros(max(len(tokenizer), model.config.vocab_size), dtype=torch.bool)
    for token, i in tokenizer.get_vocab().items():
        if ";" in token:
            stops[i] = True

    @torch.no_grad()
    def sample(prompt: str, n_samples: int, max_new_tokens: int, seed: int) -> list[str]:
        generator = torch.Generator().manual_seed(seed)
        out = model(input_ids=torch.tensor([tokenizer.encode(prompt)] * n_samples), use_cache=True)
        written = torch.zeros(n_samples, 0, dtype=torch.long)
        done = torch.zeros(n_samples, dtype=torch.bool)
        for _ in range(max_new_tokens):
            probs = torch.softmax(out.logits[:, -1, :].float() / temperature, dim=-1)
            token = torch.multinomial(probs, 1, generator=generator)[:, 0]
            token = torch.where(done, eos, token)             # finished samples just write end-of-text
            written = torch.cat([written, token[:, None]], dim=1)
            done |= (token == eos) | stops[token]
            if done.all():
                break
            out = model(input_ids=token[:, None], past_key_values=out.past_key_values, use_cache=True)
        return [tokenizer.decode([t for t in row.tolist() if t != eos]) if finished else ""
                for row, finished in zip(written, done.tolist())]

    return sample


def parse_item(text: str, catalog) -> str | None:
    """The first item in a continuation such as " Onion 1 kg; Tomato" -> "Onion 1 kg", or None if it is not a
    catalogue name (the model made one up, or the text is empty)."""
    piece = text.split(";")[0].strip()
    return piece if piece in catalog else None


def frequency_baseline(train_baskets) -> list[tuple[str, float]]:
    """Every item that was ever sold, most frequent first, with the share of baskets that hold it. This baseline
    never reads the basket: it always suggests the shop's best sellers."""
    counts = Counter(item for basket in train_baskets for item in set(basket))
    n = max(1, len(train_baskets))
    return [(item, count / n) for item, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def baseline_predictions(basket, baseline, top_n: int = 10) -> list[Prediction]:
    """The top `top_n` best sellers not already in the basket."""
    held = set(basket)
    return [Prediction(item, share, "baseline") for item, share in baseline if item not in held][:top_n]


def predict_next(basket, model=None, tokenizer=None, catalog=None, top_n: int = 10, n_samples: int = 20,
                 max_new_tokens: int = 24, seed: int = 0, baseline=None, sampler=None) -> list[Prediction]:
    """The `top_n` items most likely to be bought next with `basket`, best first.

    The model completes "Basket: <basket in a seeded random order>;" `n_samples` times; each completion counts as
    one vote for the catalogue item it names (items already in the basket and made-up names get no vote). If fewer
    than `top_n` items get a vote, the rest are filled from `baseline` (see `frequency_baseline`), marked as such.
    `sampler` replaces model + tokenizer (tests use a fake one)."""
    catalog = set(catalog if catalog is not None else catalog_names())
    sampler = sampler or model_sampler(model, tokenizer)
    held = set(basket)
    prompt = verbalize_prefix(basket, seed=seed)
    votes = Counter()
    for text in sampler(prompt, n_samples, max_new_tokens, seed):
        item = parse_item(text, catalog)
        if item is not None and item not in held:
            votes[item] += 1
    popularity = dict(baseline or [])          # ties (common with few samples) go to the better seller
    ranked = sorted(votes, key=lambda item: (-votes[item], -popularity.get(item, 0.0), item))[:top_n]
    predictions = [Prediction(item, votes[item] / n_samples, "model") for item in ranked]
    if len(predictions) < top_n and baseline:
        chosen = held | {p.item for p in predictions}
        predictions += baseline_predictions(chosen, baseline, top_n - len(predictions))
    return predictions


# --- Milestone 4: Build a Streamlit UI to input a partial basket and display predictions.
# Hint: Utilize Streamlit widgets like `st.text_input` and `st.button` for the UI.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Evaluate model performance against a simple baseline.
# Hint: Compare your model's top-N accuracy against a simple frequency-based baseline.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
