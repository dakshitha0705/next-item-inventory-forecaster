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
# A tiny randomly-initialised Llama (2 layers, hidden size 32, vocab 100) stands in for SmolLM2: same code path,
# no download, a fraction of a second per step.
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM  # noqa: E402

from app import train as training  # noqa: E402


def tiny_model(seed=0):
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                         bos_token_id=1, eos_token_id=2, pad_token_id=0)
    return LlamaForCausalLM(config).eval()


class CharTokenizer:
    """Enough of a tokenizer for `plan_batches`: one id per character, 2 is end-of-text."""
    eos_token_id = 2

    def encode(self, text):
        return [3 + ord(c) % 97 for c in text]


def test_padding_is_masked_out_of_the_loss():
    model = tiny_model()
    examples = [[5, 6, 7, 8, 9], [10, 11, 12]]
    ids, mask = training.pad(examples)
    assert ids.shape == (2, 16) and mask.sum() == 8 and ids[1, 3:].eq(0).all()
    wide_ids = torch.cat([ids, torch.zeros(2, 16, dtype=torch.long)], dim=1)   # 16 more padding columns
    wide_mask = torch.cat([mask, torch.zeros(2, 16)], dim=1)
    with torch.no_grad():
        assert torch.allclose(training.next_token_loss(model, ids, mask),
                              training.next_token_loss(model, wide_ids, wide_mask), atol=1e-6)


def test_loss_falls_when_training_on_one_tiny_batch():
    model = tiny_model()
    batch = [[1, 5, 6, 7, 8, 9, 2], [1, 10, 11, 12, 2], [1, 5, 6, 7, 2]]
    before = training.next_token_loss(model, *training.pad(batch)).item()
    history = training.train(model, [batch] * 15, peak_lr=1e-2, warmup=2, micro_batch=2, val_examples=batch,
                             eval_every=15, log=lambda m: None)
    after = training.next_token_loss(model, *training.pad(batch)).item()
    assert after < before * 0.8
    assert history[0]["step"] == 0 and history[-1]["step"] == 15 and history[-1]["val_loss"] < history[0]["val_loss"]


def test_micro_batches_give_the_same_gradient_as_one_batch():
    batch = [[1, 5, 6, 7, 8, 9, 2], [1, 10, 11, 12, 2], [1, 5, 2], [1, 7, 7, 7, 7, 7, 7, 7, 2]]
    grads = []
    for micro in (4, 1):
        model = tiny_model()
        real_total = training.pad(batch)[1][:, 1:].sum()
        for part in range(0, len(batch), micro):
            total, _ = training.summed_token_loss(model, *training.pad(batch[part:part + micro]))
            (total / real_total).backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


def test_plan_batches_and_schedule():
    baskets = [["Onion 1 kg", "Tomato 1 kg"], ["Milk 500 ml"], ["Rice 5 kg", "Salt 1 kg", "Dal 1 kg"]] * 20
    planned = training.plan_batches(CharTokenizer(), baskets, steps=5, batch_size=4, seed=0)
    assert len(planned) == 5 and all(len(b) == 4 for b in planned)
    assert all(e[-1] == 2 for b in planned for e in b)                       # every basket ends with end-of-text
    assert planned == training.plan_batches(CharTokenizer(), baskets, steps=5, batch_size=4, seed=0)
    rates = [training.learning_rate(s, 100, 1e-4, 50) for s in range(100)]
    assert rates[49] == pytest.approx(1e-4) and rates[0] < rates[49] and rates[-1] < 0.2e-4


def test_save_and_load_round_trip(tmp_path):
    model = tiny_model()
    training.save(model, None, tmp_path / "m", history=[{"step": 0}])
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path / "m").eval()
    ids = torch.tensor([[1, 5, 6, 7]])
    with torch.no_grad():
        assert torch.allclose(model(input_ids=ids).logits, loaded(input_ids=ids).logits, atol=1e-6)
    assert (tmp_path / "m" / "history.json").exists()


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
