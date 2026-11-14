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
CATALOG = {"Onion 1 kg", "Tomato 1 kg", "Potato 1 kg", "Milk 500 ml", "Bread", "Eggs 6"}
BASELINE = [("Milk 500 ml", 0.4), ("Onion 1 kg", 0.3), ("Bread", 0.2), ("Eggs 6", 0.1)]


def fake_sampler(outputs, calls=None):
    """Stands in for the model: returns the given continuations and records the prompt it was asked to complete."""
    def sample(prompt, n_samples, max_new_tokens, seed):
        if calls is not None:
            calls.append(prompt)
        return (outputs * n_samples)[:n_samples]
    return sample


def test_parse_item_takes_the_first_catalogue_name():
    assert core.parse_item(" Tomato 1 kg; Onion", CATALOG) == "Tomato 1 kg"
    assert core.parse_item(" Bread", CATALOG) == "Bread"                       # basket ended after the item
    assert core.parse_item(" Tomato 1 k", CATALOG) is None                     # cut off / made up
    assert core.parse_item("", CATALOG) is None
    assert core.parse_item(" Chocolate Cake;", CATALOG) is None


def test_predict_next_ranks_by_votes_and_skips_the_basket():
    calls = []
    outputs = [" Tomato 1 kg;", " Tomato 1 kg;", " Onion 1 kg;", " Potato 1 kg;", " Made Up Item;"]
    preds = core.predict_next(["Onion 1 kg", "Bread"], catalog=CATALOG, top_n=2, n_samples=10,
                              sampler=fake_sampler(outputs, calls))
    assert calls[0].startswith("Basket: ") and calls[0].endswith(";")
    assert "Onion 1 kg" in calls[0] and "Bread" in calls[0]
    assert [p.item for p in preds] == ["Tomato 1 kg", "Potato 1 kg"]     # Onion is in the basket already
    assert preds[0].confidence == pytest.approx(0.4) and preds[1].confidence == pytest.approx(0.2)
    assert all(p.source == "model" for p in preds)


def test_ties_go_to_the_better_seller():
    preds = core.predict_next([], catalog=CATALOG, top_n=3, n_samples=3, baseline=BASELINE,
                              sampler=fake_sampler([" Eggs 6;", " Bread;", " Milk 500 ml;"]))
    assert [p.item for p in preds] == ["Milk 500 ml", "Bread", "Eggs 6"]      # one vote each: baseline order


def test_confidences_are_sorted_and_sum_to_at_most_one():
    outputs = [" Tomato 1 kg;", " Eggs 6;", " Tomato 1 kg;", " Potato 1 kg;", "", " Eggs 6;", " Tomato 1 kg;"]
    preds = core.predict_next([], catalog=CATALOG, top_n=5, n_samples=7, sampler=fake_sampler(outputs))
    confidences = [p.confidence for p in preds]
    assert confidences == sorted(confidences, reverse=True)
    assert sum(confidences) <= 1 + 1e-9
    assert core.predict_next([], catalog=CATALOG, sampler=fake_sampler(outputs)) == \
        core.predict_next([], catalog=CATALOG, sampler=fake_sampler(outputs))


def test_short_lists_are_filled_from_the_baseline():
    preds = core.predict_next(["Onion 1 kg"], catalog=CATALOG, top_n=4, n_samples=5, baseline=BASELINE,
                              sampler=fake_sampler([" Bread;", " not an item;"]))
    assert [p.item for p in preds] == ["Bread", "Milk 500 ml", "Eggs 6"]   # baseline skips Onion and Bread
    assert [p.source for p in preds] == ["model", "baseline", "baseline"]
    assert len(core.predict_next([], catalog=CATALOG, top_n=3, sampler=fake_sampler([""]), baseline=BASELINE)) == 3


def test_real_sampler_writes_token_by_token_and_stops_at_a_semicolon():
    class Tok:                      # ids 0..99: 0 = end-of-text, 59 = ";", others = one letter each
        eos_token_id = 0

        def __len__(self):
            return 100

        def get_vocab(self):
            return {(";" if i == 59 else f"t{i}"): i for i in range(100)}

        def encode(self, text):
            return [3 + ord(c) % 90 for c in text]

        def decode(self, ids):
            return "".join(";" if i == 59 else chr(65 + i % 26) for i in ids)

    sample = core.model_sampler(tiny_model(), Tok())
    texts = sample("Basket: A;", n_samples=6, max_new_tokens=300, seed=1)
    assert len(texts) == 6 and texts == sample("Basket: A;", n_samples=6, max_new_tokens=300, seed=1)
    assert all(t.count(";") <= 1 and (t == "" or t.endswith(";") or ";" not in t) for t in texts)


# Milestone 4: Build a Streamlit UI to input a partial basket and display predictions.
@pytest.fixture
def app_without_model(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "no-model-here"))
    monkeypatch.setenv("N_BASKETS", str(N))
    at = AppTest.from_file("../app/main.py", default_timeout=60)   # relative to this file
    at.run()
    return at


def test_page_runs_without_a_model_and_says_so(app_without_model):
    at = app_without_model
    assert not at.exception
    assert "No trained model" in at.warning[0].value
    assert at.button(key="predict") and at.text_input and at.multiselect


def test_predict_button_shows_a_results_table(app_without_model):
    at = app_without_model
    basket = core.catalog_names()[:2]
    at.multiselect(key="basket").set_value(basket).run()
    at.slider[0].set_value(5).run()
    at.button(key="predict").click().run()
    assert not at.exception
    table = at.dataframe[0].value
    assert list(table.columns)[:3] == ["rank", "item", "category"] and len(table) == 5
    assert not set(table["item"]) & set(basket)                       # never suggests what is already there
    assert set(table["source"]) == {"baseline"}


def test_search_box_narrows_the_basket_choices(app_without_model):
    at = app_without_model
    at.text_input[0].set_value("toor dal").run()
    assert at.multiselect(key="basket").options and all("Toor Dal" in o for o in at.multiselect(key="basket").options)


# Milestone 5: Evaluate model performance against a simple baseline.
def test_top_n_accuracy_on_toy_inputs():
    predictions = [["a", "b", "c"], ["c", "a"], [], ["b", "a", "d"]]
    targets = ["a", "a", "a", "d"]
    assert core.hits(predictions, targets, 1) == [1, 0, 0, 0]
    assert core.top_n_accuracy(predictions, targets, 1) == 0.25
    assert core.top_n_accuracy(predictions, targets, 2) == 0.5
    assert core.top_n_accuracy(predictions, targets, 10) == 0.75
    with pytest.raises(ValueError):
        core.hits(predictions, targets[:2], 1)                      # one ranked list per target


def test_frequency_baseline_ranks_the_most_frequent_first():
    baskets = [["milk", "bread"], ["milk", "eggs"], ["milk", "milk"], ["eggs"], ["tea"]]
    baseline = core.frequency_baseline(baskets)
    assert [item for item, _ in baseline] == ["milk", "eggs", "bread", "tea"]    # ties alphabetical
    assert baseline[0][1] == pytest.approx(3 / 5)                                 # a repeat counts once per basket
    preds = core.baseline_predictions(["milk"], baseline, top_n=2)
    assert [p.item for p in preds] == ["eggs", "bread"] and {p.source for p in preds} == {"baseline"}


def test_bootstrap_range_brackets_the_difference():
    a, b = [1, 1, 0, 1, 1, 0, 1, 1] * 10, [0, 1, 0, 0, 1, 0, 0, 1] * 10
    mean, low, high = core.bootstrap_difference(a, b)
    assert mean == pytest.approx(0.375) and low < mean < high and low > 0
    assert core.bootstrap_difference(a, a) == (0.0, 0.0, 0.0)


# Stretch: filter predicted next items by product category.
def test_category_filter_keeps_only_allowed_items():
    allowed = {"Tomato 1 kg", "Potato 1 kg", "Onion 1 kg"}
    preds = core.predict_next([], catalog=CATALOG, top_n=3, n_samples=4, baseline=BASELINE, allowed=allowed,
                              sampler=fake_sampler([" Tomato 1 kg;", " Bread;", " Milk 500 ml;"]))
    assert [p.item for p in preds] == ["Tomato 1 kg", "Onion 1 kg"]     # Bread/Milk filtered; fill also filtered
    assert [p.source for p in preds] == ["model", "baseline"]
    pulses = core.items_in_categories(["Pulses"])
    assert pulses and all(core.catalog_categories()[i] == "Pulses" for i in pulses)
    assert core.items_in_categories([]) is None


def test_page_category_filter(app_without_model):
    at = app_without_model
    at.multiselect(key="categories").set_value(["Spices"]).run()
    at.button(key="predict").click().run()
    table = at.dataframe[0].value
    assert len(table) == 10 and set(table["category"]) == {"Spices"}


# Stretch: a "clear basket" button to start a new prediction.
def test_clear_basket_button_empties_basket_and_results(app_without_model):
    at = app_without_model
    at.multiselect(key="basket").set_value(core.catalog_names()[:3]).run()
    at.button(key="predict").click().run()
    assert at.dataframe
    at.button(key="clear").click().run()
    assert not at.exception
    assert at.multiselect(key="basket").value == [] and not at.dataframe


# Stretch: show a confidence score next to each predicted item.
def test_results_table_shows_a_confidence_per_item(app_without_model):
    at = app_without_model
    at.button(key="predict").click().run()
    table = at.dataframe[0].value
    assert "confidence" in table.columns
    assert table["confidence"].between(0, 1).all() and table["confidence"].is_monotonic_decreasing
    assert "Confidence" in at.caption[0].value
