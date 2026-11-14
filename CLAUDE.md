# CLAUDE.md — Project: Next-Item Inventory Forecaster (askThili)

I'm Dakshitha. This is my askThili project, built on the drill "Teach a Language Model What Sells Together"
(my drill repo: `E:\projects\learnfast-work`). This repo is what I submit: askThili reads the repo, the README and the
**GitHub Actions test results**. Commit history is reviewed, so: one commit per milestone.

Repo: `E:\projects\next-item-inventory-forecaster` → `https://github.com/dakshitha0705/next-item-inventory-forecaster`

## Goal and "done when" (from the brief)
Predict which items are likely to be bought next from a partial basket, for store managers' reordering.
Done when a Streamlit UI accepts a partial basket and shows a list of recommended next items.

## Milestones (one stub each in app/core.py, one test each in tests/test_core.py)
1. Prepare mock sales transaction data as sequences of SKUs.
2. Continue pretraining a small language model on these SKU sequences.
3. Next-item prediction with the trained model (token-by-token generation, like text completion).
4. Streamlit UI: input a partial basket, show predictions.
5. Evaluate top-N accuracy against a simple frequency baseline.
Stretch (do after 5, all small): filter predictions by category · "clear basket" button · confidence score per item.

## Rules
- Keep the starter plumbing working: don't break `app/llm.py`, `app/storage.py`, `tests/test_smoke.py`, `.github/workflows/ci.yml`.
  (The LLM chat helper isn't needed for this project; leave it, don't require a `.env`.)
- Replace the `milestone_N` stubs with real, well-named functions (keep a `milestone_N` alias if it helps clarity, or delete the
  stubs when replaced). Keep code small and readable; this is a student project, not a framework.
- For every milestone: remove the `skip` on its test, write a real test, make it pass.
- **Tests must pass on GitHub Actions (ubuntu, Python 3.12, CPU, no trained model, no internet beyond pip).**
  So tests must NOT download anything from Hugging Face or need `models/`. Use tiny inputs, a tiny
  randomly-initialised model (e.g. `transformers.LlamaConfig` with 2 layers, hidden size 32, vocab ~100, fed made-up
  token ids) or fake/injected generator functions. Keep the whole test suite under ~1 minute.
- Never commit: `.env`, `data/`, `models/`, checkpoints, `*.safetensors`, `.venv/`. Add `models/` and `results/raw/`
  to `.gitignore`. Small result files (e.g. `results/eval.md`, `results/eval.json`) ARE committed.
- Dependencies: add to `requirements.txt` with the CPU PyTorch index so CI doesn't pull 2 GB CUDA wheels:
  ```
  --extra-index-url https://download.pytorch.org/whl/cpu
  torch
  transformers
  ```
- After each milestone: tick its box in README.md, then
  `git add -A && git commit -m "Milestone N: <what was done>" && git push`
  (askThili's own format). Then run `git status` and confirm `.env`, `data/`, `models/` are not listed.
- After each push, if the `gh` CLI is available, check the Actions run (`gh run list --limit 1`); fix any red build
  before moving on. If `gh` isn't available, tell me to check the Actions tab.
- After each milestone print:
  ```
  ✅ MILESTONE N COMPLETE — <title>
  Tick it on askThili.
  What I built: <3–5 lines>
  Tests: <what they check, pass count>
  Commit: <hash>
  ```
  then go straight on to the next milestone (don't wait), unless something is truly blocked.

## Environment (Windows laptop, CPU only, no GPU)
- Shell is Git Bash: use `export MSYS_NO_PATHCONV=1` before any `docker exec` with Linux paths.
- Python: use a local venv with Python 3.12: `py -3.12 -m venv .venv` then `.venv/Scripts/python -m pip install -r requirements.txt`.
  If Python 3.12 is missing, STOP and tell me to run `winget install Python.Python.3.12`.
- Reuse the drill's work, don't redo it:
  - Data: `thilimarketbasket.dataset.generate_dataset(n_baskets=100_000, seed=0)` (same shop, planted itemsets) and
    the drill's verbalize format `Basket: item; item; item` (each item once, reshuffled per epoch/seed).
  - Training code: port the PyTorch training from `E:\projects\learnfast-work\work\mybasket\` (masked next-token loss,
    warmup + cosine schedule, half-batch gradient accumulation — batch 16 ran out of memory in one go).
  - Trained model: `E:\projects\learnfast-work\work\checkpoints\baskets-100x16` (SmolLM2-135M, 100 steps). Copy it to
    `models/forecaster/` and check it loads with `AutoModelForCausalLM.from_pretrained`. Use it by default.
    Only train more if I ask (it's ~24 s/step on this CPU). The training script must still work end to end:
    prove it with a short run (e.g. 5 steps) and record the command in the README.
- Model path for the app/eval: env var `MODEL_DIR`, default `models/forecaster`.

## Milestone notes
**M1 — data.** `app/core.py`: `make_baskets(n_baskets, seed)`, `verbalize(basket, seed)` → `"Basket: a; b; c"`
(each item once, shuffled, stable per seed), `split(baskets, val_share=0.05, seed=0)` (same split as the drill),
`catalog_names()`. A script `scripts/prepare_data.py` writes `data/train.txt` / `data/val.txt` (git-ignored).
Tests: format, no repeated items, only catalogue names, same seed → same text, split sizes deterministic.
(Use a small `n_baskets` in tests.)

**M2 — continued pretraining.** `app/train.py` (or in core): masked next-token loss + training loop + save.
`scripts/train.py --steps N --batch 16 --out models/forecaster`. Tests (tiny random model): padding is masked out
(loss unchanged by extra padding), loss falls after a few steps on one tiny batch, save/load round-trips.

**M3 — prediction.** `predict_next(basket, model, tokenizer, catalog, top_n=10, n_samples=20, max_new_tokens=24, seed=0)`:
prompt `"Basket: <visible items in a seeded random order>;"`, sample `n_samples` continuations token by token
(temperature 1.0, batched, stop at ";"), parse each into the first catalogue item, drop items already in the basket and
non-catalogue text, rank by how often each item was generated → `[(item, confidence), ...]` where confidence = share of
samples. If fewer than `top_n` distinct items come back, fill the rest from the frequency baseline (mark them as such).
Keep a helper `parse_item(text, catalog)` separate so it's testable. Tests: parsing, exclusion of basket items,
confidences sum ≤ 1 and are sorted, fallback fill — using an injected fake generator, no real model.
Make it fast enough for the UI on CPU (target a few seconds per prediction); cache model loading.

**M4 — Streamlit UI** (`app/main.py`). Inputs: `st.multiselect` over catalogue names for the basket (plus the brief's
`st.text_input` as a quick search/filter), a "Predict next items" `st.button`, a "Clear basket" button, a top-N slider,
a category filter (stretch), and a results table: rank, item, category, confidence (stretch) and source (model /
baseline). Load the model with `st.cache_resource`; if `models/forecaster` is missing, show a clear message and use the
frequency baseline so the app still runs. Test with `streamlit.testing.v1.AppTest` in a no-model setup: the page
runs without exceptions and pressing the button shows a results table.

**M5 — evaluation.** `top_n_accuracy(predictions, targets, n)`, `frequency_baseline(train_baskets)`,
`scripts/evaluate.py`: the drill's held-out cases (`thilimarketbasket.evaluate.make_cases(val, names, n_cases, seed=0)`,
hide one item), report hits@1/5/10 for model vs frequency baseline (and item-to-item counting if quick), with a
bootstrap 95% range on the difference. Choose the number of cases so it finishes in ~15 minutes on this CPU (time 5
cases first); write `results/eval.md` + `results/eval.json` (committed) and a short results section in the README,
honest about the 100-step model and the case count. Tests: top-N accuracy on toy inputs, baseline ranks most frequent
first.

## README (final)
Rewrite it for someone else to run it: what it does, a screenshot placeholder, setup on Windows and Linux/macOS,
how to get a model (copy from the drill or `scripts/train.py`), `streamlit run app/main.py`, `pytest`, the evaluation
table, honest limits (synthetic shop, short CPU training, generation-based ranking), and the milestone checklist ticked.

## Finish
After milestone 5 and the stretch ideas: print a table of all milestones with commit hashes, confirm `git status` is
clean, Actions is green (or tell me to check), and that no secrets/models/data are tracked (`git ls-files`).
