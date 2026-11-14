# Next-Item Inventory Forecaster

This 'Next-Item Inventory Forecaster' project helps store managers optimize inventory by predicting future purchases. It applies language model techniques, specifically continued-pretraining on mock sales transactions, to learn patterns in shopping baskets and suggest items likely to be bought together.

## What you are building

**Goal.** You will build a Next-Item Inventory Forecaster that predicts which items are likely to be bought next based on recent sales, aiding inventory reordering and stock optimization for store managers.

**Done when.** The project is complete when you have a Streamlit UI that accepts a partial basket and displays a list of recommended next items, demonstrating the model's ability to predict subsequent purchases.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Prepare mock sales transaction data as sequences of SKUs.**  
      Hint: Use text processing techniques to represent SKUs as tokens and create sequences.
- [x] **2. Continue pretrain a small language model on these SKU sequences.**  
      Hint: Apply continued-pretraining from the 'Teach a Language Model What Sells Together' drill.
- [ ] **3. Implement next-item prediction logic using the trained model.**  
      Hint: Implement token-by-token generation, similar to text completion, for next-item prediction.
- [ ] **4. Build a Streamlit UI to input a partial basket and display predictions.**  
      Hint: Utilize Streamlit widgets like `st.text_input` and `st.button` for the UI.
- [ ] **5. Evaluate model performance against a simple baseline.**  
      Hint: Compare your model's top-N accuracy against a simple frequency-based baseline.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Train a model

A trained model goes in `models/forecaster/` (git-ignored; set `MODEL_DIR` to use another folder). Either copy the
drill's checkpoint there, or continue pretraining SmolLM2-135M on the mock baskets yourself:

```bash
python scripts/prepare_data.py                                  # optional: writes data/train.txt and data/val.txt
python scripts/train.py --steps 100 --batch 16 --out models/forecaster
python scripts/train.py --steps 5 --out models/proof-5          # quick end-to-end check (~2 min on a laptop CPU)
```

The 5-step check was run on a Windows laptop CPU: held-out loss 3.509 -> 2.126, 2.0 min, saved and reloaded.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Add a feature to filter predicted next items by product category.
- Implement a 'clear basket' button in the Streamlit UI to easily start new predictions.
- Display a 'confidence score' or probability alongside each predicted item.
