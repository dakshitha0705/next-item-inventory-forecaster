"""Streamlit entry point: `streamlit run app/main.py`"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so `app.core` imports under `streamlit run`
from app import core  # noqa: E402

st.set_page_config(page_title="Next-Item Inventory Forecaster")
st.title("Next-Item Inventory Forecaster")
st.write("For store managers, predict which items are likely to be bought next based on recent sales, aiding "
         "inventory reordering and stock optimization.")


@st.cache_resource(show_spinner="Counting past sales for the best-seller baseline...")
def load_baseline():
    train, _ = core.split(core.make_baskets(int(os.environ.get("N_BASKETS", 100_000)), seed=0))
    return core.frequency_baseline(train)


@st.cache_resource(show_spinner="Loading the trained model...")
def load_sampler(directory: str):
    return core.model_sampler(*core.load_model(directory))


names = core.catalog_names()
categories = core.catalog_categories()
baseline = load_baseline()
sampler = load_sampler(str(core.model_dir())) if core.model_available() else None
if sampler is None:
    st.warning(f"No trained model found in `{core.model_dir()}`. Showing the frequency baseline (the shop's best "
               "sellers) instead. Copy a model there or run `python scripts/train.py` (see README).")

st.session_state.setdefault("basket", [])
search = st.text_input("Search the catalogue", placeholder="e.g. dal, milk, rice, shampoo")
matches = [n for n in names if search.strip().lower() in n.lower()]
options = [n for n in names if n in set(matches) | set(st.session_state.basket)]   # keep chosen items visible
st.multiselect(f"Partial basket ({len(matches)} catalogue items match the search)", options, key="basket",
               placeholder="Pick the items already in the basket")
top_n = st.slider("How many suggestions", min_value=3, max_value=20, value=10)

if st.button("Predict next items", type="primary", key="predict"):
    started = time.time()
    basket = st.session_state.basket
    if sampler is not None:
        predictions = core.predict_next(basket, catalog=names, top_n=top_n, baseline=baseline, sampler=sampler)
    else:
        predictions = core.baseline_predictions(basket, baseline, top_n)
    st.session_state.results = (predictions, time.time() - started)

if "results" in st.session_state:
    predictions, seconds = st.session_state.results
    table = pd.DataFrame({"rank": range(1, len(predictions) + 1),
                          "item": [p.item for p in predictions],
                          "category": [categories.get(p.item, "") for p in predictions],
                          "source": [p.source for p in predictions]})
    st.subheader("Likely next items")
    st.dataframe(table, hide_index=True)
    st.caption(f"{seconds:.1f} s. 'model': the language model wrote this item when completing the basket; "
               "'baseline': filled in from the shop's best sellers.")
