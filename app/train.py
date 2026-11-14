"""Milestone 2: continued pretraining of a small causal language model on basket text, with PyTorch on a CPU.

Ported from the drill (`learnfast-work/work/mybasket/pretrain.py`), which ported the course's MLX version:
- objective: plain next-token cross-entropy on "Basket: a; b; c<eos>", padding masked out;
- data: length-bucketed batches, every basket in a fresh random order, planned up front from the seed;
- optimiser: AdamW (weight decay 0.01), gradient clipping at 1.0, linear warmup then cosine down to a tenth of the peak;
- memory: each batch runs as micro-batches with gradient accumulation (a batch of 16 in one go ran out of memory),
  and the accumulated gradient equals the one-batch gradient exactly.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from thilimarketbasket.pretrain import _encode   # "Basket: ..." text -> token ids + end-of-text (no MLX needed)

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"


def pad(examples: list[list[int]]):
    """Token ids right-padded with 0 to a multiple of 16, and a mask that is 1 on real tokens, 0 on padding."""
    width = -(-max(len(e) for e in examples) // 16) * 16
    ids = torch.zeros(len(examples), width, dtype=torch.long)
    mask = torch.zeros(len(examples), width)
    for row, e in enumerate(examples):
        ids[row, :len(e)] = torch.tensor(e)
        mask[row, :len(e)] = 1
    return ids, mask


def summed_token_loss(model, ids, mask):
    """(sum of next-token cross-entropy over real tokens, number of real target tokens)."""
    logits = model(input_ids=ids[:, :-1]).logits.float()
    per_token = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                                                  reduction="none")   # (rows * tokens), no transposed copy
    real = mask[:, 1:].reshape(-1)
    return (per_token * real).sum(), real.sum()


def next_token_loss(model, ids, mask):
    """Average cross-entropy of predicting each token from the ones before it, ignoring padding."""
    total, count = summed_token_loss(model, ids, mask)
    return total / count


@torch.no_grad()
def validation_loss(model, examples: list[list[int]], batch_size: int = 8) -> float:
    """Mean next-token loss over held-out examples (batch size changes memory use, not the result)."""
    model.eval()
    total = count = 0.0
    for start in range(0, len(examples), batch_size):
        t, c = summed_token_loss(model, *pad(examples[start:start + batch_size]))
        total, count = total + float(t), count + float(c)
    return total / count


def plan_batches(tokenizer, baskets, steps: int, batch_size: int, seed: int = 0) -> list[list[list[int]]]:
    """The token ids of every training batch: draw 20 batches' worth of baskets, write each in a random order,
    group by length (less padding), and serve the groups in random order. Same seed -> same batches."""
    rng = np.random.default_rng(seed)
    bucket = batch_size * 20
    planned = []
    while len(planned) < steps:
        order = rng.permutation(len(baskets))
        for start in range(0, max(1, len(order) - bucket + 1), bucket):
            chosen = order[start:start + bucket]
            examples = _encode(tokenizer, [baskets[i] for i in chosen], rng.integers(2**31, size=len(chosen)))
            by_length = sorted(range(len(examples)), key=lambda i: len(examples[i]))
            groups = [by_length[i:i + batch_size] for i in range(0, len(by_length), batch_size)]
            for g in rng.permutation(len(groups)):
                planned.append([examples[i] for i in groups[g]])
                if len(planned) == steps:
                    return planned
    return planned


def learning_rate(step: int, steps: int, peak: float, warmup: int) -> float:
    """Linear warmup to `peak`, then a cosine down to a tenth of it. `step` counts from 0."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train(model, batches, peak_lr: float = 1e-4, warmup: int = 50, micro_batch: int = 8,
          val_examples=None, eval_every: int = 50, log=print) -> list[dict]:
    """One optimiser step per planned batch. Returns the history: train loss (and held-out loss) per eval point."""
    steps = len(batches)
    warmup = min(warmup, max(1, steps // 2))     # a short run still gets a schedule, not just warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    history, recent, tokens, started = [], [], 0, time.time()
    if val_examples:
        history.append({"step": 0, "loss": None, "val_loss": validation_loss(model, val_examples), "seconds": 0.0})
        log(f"step 0  val loss {history[-1]['val_loss']:.3f}")
    for step, batch in enumerate(batches, start=1):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step - 1, steps, peak_lr, warmup)
        # Each micro-batch's summed loss is divided by the WHOLE batch's real-token count, so the accumulated
        # gradient is exactly the gradient of the batch's mean loss.
        real_total = pad(batch)[1][:, 1:].sum()
        loss_value = 0.0
        for part in range(0, len(batch), micro_batch):
            total, _ = summed_token_loss(model, *pad(batch[part:part + micro_batch]))
            part_loss = total / real_total
            part_loss.backward()
            loss_value += float(part_loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens += sum(len(e) for e in batch)
        recent.append(loss_value)
        log(f"  step {step}/{steps}  loss {loss_value:.3f}  {time.time() - started:.0f} s")
        if step % eval_every == 0 or step == steps:
            val = validation_loss(model, val_examples) if val_examples else None
            history.append({"step": step, "loss": float(np.mean(recent)), "val_loss": val, "tokens": tokens,
                            "seconds": time.time() - started})
            recent = []
            if val is not None:
                log(f"step {step}  val loss {val:.3f}")
    return history


def save(model, tokenizer, out_dir, history=None) -> Path:
    """Save so that `AutoModelForCausalLM.from_pretrained(out_dir)` (and the tokenizer) load it back."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    if tokenizer is not None:
        tokenizer.save_pretrained(out)
    if history is not None:
        (out / "history.json").write_text(json.dumps(history, indent=1))
    return out


def continue_pretraining(train_baskets, out_dir, base: str = BASE_MODEL, val_baskets=None, steps: int = 100,
                         batch_size: int = 16, peak_lr: float = 1e-4, warmup: int = 50, micro_batch: int = 8,
                         eval_every: int = 50, seed: int = 0, log=print) -> list[dict]:
    """Load `base` (a Hugging Face name or a local folder), train it on the baskets, save it to `out_dir`."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    val_examples = None
    if val_baskets:
        val_examples = _encode(tokenizer, val_baskets, np.random.default_rng(123).integers(2**31, size=len(val_baskets)))
    batches = plan_batches(tokenizer, train_baskets, steps, batch_size, seed)
    history = train(model, batches, peak_lr, warmup, micro_batch, val_examples, eval_every, log)
    save(model, tokenizer, out_dir, history)
    return history
