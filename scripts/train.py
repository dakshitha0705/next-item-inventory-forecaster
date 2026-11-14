"""Continue pretraining a small language model on the mock baskets (milestone 2).

    python scripts/train.py --steps 100 --batch 16 --out models/forecaster
    python scripts/train.py --steps 5 --out models/proof-5          # a quick end-to-end check

Starts from SmolLM2-135M (downloaded from Hugging Face once) unless `--base` names another model or folder.
On a laptop CPU one step of 16 baskets takes about 24 s, so 100 steps is ~40 min.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core import make_baskets, split  # noqa: E402
from app.train import BASE_MODEL, continue_pretraining  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=4,
                        help="baskets per forward pass; the gradient is the same, only memory changes")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--val-baskets", type=int, default=300)
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--out", default="models/forecaster")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    train, val = split(make_baskets(100_000, seed=0))
    started = time.time()
    history = continue_pretraining(train, args.out, base=args.base, val_baskets=val[:args.val_baskets],
                                   steps=args.steps, batch_size=args.batch, peak_lr=args.lr, warmup=args.warmup,
                                   micro_batch=args.micro_batch, eval_every=args.eval_every,
                                   log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
    print(f"saved to {args.out} after {(time.time() - started) / 60:.1f} min; history {history}")


if __name__ == "__main__":
    main()
