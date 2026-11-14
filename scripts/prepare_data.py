"""Write the mock sales data as text: one basket per line, "Basket: item; item; item".

    python scripts/prepare_data.py [--baskets 100000] [--seed 0] [--out data]

Writes data/train.txt and data/val.txt (git-ignored), split exactly as in the drill (5% held out).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core import make_baskets, split, verbalize_all  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baskets", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    train, val = split(make_baskets(args.baskets, args.seed))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, baskets, seed in (("train", train, 1), ("val", val, 2)):
        path = out / f"{name}.txt"
        path.write_text("\n".join(verbalize_all(baskets, seed=seed)) + "\n", encoding="utf-8")
        print(f"{path}: {len(baskets):,} baskets")


if __name__ == "__main__":
    main()
