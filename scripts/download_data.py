"""Download ScholarCatalyst from the Hugging Face Hub into data/.

The dataset is CC BY-NC 4.0, so it is fetched locally and never committed.
Usage: uv run python scripts/download_data.py
"""

from pathlib import Path

from datasets import load_dataset

REPO = "ScholarCatalyst/ScholarCatalyst"
SUBSETS = ["corpus", "queries", "core_query", "subfield_query"]
OUT = Path(__file__).resolve().parents[1] / "data"


def main():
    OUT.mkdir(exist_ok=True)
    for name in SUBSETS:
        ds = load_dataset(REPO, name, split="test")
        path = OUT / f"{name}.parquet"
        ds.to_parquet(path)
        print(f"{name}: {len(ds):,} rows -> {path}")


if __name__ == "__main__":
    main()
