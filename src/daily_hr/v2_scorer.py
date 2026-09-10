"""V2 uses exactly the V1 scoring mechanics with experimental weights only."""
from pathlib import Path

import pandas as pd

from .production_scorer import score_board

WEIGHTS_V2 = {
    "baseline": 0.25,
    "pitcher_vulnerability": 0.275,
    "pitch_matchup": 0.175,
    "power_upside": 0.20,
    "park": 0.025,
    "weather": 0.05,
    "lineup": 0.025,
}


def score_frame(df: pd.DataFrame) -> pd.DataFrame:
    source = df.copy()
    if "model_rank" in source:
        source["input_model_rank"] = source["model_rank"]
    out = score_board(source, weights=WEIGHTS_V2)
    out = out.rename(columns={"production_score": "v2_score"})
    out["v2_rank"] = out["model_rank"]
    return out


def rerank_csv(input_path: str | Path, output_path: str | Path) -> None:
    scored = score_frame(pd.read_csv(input_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output, index=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Apply V2 HR ranking to a board CSV")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    rerank_csv(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
