"""Chunked historical Statcast ingestion."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .data import download_statcast


def ingest(start: date, end: date, destination: Path, chunk_days: int = 7) -> Path:
    """Download Statcast in small windows and combine into one CSV."""
    if end < start:
        raise ValueError("end must be on or after start")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        chunk_path = destination.parent / f"_statcast_{cursor}_{chunk_end}.csv"
        download_statcast(cursor, chunk_end, chunk_path)
        frame = pd.read_csv(chunk_path)
        if not frame.empty:
            dates = pd.to_datetime(frame["game_date"], errors="raise").dt.date
            frame = frame.loc[(dates >= cursor) & (dates <= chunk_end)]
        if not frame.empty:
            frames.append(frame)
        chunk_path.unlink(missing_ok=True)
        cursor = chunk_end + timedelta(days=1)

    if not frames:
        raise RuntimeError("Statcast returned no rows for the requested period")
    pd.concat(frames, ignore_index=True).drop_duplicates().to_csv(destination, index=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download historical Statcast data in chunks")
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk-days", type=int, default=7)
    args = parser.parse_args()
    output = ingest(args.start, args.end, args.output, args.chunk_days)
    print(f"wrote={output}")


if __name__ == "__main__":
    main()
