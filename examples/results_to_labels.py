#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Turn a BirdNET benchmark job's output into a birdnet-schema label file.

The benchmark returns rows of ``file`` / ``start_s`` / ``end_s`` /
``species_name`` / ``confidence``, bucketed per Triton response id. This writes
them out as birdnet's own CSV schema, so the result drops straight into
`birdnet_benchmark.py`'s scoring (or any tool that reads a `predictions.csv`):

    file_path,start_time,end_time,species_name[,confidence]
    "rec_001.wav","00:00:00.00","00:00:03.00","Turdus merula_Common Blackbird"

By default `confidence` is omitted, giving the *ground-truth* shape used by
`annotations.csv`. Pass ``--with-confidence`` for the `predictions.csv` shape.

A raw run emits top-k species per segment, which makes noisy labels. Two filters
help turn predictions into something usable as ground truth:

    --min-confidence 0.5   keep only reasonably confident rows
    --top-1                keep only the best species per segment

Usage:
    python results_to_labels.py results.json -o annotations.csv --top-1 --min-confidence 0.5
    python birdnet_benchmark.py && python results_to_labels.py results.json
"""

import argparse
import csv
import json
import sys
from itertools import chain
from pathlib import Path, PurePosixPath

COLUMNS = ["file_path", "start_time", "end_time", "species_name"]


def hms_centis(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS.CC`` -- birdnet's `helper.hms_centis_fast`."""
    hours, remainder = divmod(float(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{secs:05.2f}"


def extract_rows(payload) -> list[dict]:
    """Pull prediction rows out of whatever shape the JSON happens to be.

    Accepts a full ``BenchmarkJob`` dump, a bare ``BenchmarkInferResult``, the
    ``results`` mapping on its own, or an already-flat list of rows -- so it
    works whether you saved the job, the inference results, or just the rows.
    """
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]

    if isinstance(payload, dict):
        for key in ("inference_results", "results"):
            if key in payload:
                return extract_rows(payload[key])
        # A bare {response_id: [rows]} mapping.
        buckets = [v for v in payload.values() if isinstance(v, list)]
        if buckets:
            return [r for r in chain.from_iterable(buckets) if isinstance(r, dict)]

    raise SystemExit(
        "Could not find prediction rows in that JSON. Expected a BenchmarkJob "
        "dump, its inference_results, a {response_id: [rows]} mapping, or a list "
        "of rows."
    )


def to_label_rows(
    rows: list[dict],
    *,
    min_confidence: float,
    top_1: bool,
    path_prefix: str | None,
) -> list[tuple]:
    required = {"file", "start_s", "end_s", "species_name", "confidence"}
    usable = [r for r in rows if required <= r.keys()]
    if not usable:
        raise SystemExit(
            f"No rows carried the expected keys {sorted(required)}. Is this a "
            "BirdNET job? DenseNet/YOLO results have a different shape."
        )

    usable = [r for r in usable if float(r["confidence"]) >= min_confidence]

    if top_1:
        best: dict[tuple[str, float], dict] = {}
        for r in usable:
            key = (PurePosixPath(str(r["file"])).name, round(float(r["start_s"]), 3))
            if key not in best or float(r["confidence"]) > float(
                best[key]["confidence"]
            ):
                best[key] = r
        usable = list(best.values())

    out = []
    for r in usable:
        name = PurePosixPath(str(r["file"])).name
        file_path = f"{path_prefix.rstrip('/')}/{name}" if path_prefix else name
        out.append(
            (
                file_path,
                hms_centis(r["start_s"]),
                hms_centis(r["end_s"]),
                str(r["species_name"]),
                float(r["confidence"]),
            )
        )
    # Stable, birdnet-like ordering: by file, then time, then descending confidence.
    out.sort(key=lambda t: (t[0], t[1], -t[4]))
    return out


def write_csv(rows: list[tuple], out_path: Path, with_confidence: bool) -> None:
    header = COLUMNS + (["confidence"] if with_confidence else [])
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        # QUOTE_NONNUMERIC quotes the four string columns and leaves confidence
        # bare -- exactly how birdnet writes predictions.csv.
        writer = csv.writer(fh, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(header)
        for row in rows:
            writer.writerow(list(row) if with_confidence else list(row[:4]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", type=Path, help="benchmark results JSON ('-' for stdin)")
    ap.add_argument("-o", "--output", type=Path, default=Path("annotations.csv"))
    ap.add_argument(
        "--with-confidence",
        action="store_true",
        help="include the confidence column (predictions.csv shape)",
    )
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="drop rows below this confidence (default: 0.0)",
    )
    ap.add_argument(
        "--top-1",
        action="store_true",
        help="keep only the highest-confidence species per segment",
    )
    ap.add_argument(
        "--path-prefix",
        default=None,
        help="prepend a directory to file_path (birdnet writes absolute "
        "paths; matching is on basename either way)",
    )
    args = ap.parse_args()

    payload = json.load(sys.stdin if str(args.results) == "-" else open(args.results))
    rows = to_label_rows(
        extract_rows(payload),
        min_confidence=args.min_confidence,
        top_1=args.top_1,
        path_prefix=args.path_prefix,
    )
    if not rows:
        raise SystemExit("Nothing left after filtering -- lower --min-confidence.")

    write_csv(rows, args.output, args.with_confidence)

    files = {r[0] for r in rows}
    segments = {(r[0], r[1]) for r in rows}
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"  files    : {len(files)}")
    print(f"  segments : {len(segments)}")
    print(f"  species  : {len({r[3] for r in rows})}")
    print(
        f"  columns  : {', '.join(COLUMNS + (['confidence'] if args.with_confidence else []))}"
    )


if __name__ == "__main__":
    main()
