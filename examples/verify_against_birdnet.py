#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prove the Triton-served model matches the reference `birdnet` implementation.

Runs the *same* audio twice -- once through `birdnet` locally (CPU, onnxruntime)
and once through the Edge-Farm benchmark on a device -- with every knob matched,
then diffs the per-segment species and confidences.

Method borrowed from the upstream `benchmarks/consistency_test.py`: to compare
two implementations you must switch *off* the filtering, or you are comparing
two top-5 lists rather than the model. Hence a large `top_k` and a zero
threshold on both sides.

What a pass looks like
----------------------
The upstream README states the model formats agree to ~2 decimal places, with
differences appearing from the third onward. So expect max |delta| < 0.01, and
identical species sets per segment. Larger drift means the preprocessing port
diverged (resampler or sigmoid); differing *sets* usually means a parameter is
not actually matched.

Note the two sides legitimately differ in execution provider -- the reference
runs onnxruntime on this machine's CPU, the benchmark runs Triton on the device
(CUDA/TensorRT). Small numeric differences are the expected outcome, not a bug.

Usage:
    conda activate birdnet
    cd examples
    python verify_against_birdnet.py path/to/clip.wav
"""

import csv
import os
import sys
import tempfile
from collections import defaultdict
from itertools import chain
from pathlib import Path

from dotenv import load_dotenv

from edge_benchmarking_client.client import EdgeBenchmarkingClient
from edge_benchmarking_types.edge_farm.models import TritonBirdNetClient

EDGE_DEVICE_HOST = "edge-03"
EXAMPLE_ROOT_DIR = "birdnet_onnx"

# Matched on BOTH sides. Filtering off so we compare the model, not a top-5.
PRECISION = "fp32"  # must match the model.onnx staged in birdnet_onnx/
TOP_K = 25
THRESHOLD = 0.0
SEGMENT_S = 3.0
OVERLAP_S = 0.0
SIGMOID_SENSITIVITY = 1.0
TOLERANCE = 0.01  # ~2 decimal places, per the upstream README


def parse_timestamp(value: str) -> float:
    value = value.strip().strip('"')
    if ":" in value:
        seconds = 0.0
        for part in value.split(":"):
            seconds = seconds * 60.0 + float(part)
        return seconds
    return float(value)


def reference_rows(wav: Path) -> dict:
    """Run birdnet locally and return {(start_s, species): confidence}."""
    import birdnet

    model = birdnet.load("acoustic", "3.0", "onnx", precision=PRECISION)
    result = model.predict(
        str(wav),
        top_k=TOP_K,
        default_confidence_threshold=THRESHOLD,
        overlap_duration_s=OVERLAP_S,
        segment_size_s=SEGMENT_S,
        apply_sigmoid=True,
        sigmoid_sensitivity=SIGMOID_SENSITIVITY,
        half_precision=False,
        custom_species_list=None,
        device="CPU",
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "reference.csv"
        result.to_csv(out, silent=True)
        rows = {}
        with open(out, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                key = (
                    round(parse_timestamp(r["start_time"]), 3),
                    r["species_name"].strip().strip('"'),
                )
                rows[key] = float(r["confidence"])
    return rows


def benchmark_rows(wav: Path) -> dict:
    """Run the same audio through Edge-Farm and return {(start_s, species): confidence}."""
    load_dotenv()
    client = EdgeBenchmarkingClient(
        protocol=os.getenv("EDGE_FARM_API_PROTOCOL"),
        host=os.getenv("EDGE_FARM_API_HOST"),
        username=os.getenv("EDGE_FARM_API_BASIC_AUTH_USERNAME"),
        password=os.getenv("EDGE_FARM_API_BASIC_AUTH_PASSWORD"),
    )
    model = client.find_model(root_dir=EXAMPLE_ROOT_DIR)
    labels = client.find_labels(root_dir=EXAMPLE_ROOT_DIR)

    job = client.benchmark(
        edge_device=EDGE_DEVICE_HOST,
        dataset=[wav],
        model=model,
        labels=labels,
        inference_client=TritonBirdNetClient(
            protocol="http",
            host=EDGE_DEVICE_HOST,
            port=8000,
            model_name=model.stem,
            model_version="1",
            segment_seconds=SEGMENT_S,
            overlap_seconds=OVERLAP_S,
            top_k=TOP_K,
            confidence_thres=THRESHOLD,
            sigmoid_sensitivity=SIGMOID_SENSITIVITY,
            batch_size=1,
            # off: the warm-up pass would otherwise double the first batch's work
            warm_up=False,
        ),
        chunk_size=1,
        cleanup=True,
    )
    rows = {}
    for r in chain.from_iterable(job.inference_results.results.values()):
        rows[(round(float(r["start_s"]), 3), r["species_name"])] = float(
            r["confidence"]
        )
    return rows


def _segment_cutoffs(rows: dict) -> dict:
    """Per-segment lowest reported confidence == that side's top-k cutoff."""
    cutoff = {}
    for (start_s, _), conf in rows.items():
        cutoff[start_s] = min(cutoff.get(start_s, conf), conf)
    return cutoff


def _explain_one_sided(
    only: set, source: dict, other_cutoff: dict
) -> tuple[list, list]:
    """Split rows present on one side only into boundary swaps vs real misses.

    A row is a *boundary swap* when its confidence sits at or below the other
    side's k-th-place cutoff (within tolerance): both models scored it, but it
    fell outside that side's top-k. With top_k filtering on, this is expected
    whenever the two sides differ at all -- it is not a disagreement about the
    species, only about rank ordering at the cutoff.
    """
    swaps, real = [], []
    for key in only:
        start_s, species = key
        cutoff = other_cutoff.get(start_s)
        conf = source[key]
        if cutoff is not None and conf <= cutoff + TOLERANCE:
            swaps.append((start_s, species, conf, cutoff))
        else:
            real.append((start_s, species, conf, cutoff))
    return sorted(swaps), sorted(real)


def compare(ref: dict, got: dict) -> bool:
    ref_segments = {s for s, _ in ref}
    got_segments = {s for s, _ in got}

    print(f"\n  segments  reference={len(ref_segments)}  benchmark={len(got_segments)}")
    if ref_segments != got_segments:
        missing, extra = sorted(ref_segments - got_segments), sorted(
            got_segments - ref_segments
        )
        print(f"  SEGMENT GRID MISMATCH -- missing {missing[:5]}, extra {extra[:5]}")
        print("  (check segment_seconds / overlap_seconds, and the end_s clamp)")
        return False

    both = ref.keys() & got.keys()
    only_ref, only_got = ref.keys() - got.keys(), got.keys() - ref.keys()
    deltas = {k: abs(ref[k] - got[k]) for k in both}
    worst = max(deltas, key=deltas.get) if deltas else None

    print(
        f"  rows      shared={len(both)}  only-reference={len(only_ref)}  only-benchmark={len(only_got)}"
    )

    over = {}
    if worst:
        print(
            f"  max |delta| confidence = {deltas[worst]:.6f}   at {worst[0]:.1f}s  {worst[1]}"
        )
        over = {k: v for k, v in deltas.items() if v > TOLERANCE}
        print(f"  rows over tolerance ({TOLERANCE}): {len(over)}")
        for k in sorted(over, key=over.get, reverse=True)[:5]:
            print(
                f"    {k[0]:7.1f}s  {k[1]:45s} ref={ref[k]:.4f} got={got[k]:.4f}  d={over[k]:.4f}"
            )

    # One-sided rows are expected at the top-k boundary; only unexplained ones matter.
    ref_swaps, ref_real = _explain_one_sided(only_ref, ref, _segment_cutoffs(got))
    got_swaps, got_real = _explain_one_sided(only_got, got, _segment_cutoffs(ref))

    if only_ref or only_got:
        print(
            f"\n  top-{TOP_K} boundary swaps (expected): "
            f"{len(ref_swaps)} reference-only, {len(got_swaps)} benchmark-only"
        )
        for start_s, species, conf, cutoff in (ref_swaps + got_swaps)[:5]:
            print(
                f"    {start_s:7.1f}s  {species:45s} conf={conf:.4f} vs cutoff={cutoff:.4f}"
            )
        unexplained = ref_real + got_real
        if unexplained:
            print(f"  UNEXPLAINED one-sided rows: {len(unexplained)}")
            for start_s, species, conf, cutoff in unexplained[:5]:
                c = f"{cutoff:.4f}" if cutoff is not None else "n/a"
                print(f"    {start_s:7.1f}s  {species:45s} conf={conf:.4f} cutoff={c}")

    ok = not over and not ref_real and not got_real
    if ok:
        print("\n  RESULT: MATCH -- the served model reproduces the reference")
        print(
            f"          all shared confidences within {TOLERANCE}; "
            f"membership differs only at the top-{TOP_K} cutoff"
        )
    else:
        print("\n  RESULT: MISMATCH -- see above")
    return ok


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python verify_against_birdnet.py <clip.wav>")
    wav = Path(sys.argv[1]).expanduser().resolve()
    if not wav.is_file():
        sys.exit(f"no such file: {wav}")

    print(
        f"Verifying '{wav.name}' -- birdnet(CPU, {PRECISION}) vs Edge-Farm({EDGE_DEVICE_HOST})"
    )
    print(
        f"matched: top_k={TOP_K} threshold={THRESHOLD} segment={SEGMENT_S}s overlap={OVERLAP_S}s"
    )

    print("\n[1/2] reference run (local birdnet) ...")
    ref = reference_rows(wav)
    print(f"      {len(ref)} rows")

    print("\n[2/2] benchmark run (Edge-Farm -> Triton) ...")
    got = benchmark_rows(wav)
    print(f"      {len(got)} rows")

    sys.exit(0 if compare(ref, got) else 1)
