#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark BirdNET v3.0 on an edge device, and score it locally.

A reference to copy, and the repo's smoke test for the BirdNET path -- mirrors
``main.py`` (the DenseNet workflow) so the differences stand out.

What is different from an image benchmark
-----------------------------------------
* The dataset is audio; ``AUDIO_EXTENSIONS`` is a convenience set, not a
  restriction.
* A **species label file is mandatory**. Unlike DenseNet -- where Triton
  resolves labels server-side via ``class_count`` -- BirdNET applies its own
  top-k after the flat sigmoid, so the client maps indices to names itself. The
  names must match the ground truth byte-for-byte.
* **Scoring happens here, on the client.** The Edge-Farm API computes nothing for
  BirdNET: the job hands back the parsed model output in
  ``inference_results.results`` and leaves ``metrics`` empty. The ground truth
  never leaves this machine.
* **Throughput and latency count 3-second segments, not files.** A one-minute
  recording is 20 inference samples. These numbers are not comparable to a
  DenseNet or YOLO run.

Layout expected under ``EXAMPLE_ROOT_DIR``::

    birdnet_onnx/
      model.onnx              # v3.0, published as ONNX (no conversion needed)
      labels.txt              # one "Sci name_Common Name" per line
      annotations.csv         # optional ground truth, read locally (never uploaded)
      *.wav                   # the dataset

The ground truth is birdnet's own ``predictions.csv`` schema minus the
``confidence`` column, so the quickest way to author one is to run the reference
package over the same audio, ``to_csv`` it, and delete that column::

    file_path,start_time,end_time,species_name
    "/…/rec_001.wav","00:00:00.00","00:00:03.00","Turdus merula_Common Blackbird"
"""

import csv
import os
from collections import defaultdict
from itertools import chain
import pathlib
from pathlib import Path, PurePosixPath

from dotenv import load_dotenv

from edge_benchmarking_client.client import AUDIO_EXTENSIONS, EdgeBenchmarkingClient
from edge_benchmarking_types.edge_farm.models import TritonBirdNetClient

GROUND_TRUTH_COLUMNS = ("file_path", "start_time", "end_time", "species_name")


# --------------------------------------------------------------------- scoring
# Everything in this section is plain stdlib so it can be lifted into your own
# script unchanged. It never talks to the Edge-Farm API.


def parse_timestamp(value: str) -> float | None:
    """Parse ``HH:MM:SS.CC`` (birdnet's format) or bare seconds.

    ``None`` means the row carries no usable time, which downgrades scoring to
    file level rather than silently mismatching every segment.
    """
    value = value.strip().strip('"')
    if not value:
        return None
    try:
        if ":" in value:
            seconds = 0.0
            for part in value.split(":"):
                seconds = seconds * 60.0 + float(part)
            return seconds
        return float(value)
    except ValueError:
        return None


def load_ground_truth(path: Path) -> dict:
    """Read birdnet long-format ground truth into lookup tables.

    Keyed by **basename**: birdnet writes absolute paths, while a benchmark run
    reports the object's file name.
    """
    per_segment: dict[tuple[str, float], set[str]] = defaultdict(set)
    per_file: dict[str, set[str]] = defaultdict(set)
    rows_with_times = total_rows = 0

    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [
            column
            for column in GROUND_TRUTH_COLUMNS
            if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(
                f"'{path}' is missing column(s) {missing}; expected birdnet's "
                f"predictions.csv schema minus 'confidence': {GROUND_TRUTH_COLUMNS}"
            )

        for row in reader:
            species = (row.get("species_name") or "").strip().strip('"')
            file_path = (row.get("file_path") or "").strip().strip('"')
            if not species or not file_path:
                continue

            basename = PurePosixPath(file_path.replace("\\", "/")).name
            total_rows += 1
            per_file[basename].add(species)

            start_s = parse_timestamp(row.get("start_time") or "")
            if start_s is not None:
                rows_with_times += 1
                per_segment[(basename, round(start_s, 3))].add(species)

    if not total_rows:
        raise ValueError(f"'{path}' contains no usable rows.")

    return {
        "per_segment": dict(per_segment),
        "per_file": dict(per_file),
        # Only segment-resolved if *every* row has a time; a few parseable rows
        # among many would give a precise-looking but mostly-unmatched score.
        "has_times": rows_with_times == total_rows,
    }


def score(predictions: list[dict], truth: dict, top_k: int) -> dict:
    """Micro precision/recall/F1 plus a per-file top-k rollup.

    Scores per 3-second segment when the ground truth has usable times and its
    segment grid lines up with the run. If the grids disagree -- e.g. the ground
    truth was produced with a different ``overlap_seconds`` -- every key misses
    and the naive score is 0.0, indistinguishable from a bad model, so it falls
    back to file level and says so.
    """
    pred_segment: dict[tuple[str, float], set[str]] = defaultdict(set)
    pred_file: dict[str, set[str]] = defaultdict(set)
    best_confidence: dict[tuple[str, str], float] = {}

    for row in predictions:
        name = PurePosixPath(str(row["file"])).name
        species = str(row["species_name"])
        start_s = round(float(row["start_s"]), 3)
        confidence = float(row["confidence"])

        pred_segment[(name, start_s)].add(species)
        pred_file[name].add(species)
        key = (name, species)
        best_confidence[key] = max(best_confidence.get(key, 0.0), confidence)

    note = None
    use_segments = truth["has_times"] and bool(truth["per_segment"])
    if use_segments and not (set(truth["per_segment"]) & set(pred_segment)):
        use_segments = False
        note = (
            "ground-truth segment boundaries do not line up with this run "
            "(check segment_seconds / overlap_seconds); scored per file"
        )
    elif not truth["has_times"]:
        note = "ground truth carries no times; scored per file"

    if use_segments:
        expected_map, actual_map = truth["per_segment"], pred_segment
        granularity = "segment"
    else:
        expected_map, actual_map = truth["per_file"], pred_file
        granularity = "file"

    true_positives = predicted_total = truth_total = 0
    for key in set(expected_map) | set(actual_map):
        expected = expected_map.get(key, set())
        actual = actual_map.get(key, set())
        true_positives += len(expected & actual)
        predicted_total += len(actual)
        truth_total += len(expected)

    precision = true_positives / predicted_total if predicted_total else 0.0
    recall = true_positives / truth_total if truth_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # Per-file rollup: is a true species among the file's top-k most confident?
    hits = scored = 0
    for name, expected in truth["per_file"].items():
        if not expected:
            continue
        scored += 1
        ranked = sorted(
            (
                (confidence, species)
                for (file_name, species), confidence in best_confidence.items()
                if file_name == name
            ),
            reverse=True,
        )
        if any(species in expected for _, species in ranked[:top_k]):
            hits += 1

    return {
        "granularity": granularity,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": true_positives,
        "predicted": predicted_total,
        "ground_truth": truth_total,
        "top_k_accuracy": round(hits / scored, 4) if scored else 0.0,
        "k": top_k,
        "files_scored": scored,
        "note": note,
    }


# ------------------------------------------------------------------- benchmark

if __name__ == "__main__":
    load_dotenv()

    EDGE_DEVICE_HOST = "edge-03"
    EXAMPLE_ROOT_DIR = "birdnet_onnx"
    TOP_K = 5

    client = EdgeBenchmarkingClient(
        protocol=os.getenv("EDGE_FARM_API_PROTOCOL"),
        host=os.getenv("EDGE_FARM_API_HOST"),
        username=os.getenv("EDGE_FARM_API_BASIC_AUTH_USERNAME"),
        password=os.getenv("EDGE_FARM_API_BASIC_AUTH_PASSWORD"),
    )

    print(client.get_device_header(hostname=EDGE_DEVICE_HOST))

    dataset = client.find_dataset(
        root_dir=EXAMPLE_ROOT_DIR, file_extensions=AUDIO_EXTENSIONS
    )
    model = client.find_model(root_dir=EXAMPLE_ROOT_DIR)
    labels = client.find_labels(root_dir=EXAMPLE_ROOT_DIR)

    inference_client = TritonBirdNetClient(
        protocol="http",
        host=EDGE_DEVICE_HOST,
        port=8000,
        # Must match the Triton model name, which the Edge-Farm API derives from
        # the uploaded model file's stem (model.onnx -> "model"). The pydantic
        # default is None, and that None is passed straight into the runtime
        # client, so leaving it unset asks Triton for a nameless model.
        model_name=model.stem,
        model_version="1",
        # Defaults reproduce the reference birdnet v3.0 pipeline: 32 kHz, 3 s
        # segments, no overlap, flat sigmoid.
        sample_rate=32_000,
        segment_seconds=3.0,
        overlap_seconds=0.0,
        top_k=TOP_K,
        confidence_thres=0.1,
        # Bounds the request size: one long recording can otherwise reach
        # hundreds of MB (1000 segments x 96000 samples x 4 bytes ~ 384 MB).
        # Clamped at run time to the model's max_batch_size.
        max_segments_per_request=256,
        batch_size=1,
        warm_up=True,
    )

    benchmark_job = client.benchmark(
        edge_device=EDGE_DEVICE_HOST,
        dataset=dataset,
        model=model,
        labels=labels,
        inference_client=inference_client,
        # Audio files are much larger than images; keep the upload chunk small.
        chunk_size=4,
        cleanup=True,
    )

    results = benchmark_job.inference_results

    # Rows are bucketed per Triton response id; flatten for a predictions.csv-shaped
    # table. This is the parsed model output, exactly as the device produced it.
    predictions = list(chain.from_iterable(results.results.values()))
    print(f"\n{len(predictions)} detections")
    for row in predictions[:10]:
        print(
            f"  {row['file']:24s} {row['start_s']:7.2f}-{row['end_s']:<7.2f} "
            f"{row['species_name']:45s} {row['confidence']:.4f}"
        )

    # Persist the raw job so results_to_labels.py can turn it into a label file
    # and so a run can be re-scored later without re-running the benchmark.
    pathlib.Path("results.json").write_text(benchmark_job.model_dump_json(indent=2))
    print("  (raw job written to results.json)")

    inference = results.performance.inference
    print(
        f"\nInference: {inference.sample_count} SEGMENTS "
        f"({inference.samples_per_second:.2f} segments/s, "
        f"avg {inference.latency.average * 1000:.2f} ms/batch)"
    )
    print(
        "  NOTE: the unit is 3-second segments, not files -- these numbers are "
        "not comparable to image-model benchmarks."
    )

    # ------------------------------------------------------------ local scoring
    ground_truth_path = Path(EXAMPLE_ROOT_DIR) / "annotations.csv"
    if not ground_truth_path.is_file():
        print(
            f"\nNo ground truth at '{ground_truth_path}'; skipping scoring. "
            "The detections above are still the full model output."
        )
    else:
        metrics = score(predictions, load_ground_truth(ground_truth_path), TOP_K)
        print(f"\nMetrics (computed locally, {metrics['granularity']} level):")
        for key in ("precision", "recall", "f1", "top_k_accuracy"):
            print(f"  {key:16s} {metrics[key]}")
        print(
            f"  {'matched':16s} {metrics['true_positives']} of "
            f"{metrics['ground_truth']} true / {metrics['predicted']} predicted"
        )
        if metrics["note"]:
            print(f"  NOTE: {metrics['note']}")
        print(
            "\n  precision/recall are bounded by the run's top_k and "
            "confidence_thres -- record them when comparing runs."
        )
