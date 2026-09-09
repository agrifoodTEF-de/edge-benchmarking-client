# BirdNET benchmark example

Everything for the BirdNET example lives here — scripts next to their data.

```
birdnet_onnx/
  benchmark.py                  # run a benchmark, then score it locally
  verify_against_reference.py   # prove the served model matches birdnet itself
  results_to_labels.py          # turn a run's output into a birdnet-schema label file

  model.onnx                    # BirdNET v3.0 ONNX      (you stage these)
  labels.txt                    # "Sci name_Common Name" per line
  *.wav                         # the dataset
  annotations.csv               # optional ground truth, read locally (never uploaded)
```

Credentials come from `../.env` — `load_dotenv()` walks up, so there is no second
copy to maintain.

```sh
conda activate birdnet
cd examples/birdnet_onnx

python benchmark.py                                  # -> results.json + metrics
python results_to_labels.py results.json --top-1     # -> annotations.csv
python verify_against_reference.py soundscape.wav    # -> MATCH / MISMATCH
```

> The folder is deliberately **not** called `birdnet`: a directory (or file) of
> that name inside `examples/` shadows the installed `birdnet` package whenever
> you run anything from there, and `import birdnet` silently picks up the empty
> local namespace instead.

## Getting the model

BirdNET **v3.0 is published as ONNX** — there is nothing to convert:

```python
import birdnet
model = birdnet.load("acoustic", "3.0", "onnx")   # needs birdnet[onnx]
print(model.model_path)   # cached .onnx -> copy it here as model.onnx
print(model.species_list) # -> labels.txt, one per line
```

The same file is downloadable straight from the Zenodo record if you would
rather not install the package.

### What is staged here

`model.onnx` and `labels.txt` are byte-identical to what `birdnet` **v1.1.1**
(`5685f3d`, github.com/birdnet-team/birdnet) ships for acoustic v3.0 ONNX —
verified by sha256 against `model.model_path`, and by a 0.0 max-delta run
against `birdnet.load(...)`. Re-stage them with the snippet above after any
`birdnet` upgrade, and re-check both.

- `model.onnx` is the **fp32** export: 378 `FLOAT` initializers, no `Cast`
  nodes, 541 MB.
- Until 2026-09-07 an **fp16** export was staged instead (378 `FLOAT16` + 96
  `Cast` nodes, 271 MB), kept as `model-fp16.onnx.bak` alongside the previous
  `labels-old.txt.bak`. It differs from fp32 by up to **2.06e-02** in
  confidence — above this directory's 0.01 verification tolerance — so results
  produced before that date are not comparable to later ones. Delete the `.bak`
  files once nothing depends on the old numbers; note that a second `.onnx` in
  this directory would make `find_model` "select first match" with only a
  warning, which is why the backup does not carry that extension.

> **v2.4 is out of scope.** It has no official ONNX export, and edge-farm's
> tools-api converts PyTorch, not TensorFlow. You would need an external
> `tf2onnx` pass first. If you do produce one, set `sample_rate=48000` and
> `segment_seconds=3.0` (144 000 samples) on the client.

## Scoring

Nothing is scored server-side — `inference_results.metrics` stays empty. The
scoring lives in `../birdnet_benchmark.py` (`load_ground_truth` / `score`), which reads
`annotations.csv` **from this directory** and never uploads it. Both functions are
plain stdlib, so lift them into your own script unchanged.

It reports micro precision / recall / F1 plus a per-file top-k rollup, scored per
3-second segment when the times line up and per file otherwise. Delete
`annotations.csv` and the run still works — it prints the detections and skips
scoring.

The format is birdnet's own `predictions.csv` **minus the `confidence` column**,
so the quickest way to author one is to run the reference package over the same
audio and delete that column:

```python
model.predict("recordings/").to_csv("predictions.csv")
```

which gives one row per (file, segment, species):

```
file_path,start_time,end_time,species_name,confidence
"/…/rec_001.wav","00:00:00.00","00:00:03.00","Turdus merula_Common Blackbird",0.814058
"/…/rec_001.wav","00:00:03.00","00:00:06.00","Parus major_Great Tit",0.308293
```

Two things to watch: `species_name` must match `labels.txt` byte-for-byte, and the
segment grid must line up (same `segment_seconds` / `overlap_seconds`). `score()`
detects a grid mismatch and falls back to file-level scoring with a `NOTE` instead
of reporting a misleading 0.0 — check the `granularity` it prints.

Times may also be bare seconds (`3.0`), or left empty to score per file.

## Reading the results

`inference_results.results` holds the parsed model output, bucketed per Triton
response id. Flatten it with `chain.from_iterable(results.values())` for a flat
table of `file` / `start_s` / `end_s` / `species_name` / `confidence` rows.
`inference_results.metrics` stays empty — scoring is yours to do.

`performance.inference.sample_count` counts **3-second segments, not files** —
a one-minute recording is 20 samples. Do not compare these throughput or
latency numbers against a DenseNet or YOLO run.
