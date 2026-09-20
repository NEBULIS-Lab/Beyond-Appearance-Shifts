# Offline diagnostics

These tools summarize supplied evaluation records. All input files are JSON lists unless otherwise stated. Use one fixed protocol, carrier revision and hardware configuration per input when comparing results.

## Wilson intervals

```bash
python scripts/wilson_counts.py --input counts.json --output counts.md --format markdown
```

A row has `label`, integer `successes`, and positive integer `total`; optional metadata is retained in JSON/CSV. Formats are JSON, CSV, Markdown and LaTeX. The default confidence level is 95%. Intervals apply to aggregate binary counts, without converting them into paired-difference or seed-level uncertainty. Zero denominators and invalid counts are rejected. Every formatted rate and bound is expressed in percent.

## Gate condition matrix

```bash
python scripts/gate_diagnostics.py --input gate_chunks.json --output gate_matrix.json --unit chunk
```

Each record requires `task_id`, `episode_id`, `chunk_id` (for chunk units), boolean `semantic_changed` and `visual_shift`, scalar `gate` in `[0,1]`, and equally shaped `base_action`, `probe_action`, `reference_action`. Include `seed` to distinguish repeated episode IDs. Conditions produce a 2×2 matrix. The expert/reference target is true only when the probe has strictly smaller L2 distance than the base; ties are negative. Activation is `gate > threshold`, with default threshold zero.

Both false-event fractions over all observations and conditional false-positive/negative rates are exported, with each denominator recoverable from counts. Empty cells/rates are `null`. A chunk-level rate must not be labeled an episode-level rate. For an episode-level diagnostic, provide one declared representative decision/reference comparison per episode and use `--unit episode`; record that selection rule with the dataset.

Episode-unit records may additionally contain boolean `full_success` and `oracle_router_success`. The latter must come from a separate oracle-routed rollout under matched initialization. It is reported separately with its labeled denominator and never reconstructed by choosing whichever of two observed trajectories succeeded. Family labels and expert/reference information are diagnostic inputs only, not ordinary Full routing inputs. Chunk input rejects episode success labels to prevent repeating outcomes across chunks.

## Mask localization controls

```bash
python scripts/mask_diagnostics.py corrupt --input normal.npy --output corrupted.npy --kind iou --target-iou 0.5 --seed 7
python scripts/mask_diagnostics.py corrupt --input normal.npy --target-mask target.npy --output missed.npy --kind missed_target --seed 7
python scripts/mask_diagnostics.py corrupt --input normal.npy --output random.npy --kind random --seed 7
python scripts/mask_diagnostics.py summarize --input mask_outcomes.json --output mask_summary.json
```

Masks are binary 2D NumPy arrays. IoU corruption retains a seeded subset of foreground pixels, quantized to mask area; the sidecar JSON records requested and achieved IoU. For an IoU≥0.7 condition, verify achieved IoU meets that bound rather than relying on the request alone. Missed-target corruption removes an explicitly supplied target mask. The random control samples pixels uniformly without replacement and exactly matches foreground area. It may overlap the true target by chance. Empty/empty mask IoU is defined as one; IoU corruption requires foreground.

Feed corrupted masks into the same probe/fusion path, keep initialization and all non-mask settings matched, and supply measured outcomes. Outcome records require `task_id`, `corruption`, `seed`, `episode_id`, boolean `success`, and matched boolean `normal_success`. The summary reports count and percentage-point differences. Completeness means exactly four seeds with 50 episodes each, as in the appendix; smaller inputs are summarized and marked incomplete. Reference outcomes must agree across corruption levels.

## Latency protocol

```bash
python scripts/summarize_latency.py --input timing_chunks.json --output latency.json
```

Runtime episode outputs can first be flattened using `python scripts/export_timing_records.py --input episodes.jsonl --jsonl --pass-id pass1 --output timing_chunks.json`. The exporter also accepts OpenPI and appearance `summary.json` files with `per_episode` (`chunk_trace` or `preserving_trace`), and OFT summaries with `records` containing clean/control/break preserving traces; omit `--jsonl` for summaries. Triplet conditions are retained in `condition` and in the episode identity, so reset chunk indices do not collide. Summary-level seeds are inherited where present. The latency summarizer pools supplied conditions within each task/mode; filter exported rows by `condition` when separate condition distributions are required. Repeat for independently measured passes and concatenate the resulting row lists. This preserves synchronization and warm-up flags; it never upgrades ordinary traces into synchronized profiling data.

Each row requires `task_id`, `mode`, `pass_id`, `chunk_id`, boolean `warmup`, boolean `synchronized`, and `timing_ms`. Include `episode_id` if chunk indices restart by episode. `timing_ms.end_to_end` is a directly measured chunk interval in milliseconds. Other keys name measured components, such as `base`, `calibration`, `backend_setup`, `probe_prepare`, `probe`, and `gate_fusion`. Component intervals can overlap or be nested; the tool never sums them to infer end-to-end latency. A missing component is omitted, not imputed as zero, and every component reports its own sample count.

A profiling harness must mark warm-up chunks and ensure completion at timing boundaries. For accelerator work, synchronize at the timing boundaries before starting/stopping measured intervals; record the synchronization method and warm-up policy with the dataset. CPU operations qualify when they finish synchronously. End-to-end sidecar/Full timing includes grounding, mask refinement, probe preparation and the extra frozen-policy query even if the gate later becomes zero. Set timing boundaries around all these operations. Full-episode wall-clock duration is a separate unit and must not be mixed into this chunk distribution.

The summary reports p50/p95 per task/mode/pass and pooled per task/mode, using linear percentiles. Protocol completeness requires at least 200 measured chunks in each of at least three passes, with synchronized timing boundaries. Warm-up chunks are excluded. Unsynchronized rows are rejected by default. `--allow-unsynchronized` enables descriptive summaries of ordinary wall-clock traces; these are always marked incomplete. Runtime trace exports still need explicit pass IDs and complete profiling metadata before they qualify as protocol measurements.
