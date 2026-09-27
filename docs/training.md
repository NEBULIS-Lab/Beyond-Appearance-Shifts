# Adapter training and records

The trainer optimizes a small bounded residual MLP over precomputed frozen-carrier actions and binary bag-of-words instruction features. It does not load or update a VLA. Vocabulary is built from the training split only; unseen validation tokens are ignored. All clean/control/break entries in a record must come from the same observation and context, and `expert_action` must identify the clean expert/reference action in the same coordinate system.

`examples/records/synthetic_triplets.json` is a two-record synthetic format example. `bas_vla/records/schema.json` describes the JSON list. Python validation additionally checks finite numbers, matching action dimensions, duplicate optional record IDs, and train/validation splits. Each vector is a single action or an explicitly flattened chunk; do not mix layouts. Assign all records from the same rollout/state source to one split before export.

```bash
python scripts/export_triplet_records.py --input my_triplets.jsonl --jsonl --output data/triplets.json
python scripts/train_breaking_adapter.py --records-path data/triplets.json \
  --config configs/training/paper_three_component.json \
  --carrier MY_EXACT_CARRIER_CHECKPOINT --action-space libero_env \
  --action-layout per_step --action-normalization MY_ACTION_CONVENTION \
  --device cpu --output-dir artifacts/my_adapter
```

The export tool validates and packages already computed matched triplets; it does not infer actions, construct expert labels, or assign train/validation splits. Preserve source state IDs, carrier revision, action normalization, and data provenance alongside your records. An environment-specific data collector must supply those inputs. `examples/train_breaking_adapter.sh` exposes the same command through required environment variables. `--help` does not query accelerator availability; the default device is CPU.

## Objective definitions

Let `C,U,B,E` be calibrated clean, control, break and reference actions. The separation term is `max(0, margin - ||B-C||₂ + ||U-C||₂)`, averaged over records. The expert-relative extra term replaces `C` by `E`. The anchor is `MSE(C,Cbase) + MSE(U,Ubase) + 0.25 MSE(B,Bbase)`.

The total is `lambda_imitation * imitation + lambda_consistency * consistency + lambda_margin * separation + lambda_margin * lambda_expert_margin * expert_margin + lambda_anchor * anchor`.

| Preset | Squared-error reduction | Consistency | Separation | Extra expert multiplier | Anchor |
| --- | --- | ---: | ---: | ---: | ---: |
| `legacy` | mean over all coordinates/records | 0.5 | 1.0 | 1.0 | 0.2 |
| `paper_three_component` | sum coordinates, mean records | 0.5 | 1.0 | 0.0 | 0.0 |
| `paper_without_consistency` | sum coordinates, mean records | 0.0 | 1.0 | 0.0 | 0.0 |
| `paper_without_separation` | sum coordinates, mean records | 0.5 | 0.0 | 0.0 | 0.0 |

Imitation sums the clean/reference and control/reference squared errors; consistency measures clean/control squared error. The `legacy` preset preserves the supplied implementation's objective and numeric defaults. The paper-form preset implements the three mathematical components in the method section, including squared L2 rather than coordinate-mean errors. Configure numerical settings for the selected carrier and training data. The two removal presets define objective ablations. The instruction-margin-only and semantic-margin-only comparators use their own recipes and checkpoints.

JSON configurations contain `objective_preset` and `overrides`. An explicit CLI preset or numeric option overrides the JSON. The complete resolved objective, including zero-valued extras and reductions, is stored in results and metadata. The seed, carrier, action space/layout and normalization are also recorded. The vocabulary is saved alongside the adapter. Use `libero_env` for the OpenPI runtime convention and `oft_pre_process_action` for the OFT runtime convention only when those names correctly describe your precomputed actions.

## Artifact inventory

A training run writes `residual_adapter.pt`, `adapter_metadata.json`, `vocab.json`, and `results.json`. Inspect/hash them without deserializing weights:

```bash
python scripts/inventory_checkpoint.py --artifact-dir artifacts/my_adapter --output artifacts/my_adapter/inventory.json
```

`configs/checkpoints.json` lists the required adapter files and metadata. Provide the trained weights, matching data/split manifests, carrier revisions and comparator configurations for your evaluation. Use matched records from the selected carrier for training.
