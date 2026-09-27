# Evaluation protocols

The evaluation package implements independent goal checking, matched simulator initialization, and complete episode/chunk records. The scripts load simulator and carrier dependencies only when executing a rollout.

## Dual-goal object swaps

```bash
python scripts/eval_dual_goal.py \
  --pairs-config configs/evaluation/dual_goal_pairs.json \
  --checkpoint-dir "$BAS_OPENPI_CHECKPOINT" \
  --methods old_instruction changed_instruction \
  --seeds 7 --episode-indices 0 1 --output-dir runs/dual_goal
```

This example selects two episodes; configure the seeds and episode indices for your evaluation. Set the OpenPI/LIBERO roots as described in the runtime documentation. For calibrated actions, add `bas` to `--methods`, select `--mode core`, and supply the residual checkpoint, metadata, and vocabulary or frozen instruction provider. `pres` and `full` additionally require frozen carrier feature providers and gate configuration. The shared runtime validates action dimensions and conventions.

The supplied pair files contain five swaps and separate native, co-present corrected, and new-native crossover scene definitions. Four BDDL files supply the necessary co-present objects. Relative BDDL overrides resolve from the pair configuration's directory. These scene protocols remain separate in aggregate output. Use the scene protocol and fixed episode identities appropriate to the evaluation being reproduced.

Every method receives a fresh environment with the same deterministic seed and the same selected initial-state vector. The runner compares simulator-state SHA-256 digests after initialization. Missing or out-of-range initial-state indices raise an error; trials are never silently dropped. Matching covers initialization and policy noise, not later trajectories, which diverge with executed actions.

After each simulator step, including settling steps, the evaluator checks `In(object_1, basket_1_contain_region)` for the old target, new target, and every other physical object in the scene. It stops on the first hit. Native environment `done` is recorded but does not define changed-task success. Simultaneous goals produce `conflict`; the three-way report maps conflict, third-object success, and timeout to **Other/timeout**. A goal already satisfied at initialization is an error.

`old_instruction` and `changed_instruction` are frozen carrier queries with their respective prompts. `bas` executes the selected calibration mode using the changed instruction. No prompt-only intervention is labeled BAS.

### Random direction controls

Use `--methods random_direction` and explicitly choose one of:

- `--random-norm-reference carrier_delta`: query the old and changed instructions at the same observation, then perturb the old-instruction action by a random direction matching the changed-minus-old norm. Keep the old gripper command.
- `--random-norm-reference bas_residual`: query the changed instruction, apply the loaded residual calibrator, and perturb the changed-instruction action by a random direction matching the raw calibrated-minus-carrier norm. Keep the changed gripper command. Residual checkpoint arguments are required even when `--mode frozen` is selected for the other methods.

Both match each action's six continuous LIBERO dimensions before clipping, not the gripper or whole-chunk norm. Records include the reference, random pre-clipping, and random post-clipping norms. Continuous random actions are clipped to `[-1, 1]`; calibrated BAS actions also record their final clipping. A norm match before clipping need not remain exact afterward. These two controls have different reference actions and must be reported separately.

The default `paired_deterministic` policy-noise mode passes the same diffusion-noise tensor to queries at an identical pair/seed/episode/step. `internal_rng` is available explicitly and is recorded as a different protocol.

Each run writes `episodes.jsonl`, `summary.json`, and terminal simulator states. Action and inference traces are complete. Combine runs with:

```bash
python scripts/aggregate_dual_goal.py \
  --records runs/seed7/episodes.jsonl runs/seed11/episodes.jsonl \
  --require-methods old_instruction changed_instruction bas \
  --output runs/dual_goal_summary.json
```

The aggregator rejects duplicate identities, mismatched initial states, incomplete requested method coverage, and pooling different mode/norm/visual/noise settings under one method label.

## External Plus/PRO and non-object manifests

`eval_external_manifest.py` executes a fixed manifest against BDDL and initial-state files. It supports Plus/PRO success predicates and non-object old/new outcomes. A built-in OpenPI backend is included. Other carriers can supply `--backend-factory module:factory`; the returned backend implements `validate_carrier(carrier_manifest)` and `predict(raw_observation, instruction, noise_seed, step) -> (actions, trace)`.

Configure the checked-in `*.template.json` files with your evaluation inputs and set `status: "ready"`. Supply:

- An immutable benchmark revision, exact case selection ID, and case list.
- Carrier configuration, checkpoint ID, actual normalization key, and `checkpoint_files`, a mapping from checkpoint-relative filenames to SHA-256 digests.
- Per-case instruction, category, BDDL/init-state relative filenames and SHA-256 digests, normalization key, and explicit `{seed, episode_idx}` trial identities.
- `goal` for external success, or `old_goal` and `new_goal` for non-object evaluation. A goal is `{"kind": "all", "predicates": [["in", "object_1", "basket_1_contain_region"]]}` or an `ordered` list of predicates.

All case asset paths resolve under `--asset-root`; paths escaping that root are rejected. A manifest run uses one normalization key. Split a heterogeneous benchmark list into homogeneous carrier-normalization manifests without changing case/trial identities.

```bash
python scripts/eval_external_manifest.py --manifest my_manifest.json \
  --validate-only --paper-budget

python scripts/eval_external_manifest.py --manifest my_manifest.json \
  --asset-root "$BENCHMARK_ASSETS" --checkpoint-dir "$BAS_OPENPI_CHECKPOINT" \
  --checkpoint-id MY_CHECKPOINT_ID --normalization-key MY_NORMALIZATION_KEY \
  --mode frozen --output-dir runs/external_frozen --trust-init-files
```

Validation alone checks the schema and requested budgets without loading scene assets, models, or a simulator. Execution verifies scene/init-state digests. The built-in OpenPI backend obtains `asset_id` from the actual OpenPI data configuration, compares it with the manifest normalization key, and verifies all loaded weight files and `assets/<asset_id>/norm_stats.json` against `checkpoint_files` before loading the policy. Alternate backends must validate their own loaded weights and normalization in `validate_carrier`. The supplied checkpoint ID is a label; file digests provide the artifact binding.

Native `.pruned_init` files use PyTorch serialization and require `--trust-init-files`. Numeric `.npy` or `.npz` files (key `states`) need no pickle support. Use the matching benchmark fork and its assets. The environment factory has signature `(bddl_path, resolution, seed, env_kwargs)`. The default creates `OffScreenRenderEnv`; benchmark versions requiring runtime camera/noise/lighting wrappers must provide their actual wrapper factory and parameters. A category label alone does not apply a perturbation.

`--paper-budget` enforces Plus's four named categories × ten cases × 120 trials, PRO's ten cases × 120 trials, or non-object's three families × two tasks × four seeds × fifty trials. Specify the exact Plus 40-case or PRO 10-case selection in the corresponding manifest.

## Non-object and preserving protocols

For `protocol: "nonobject"`, the same manifest runner evaluates destination, spatial-relation, and order families. Destination and spatial-relation predicates use explicit LIBERO relations and argument identities. Order goals use `kind: "ordered"` for both old and new sequences: a future subgoal reached before its turn, or two subgoals first completed simultaneously, invalidates that sequence. Earlier subgoals may persist after completion. Initially satisfied ordered subgoals are rejected. Thus a final state satisfying all subgoals does not establish correct order. The exact six task definitions, BDDL assets, and trial list must be provided for the paper protocol.

`preserving_bowl_on_ramekin.template.json` describes the OFT Table 3 condition/method matrix, four-seed budget, and style family. Clean and matched-clean are separate columns. Configure the matched-clean identities, style assets/transform, checkpoint, and preserving settings for this matrix. `gate_condition_matrix.json` specifies the fixed Full 2×2 diagnostic conditions; oracle routing is reported separately.
