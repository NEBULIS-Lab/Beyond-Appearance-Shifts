# Reproducing the paper evaluations

The code exposes explicit modes and protocols. `frozen` executes the carrier;
`core` applies the residual calibrator; `pres` adds the preserving probe to the
frozen carrier; `full` adds it to the calibrated action. Use the same carrier,
action normalization and initial states when comparing modes.

## Evaluation map

| Paper evaluation | Entry point / configuration | Required evaluation assets |
| --- | --- | --- |
| Clean/control/break success | `eval_openpi_libero.py`, `eval_openvla_oft_libero.py`; semantic pair configs | Carrier and compatible adapter checkpoints; instruction/task selection |
| Bowl-on-Ramekin style shift | OFT runner with `--scene-style style_swap`; `configs/evaluation/` | Evaluated OFT checkpoint, feature layer/gate settings, exact seeds and matched episode indices |
| New/Old/Other outcomes and random residual | `eval_dual_goal.py`; `configs/evaluation/dual_goal_pairs*.json`, `bddl/` | Carrier and adapter checkpoints; selected scene protocol; explicit random norm reference |
| LIBERO-Plus / LIBERO-PRO | `eval_external_manifest.py`; external protocol templates | Exact case list, benchmark revision, BDDL and initial-state files, checkpoint/normalization identifiers |
| Destination, relation and order changes | External manifest runner with `protocol=nonobject` | Exact tasks, instruction pairs, predicates, four-seed initial-state assignments |
| Gate and mask diagnostics | `gate_diagnostics.py`, `mask_diagnostics.py` | Expert/reference annotations, model traces and reference/estimated masks |
| Chunk latency | `summarize_latency.py` | Measured timing records with warm-up, synchronization, task and pass identifiers |
| Real-robot cases | `bas_vla/deployment/`, `summarize_robot_trials.py` | Rig-specific controller/acquisition backend, calibration, checkpoints and trial annotations |
| External strategy comparisons | `run_strategy_comparison.py`; `configs/comparisons/` | Installed method implementations, exact revisions/licenses and matched initialization manifest |
| Wilson intervals | `wilson_counts.py` | Success/total counts from the corresponding evaluation |

See [evaluation](evaluation.md), [runtime](runtime.md), [diagnostics](diagnostics.md),
[robot deployment](robot.md), and [comparisons](comparisons.md) for input schemas
and commands. Paths in the table refer to files in `scripts/` unless indicated.

## Artifacts and configuration

[`configs/checkpoints.json`](../configs/checkpoints.json) lists required adapter
files and metadata. Provide the carrier and adapter weights for your evaluation.
The training recipe, record schema and export tool are described in
[training](training.md).

Configure the case selection, benchmark revision, initial-state files and trial
identities in the evaluation manifests. For OFT style evaluation, specify the
matched-clean episode indices and carrier feature configuration. Install the
comparison methods at their specified revisions. For robot deployment, configure
the PiPER/Orbbec backend, camera settings and action calibration.

Capture installed versions, source revisions and artifact hashes using
[`capture_environment.py`](../scripts/capture_environment.py). Detailed
[environment instructions](environment.md) explain how to keep these records with
rollout outputs without recording machine paths.
