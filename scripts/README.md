# Command-line tools

Carrier and benchmark runtimes are installed separately; export their paths from
[`.env.example`](../.env.example). See the [reproduction guide](../docs/reproduction.md)
for the paper-to-code map and required artifacts.

| Workflow | Scripts | Guide |
| --- | --- | --- |
| Adapter training and records | `train_breaking_adapter.py`, `export_triplet_records.py`, `summarize_cached_records.py`, `inventory_checkpoint.py` | [Training](../docs/training.md) |
| Native LIBERO rollouts | `eval_openpi_libero.py`, `eval_openpi_appearance_libero.py`, `eval_openvla_oft_libero.py` | [Runtime](../docs/runtime.md) |
| Independent new/old goals and random controls | `eval_dual_goal.py` | [Evaluation](../docs/evaluation.md) |
| External and non-object protocols | `eval_external_manifest.py` | [Evaluation](../docs/evaluation.md) |
| Matched strategy matrix | `run_strategy_comparison.py` | [Comparisons](../docs/comparisons.md) |
| Wilson confidence intervals | `wilson_counts.py` | [Diagnostics](../docs/diagnostics.md) |
| Gate, mask and timing summaries | `gate_diagnostics.py`, `mask_diagnostics.py`, `export_timing_records.py`, `summarize_latency.py` | [Diagnostics](../docs/diagnostics.md) |
| Real-robot trial summaries | `summarize_robot_trials.py` | [Robot annotations](../docs/robot.md) |
| Runtime paths and version/hash records | `check_runtime_env.py`, `capture_environment.py` | [Environment](../docs/environment.md) |
| Bootstrap/process summaries | `bootstrap_eval.py`, `export_process_metrics.py` | Script `--help` |
| Media and pair utilities | `make_pair_montage.py`, `make_video_montage.py`, `check_semantic_break_pairs.py` | Script `--help` |

Shell scripts provide clean/control/break and appearance campaign examples. The
[top-level launchers](../launchers/README.md) wrap them. Configure their task selections, seeds and rollout budgets for your evaluation.
Carrier runners use `--mode frozen|core|pres|full`; core/full require a compatible
residual checkpoint, while pres/full require explicit frozen-feature settings.
