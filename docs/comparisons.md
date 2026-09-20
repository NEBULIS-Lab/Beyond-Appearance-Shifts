# Matched-carrier strategy comparisons

[`milk_swap_protocol.json`](../configs/comparisons/milk_swap_protocol.json)
records the task, methods, and 4 × 50 episode budget per condition. The Break
metric is original-goal success under the changed instruction. Its interpretation
is distinct from the New/Old/Other metric in [dual-goal evaluation](evaluation.md).

Third-party strategy implementations remain separately installed dependencies.
The driver does not substitute a heuristic implementation for SGAC-AC, AAC,
RTC, PCD, ST4-VLA, or HAMLET. Supply each method's actual implementation, source
revision, and license together with the common carrier checkpoint and task state
artifacts. Record the source revision for each installed comparison method.

## Manifest format

`run_strategy_comparison.py` accepts a JSON object with:

- `carrier`: `name`, a 40-character `revision`, a 64-character
  `checkpoint_sha256`, and a nonempty `normalization_id`.
- `seeds`: distinct integer seeds; `episodes_per_seed`: a positive integer.
- `tasks`: each has `task_id`, an `init_states_sha256`, exactly
  `episodes_per_seed` distinct `init_state_ids`, and `instructions` containing
  `clean`, `control`, and `break` strings.
- `methods`: each has `name`, a 40-character `revision`, `source_url`, `license`,
  and a `command` argument list. Extra method configuration fields pass through
  unchanged. Include `{spec}` and `{output}` as separate arguments, for example
  `["python", "my_method_adapter.py", "--spec", "{spec}", "--output", "{output}"]`.

Names and task IDs use letters, digits, single underscores, dots and hyphens;
double underscores are reserved for run-directory separators.

Each adapter reads the generated specification, executes its condition using the
specified carrier, initial-state IDs, episode seed, and instruction, and writes
its result to the designated path. It must preserve the original environment
goal when reporting this table's metric. Source hashes identify requested
artifacts; the adapter is responsible for checking its installed artifacts and
recording those hashes in the output.

```bash
# Validate the manifest and generate the complete matched run matrix.
python scripts/run_strategy_comparison.py \
  --manifest /path/to/comparison_manifest.json --output-dir outputs/comparison_plan

# Execute the installed adapters in a new output directory.
python scripts/run_strategy_comparison.py \
  --manifest /path/to/comparison_manifest.json --output-dir outputs/comparison_run \
  --execute
```

The default command only writes specifications. Execution uses argument lists
without a shell and stops on a failed adapter. Existing per-run directories are
not overwritten. Use the same task-state artifact and IDs across all methods;
changing a method does not change the initialization or episode budget.
