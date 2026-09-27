# Carrier calibration runtime

All three carrier runners accept `--mode frozen|core|pres|full`. The default is
`frozen`, which needs only the carrier checkpoint. `core` is the BAS breaking
core; `pres` fuses the frozen action with a probe; `full` fuses the calibrated
core action with that probe. Compatibility aliases are `baseline -> frozen`,
`breaking -> core`, and `default -> core`. `--preserving-mode` remains an alias
for `--mode`. In particular, the old `default` spelling now requires a residual
checkpoint; use `frozen` for an unmodified baseline.

Core and Full require `--residual-checkpoint`, `--residual-metadata`, and either
`--residual-vocab` or `--instruction-provider module:factory`. Metadata must
contain the existing `input_dim`, `hidden_dim`, `output_dim`, `delta_scale`, plus
`action_space` and `action_layout`. Legacy files can supply the last two via
`--residual-action-space` and `--residual-action-layout`; contradictions are
rejected. The spaces are `libero_env` for OpenPI and `oft_pre_process_action`
for OFT (before its carrier `process_action` gripper conversion). A checkpoint
trained in another normalization must be converted or wrapped explicitly.

`per_step` applies the residual to each row of a chunk with shared instruction
features; `flattened_chunk` applies it once to the flattened chunk and restores
the shape. `output_dim` must match the selected layout exactly. No dimensions
are truncated, padded, clipped, or silently exempted. The checkpoint's
`delta_scale` is applied by its adapter; `--residual-scale` multiplies the
resulting residual once. `--residual-device cpu` is the default.

Binary BoW features use the checkpoint vocabulary's original contiguous indices
and presence values, including the original lowercase alphanumeric tokenizer.
Unknown tokens have no vocabulary entry. Match the feature representation to
the checkpoint metadata. A different frozen text encoder is supplied as a
factory called with `carrier=...,
config=metadata`, returning a callable `instruction -> one-dimensional vector`.
Its dimension must equal `input_dim - output_dim`.

## Frozen carrier evidence

Pres and Full require `--feature-provider` and `--feature-config`; no carrier
layer numbers are guessed. The configuration and provider metadata must state
`mid_layer`, `late_layer`, `pooling`, and `normalization`. Gaps are Euclidean
norms after flattening the provider arrays, optionally normalizing each array to
unit L2 norm (`normalization: l2`); `none` preserves the array scale. Zero arrays
cannot be L2-normalized. The mid gap controls nuisance evidence; the late gap
controls semantic consistency. Action disagreement always compares the frozen
base chunk with the probe chunk, including in Full.

For a PyTorch carrier, the built-in provider is
`bas_vla.runtime.features:named_module_features`. Set exact `named_modules()`
paths in `mid_layer` and `late_layer`, and choose `pooling: flatten` or
`pooling: mean_tokens` (the last tensor dimension is channels). An optional
integer `output_index` selects a tuple output. These are forward hooks on the
existing frozen carrier, so they add no extra carrier forward. Repeated calls
to a selected module are concatenated in execution order. Supply paths for the
actual visual backbone of your checkpoint; a text-only layer is not a visual
semantic feature. The OFT runner passes the loaded model to this factory.

OpenPI implementations differ in their exposed intermediates. Supply a factory
for the actual loaded policy rather than applying PyTorch hooks to an opaque
JAX policy. Every provider implements:

```python
metadata = {"mid_layer": "...", "late_layer": "...",
            "pooling": "...", "normalization": "none"}
def before_inference(): ...  # clear previous captures

def after_inference(observation, instruction):
    return {"mid": captured_mid_array, "late": captured_late_array}
```

The factory receives `carrier` and the parsed feature JSON as `config`. It must
capture frozen visual representations during the carrier call. A missing hook
or invalid feature shape raises an error. The legacy
`compute_visual_gap_from_observations` helper remains a pixel-distance diagnostic;
these runners never use it for feature evidence. No semantic gap is filled with
a constant. A disabled probe is logged without unmeasured feature gaps.

Gate parameters can be supplied in `--gate-config` as `PreservingGateConfig`
fields: `alpha`, `phase_horizon_steps`, `phase_start_step`, `kappa_vis`,
`kappa_sem`, and `kappa_act`. Feature scaling and gate thresholds must be selected
together using the selected checkpoint's feature configuration.

## Composition and trace contract

```python
from bas_vla.runtime.calibration import CalibrationRuntime, add_calibration_arguments

add_calibration_arguments(parser)
runtime = CalibrationRuntime.from_args(args, carrier, action_space="libero_env")
# observation contains full_image, wrist_image, state.
# infer(observation, instruction) returns a decoded action chunk, or (chunk, timing).
actions, trace = runtime.predict(observation, instruction, infer, step_index=step)
```

A runtime caches one grounded backend. Each chunk prepares its probe once and
passes that same prepared probe to final gate/fusion evaluation. Full's zero gate
returns the calibrated action exactly; Pres's zero gate returns the frozen
action. The default selector can use the explicit `style_probe` fallback when
grounded models are unavailable; `probe_name` and `backend_name` record which
path was used. For a grounded-only evaluation construct `PreservingPipelineConfig`
with `selector_config.allow_style_fallback=False` and configure the grounded
backend described in the preserving package.

Every chunk is retained, including disabled probes. Traces contain base, anchor,
probe (when queried), output actions, all four gate factors, measured gaps,
feature metadata, and component wall-clock timings. `timing_ms.end_to_end`
includes carrier calls, feature capture, residual evaluation, probe preparation,
and gate/fusion; it excludes environment stepping and image preprocessing.
`chunk_id` increases for the runtime lifetime. `warmup=false` and
`synchronized=false` identify ordinary rollout timing. Carrier timing
dictionaries are retained separately. For synchronized profiling, set `runtime.synchronize` to the carrier backend's
blocking synchronization function (covering every device used by carrier and
auxiliary), then call `predict(..., task_id=..., pass_id=..., warmup=...)`.
The runtime calls that boundary before total timing and after timed components,
and sets `synchronized=true` only when a callback is supplied. The callback is
the caller's responsibility; no device query or guessed backend is used. Retain
at least the protocol's required chunks and passes after declared warm-up before
reporting device latency percentiles.

## OFT scene style evaluation

`eval_openvla_oft_libero.py --scene-style style_swap` configures LIBERO's
`scene_properties` with `floor_style=rustic` and `wall_style=dark-blue`; both are
CLI configurable. This changes renderer materials and leaves the initial state
and task geometry intact. `--scene-style clean` omits those overrides. It is a
scene intervention, separate from the preserving probe.

Use identical seed and explicit `--episode-indices` in clean and style runs to
match initial states; record the exact subset indices externally. The runner
applies each selected initial state to all three instructions. Configure the
matched-clean subset using the evaluation's episode identities. The carrier
runners report native LIBERO success;
use the separate dual-goal evaluator for New/Old/Other outcomes.
