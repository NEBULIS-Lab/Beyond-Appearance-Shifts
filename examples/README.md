# Examples

Export runtime paths from [`.env.example`](../.env.example) before running carrier
examples. These are entrypoint demonstrations; full paper protocols require the
artifacts listed in the [reproduction guide](../docs/reproduction.md).

- `run_openpi_semantic_break.sh`: one native LIBERO carrier evaluation.
- `run_openpi_appearance_shift.sh`: one OpenPI appearance intervention.
- `run_openvla_oft_semantic_break.sh`: native OFT clean/control/break evaluation.
- `train_breaking_adapter.sh`: train from user-supplied matched records with
  explicit carrier, action-space and normalization metadata.
- `records/`: synthetic records illustrating the training schema.

For calibrated modes, use the explicit arguments in the [runtime guide](../docs/runtime.md).
For training and record export, see [training](../docs/training.md).
