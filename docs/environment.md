# Runtime dependencies and provenance

The common Python package requirements are listed in
[`requirements.txt`](../requirements.txt). Install the carrier and benchmark
with their own supported dependency set. OpenPI and OpenVLA-OFT may require
separate environments; the common requirements are not a combined lockfile for
both upstream projects.

Before an evaluation, record the exact installed versions, external repository
revisions and relevant artifact hashes:

```bash
python scripts/capture_environment.py \
  --repo openpi="$BAS_OPENPI_ROOT" \
  --repo libero="$BAS_LIBERO_ROOT" \
  --artifact adapter=/path/to/residual_adapter.pt \
  --artifact init_states=/path/to/init_states.pt \
  --output outputs/environment.json
```

This command reads package metadata without importing a model, reports dirty
external checkouts, and hashes artifacts on the CPU. It records labels and
hashes, not machine paths or Git remote credentials. Supply additional
`--artifact LABEL=PATH` entries for checkpoint shards and normalization files.
Keep the environment record with the corresponding rollout outputs.

Provide the trained weights and record the matching carrier revision. The
checkpoint inventory, evaluation manifests and runtime documentation list the
inputs for each workflow. Use an evaluated
checkpoint's own normalization and action layout; a different carrier or feature
representation requires a compatible adapter.
