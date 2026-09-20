# Robot deployment and trial annotations

The paper uses a single-arm Songling PiPER controlled through `piper_sdk` over
CAN, with two Orbbec RGB-D input cameras and one overview camera used for
recording. Policy inputs use RGB only. The four cases and trial budgets are in
[`real_robot_cases.json`](../configs/deployment/real_robot_cases.json).

## Connecting a calibrated controller

[`bas_vla.deployment.robot`](../bas_vla/deployment/robot.py) provides a portable
boundary between policy inference and a hardware controller:

1. Acquire full-view and wrist-view RGB images and checkpoint-compatible state.
   `model_observation(full_rgb, wrist_rgb, state)` creates the policy observation;
   depth and overview-camera streams stay in the recording pipeline.
2. Query the carrier and apply the selected calibration mode using the runtime
   integration. Decode its action convention before robot execution.
3. Create `ActionCalibration` with the action scale, offset, lower/upper command
   bounds, and an explicit coordinate frame. It accepts seven coordinates:
   six absolute joint angles in radians or an absolute Cartesian pose
   `(x, y, z, roll, pitch, yaw)` in metres/radians, followed by gripper opening in
   metres. It does not silently convert delta actions to absolute poses.
4. Implement `RobotBackend.send_action(action, *, space, coordinate_frame)` in
   your PiPER controller. `execute_chunk` validates the entire chunk before
   sending any command and schedules commands at the specified `control_hz`.
   It records scheduled and actual send times; those are command timestamps,
   not measurements of achieved motion.

Configure the PiPER/Orbbec acquisition and control backend for your rig, including
camera settings, action calibration, scene layouts and acquisition synchronization.
Provide the matching robot checkpoint before deployment. Hardware initialization
belongs to the acquisition/control backend.

## Trial records

Store a JSON list, with one entry per physical trial:

```json
{
  "trial_id": "T1-core-break-000",
  "case_id": "T1",
  "method": "core",
  "condition": "break",
  "reset_id": "T1-layout-000",
  "instruction": "the exact instruction used for this trial",
  "annotator": "annotator identifier",
  "new_success": null,
  "original_success": null,
  "old_target_executed": null,
  "new_target_first_commit": null,
  "original_subset": false
}
```

Use booleans for assessed labels and `null` for unassessed labels. Include
`original_success` for clean/control conditions and `new_success` for changed
instructions. First-commit and old-target execution are independently annotated
behavioral labels. Record the same `reset_id` for matched physical layouts.
`trial_id` must be globally unique. Optional video paths and measured timestamps
can remain alongside these fields.

```bash
python scripts/summarize_robot_trials.py \
  --records /path/to/trials.json --output outputs/robot_summary.json
```

The summary reports each label's observed denominator. Unknown labels do not
become failures. Separation is the mean of old-target suppression and new-target
first-commit over trials where both labels are observed. Use
`--original-subset-only` for the initial T1/T2 protocol. Its 17 changed trials per
method/case are already part of the 80-trial extended evaluation and should not
be counted twice.
