"""Lazy OpenPI/LIBERO bindings shared by the portable evaluation runners."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def add_openpi_arguments(parser):
    from bas_vla.runtime import env_path
    from bas_vla.runtime.calibration import add_calibration_arguments
    for flag, variable in [('openpi-root', 'BAS_OPENPI_ROOT'), ('libero-root', 'BAS_LIBERO_ROOT'),
                           ('libero-site-packages', 'BAS_LIBERO_SITE_PACKAGES'),
                           ('libero-config-path', 'LIBERO_CONFIG_PATH'), ('checkpoint-dir', 'BAS_OPENPI_CHECKPOINT')]:
        parser.add_argument('--' + flag, type=Path, default=env_path(variable))
    parser.add_argument('--config-name', default='pi05_libero')
    parser.add_argument('--resize-size', type=int, default=224)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--max-steps', type=int, default=280)
    parser.add_argument('--num-steps-wait', type=int, default=10)
    parser.add_argument('--replan-steps', type=int, default=5)
    parser.add_argument('--policy-noise-mode', choices=['paired_deterministic', 'internal_rng'], default='paired_deterministic')
    parser.add_argument('--image-shift-preset', choices=['clean', 'margin_noise_band', 'margin_noise_band_hard', 'perimeter_clutter_noise_mix'], default='clean')
    parser.add_argument('--distractor-manifest', type=Path)
    add_calibration_arguments(parser)


class OpenPIBackend:
    def __init__(self, args, expected_carrier=None):
        from bas_vla.integrations import openpi_libero as lib
        from bas_vla.runtime import require_path
        from bas_vla.runtime.calibration import CalibrationRuntime
        self.args, self.lib = args, lib
        lib.apply_coverage_compat()
        lib.register_external_roots(
            openpi_root=require_path(args.openpi_root, '--openpi-root', 'BAS_OPENPI_ROOT'),
            libero_root=require_path(args.libero_root, '--libero-root', 'BAS_LIBERO_ROOT'),
            libero_site_packages=args.libero_site_packages)
        lib.ensure_runtime_env(args.libero_config_path)
        from openpi.policies import policy_config
        from openpi.training import config
        train_config = config.get_config(args.config_name)
        checkpoint = require_path(args.checkpoint_dir, '--checkpoint-dir', 'BAS_OPENPI_CHECKPOINT')
        if expected_carrier is not None:
            from .manifests import verify_checkpoint
            data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
            if data_config.asset_id != expected_carrier['normalization_key']:
                raise ValueError('OpenPI config asset_id differs from manifest normalization_key')
            verify_checkpoint(expected_carrier, checkpoint, normalization_key=data_config.asset_id)
        self.policy = policy_config.create_trained_policy(train_config, checkpoint)
        self.runtime = CalibrationRuntime.from_args(args, self.policy)
        self.assets = lib.load_distractor_assets(args.distractor_manifest)

    def observation(self, obs, seed):
        image, wrist = self.lib.preprocess_image(obs, self.args.resize_size,
            dict(self.lib.IMAGE_SHIFT_PRESETS[self.args.image_shift_preset]),
            noise_seed=seed, distractor_assets=self.assets)
        return {'full_image': image, 'wrist_image': wrist,
                'state': np.concatenate((obs['robot0_eef_pos'],
                    self.lib.quat_to_axis_angle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']))}

    def infer(self, obs, prompt, seed):
        element = {'observation/image': obs['full_image'], 'observation/wrist_image': obs['wrist_image'],
                   'observation/state': obs['state'], 'prompt': prompt}
        kwargs = {}
        if self.args.policy_noise_mode == 'paired_deterministic':
            model = self.policy._model
            kwargs['noise'] = np.random.default_rng(seed).normal(
                size=(int(model.action_horizon), int(model.action_dim))).astype(np.float32)
        output = self.policy.infer(element, **kwargs)
        return np.asarray(output['actions'], dtype=np.float32), output.get('policy_timing', {})

    def predict(self, obs, prompt, seed, step):
        return self.runtime.predict(self.observation(obs, seed), prompt,
            lambda view, instruction: self.infer(view, instruction, seed), step_index=step)


def create_env(bddl_path, resolution, seed, env_kwargs=None):
    from libero.libero.envs import OffScreenRenderEnv
    kwargs = dict(env_kwargs or {})
    reserved = {'bddl_file_name', 'camera_heights', 'camera_widths'}
    if reserved & kwargs.keys():
        raise ValueError('env_kwargs cannot override BDDL or camera resolution')
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=resolution,
                             camera_widths=resolution, **kwargs)
    env.seed(seed)
    return env


def load_initial_states(path, *, allow_pickle=False):
    path = Path(path)
    if path.suffix == '.npy':
        return np.load(path, allow_pickle=False)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            return data['states'].copy()
    if not allow_pickle:
        raise ValueError('native LIBERO init files require --trust-init-files; only load trusted assets')
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)
