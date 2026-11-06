"""Load a released policy bundle and run it with NumPy.

A bundle is a directory from ezhou7/dreamerv3 (a GitHub release or
aerial/deploy/export_model.py), e.g. `skydreamer-smooth-9.4m-1c-2026-10-18/`
with agent.pkl, config.yaml and meta.json. The first load extracts the
policy weights from agent.pkl into policy_weights.npz (~23 MB for a hover
policy, ~28 MB for a camera policy) next to it.

Two kinds of policy:
  hover   22-dim vector input (autobot/policy/spec.py)
  camera  image + depth + 23-dim vector (autobot/policy/vision_spec.py)

Needs only numpy and ruamel.yaml, so the policy runs on the Orange Pi's
CPU and leaves its NPU (neural processing unit) to the object detector.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from autobot.policy.reference import NumpyPolicyInference
from autobot.policy import vision_spec
from autobot.policy.spec import ACTION_DIM, OBS_DIM
from autobot.policy.weights import META_KEY, extract_bundle

WEIGHTS_FILE = "policy_weights.npz"


@dataclass
class PolicyInfo:
    name: str
    path: Path
    meta: dict = field(default_factory=dict)
    kind: str = "hover"          # or "camera"


def load_policy(model_dir, sampling="mean_action", seed=None):
    """Returns (NumpyPolicyInference, PolicyInfo).

    sampling: 'mean_action' (deploy default: the policy's mean action),
    'deterministic' (also fixes the latent; repeatable, for tests) or
    'stochastic' (adds exploration noise; never fly with it).
    """
    path = Path(model_dir).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"no bundle directory at {path}")
    weights = path / WEIGHTS_FILE
    if not weights.exists() or _format_version(weights) < 2:
        extract_bundle(path, weights)     # (re)extract older, vector-only files
    policy = NumpyPolicyInference(weights, sampling=sampling, seed=seed)
    kind = "camera" if policy.ref.hp.get("cnn_layers", 0) else "hover"

    # The weights must match the observation/action layout this code builds.
    enc_in = policy.ref.w["enc/mlp0/kernel"].shape[0]
    want = vision_spec.VECTOR_DIM if kind == "camera" else OBS_DIM
    if enc_in != want or policy.action_dim != ACTION_DIM:
        raise ValueError(
            f"{path.name}: {kind} policy expects vector {enc_in} / action "
            f"{policy.action_dim}, autobot builds {want} / {ACTION_DIM}")
    if kind == "camera":
        cin = policy.ref.hp["cnn_in_channels"]
        if cin != vision_spec.IMAGE_CHANNELS:
            raise ValueError(f"{path.name}: camera policy expects {cin} image "
                             f"channels, autobot builds {vision_spec.IMAGE_CHANNELS}")

    meta = {}
    if (path / "meta.json").exists():
        meta = json.loads((path / "meta.json").read_text())
    return policy, PolicyInfo(meta.get("name", path.name), path, meta, kind)


def _format_version(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:   # reads only this key
        return json.loads(str(data[META_KEY])).get("format_version", 0)


def zero_observation(kind):
    if kind == "camera":
        res = vision_spec.CAM_RES
        return {"vector": np.zeros(vision_spec.VECTOR_DIM, np.float32),
                "image": np.zeros((res, res, 3), np.uint8),
                "depth": np.zeros((res, res, 1), np.uint8)}
    return np.zeros(OBS_DIM, np.float32)


def warm_up(policy, steps=5, kind="hover"):
    """Run a few steps on zero input so the first real step isn't slow."""
    policy.reset()
    for i in range(steps):
        policy.step(zero_observation(kind), is_first=(i == 0))
    policy.reset()
