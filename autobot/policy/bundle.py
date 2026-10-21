"""Load a released hover-policy bundle and run it with NumPy.

A bundle is a directory from an ezhou7/dreamerv3 GitHub release, e.g.
`skydreamer-smooth-9.4m-1c-2026-10-18/` with agent.pkl, config.yaml and
meta.json. The first load extracts the policy weights from agent.pkl
into policy_weights.npz (~23 MB) next to it.

Needs only numpy and ruamel.yaml, so the policy runs on the Orange Pi's
CPU and leaves its NPU (neural processing unit) to the object detector.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from autobot.policy.reference import NumpyPolicyInference
from autobot.policy.spec import ACTION_DIM, OBS_DIM
from autobot.policy.weights import extract_bundle

WEIGHTS_FILE = "policy_weights.npz"


@dataclass
class PolicyInfo:
    name: str
    path: Path
    meta: dict = field(default_factory=dict)


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
    if not weights.exists():
        extract_bundle(path, weights)
    policy = NumpyPolicyInference(weights, sampling=sampling, seed=seed)

    # The weights must match the observation/action layout this code builds.
    enc_in = policy.ref.w["enc/mlp0/kernel"].shape[0]
    if enc_in != OBS_DIM or policy.action_dim != ACTION_DIM:
        raise ValueError(
            f"{path.name}: policy expects obs {enc_in} / action "
            f"{policy.action_dim}, autobot builds {OBS_DIM} / {ACTION_DIM}")

    meta = {}
    if (path / "meta.json").exists():
        meta = json.loads((path / "meta.json").read_text())
    return policy, PolicyInfo(meta.get("name", path.name), path, meta)


def warm_up(policy, steps=5):
    """Run a few steps on zero input so the first real step isn't slow."""
    policy.reset()
    for i in range(steps):
        policy.step(np.zeros(OBS_DIM, np.float32), is_first=(i == 0))
    policy.reset()
