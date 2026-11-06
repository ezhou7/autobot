"""DreamerV3 policies trained in ezhou7/dreamerv3, running on NumPy: the
hover policy (vector input) and the camera tracking policy (image, depth
and vector inputs)."""
from autobot.policy.bundle import PolicyInfo, load_policy, warm_up, zero_observation
from autobot.policy.spec import (
    ACTION_DIM, MAX_BODY_RATE_RPY, OBS_DIM, STEP_RATE_HZ, get_obs_vector,
    scale_action)

__all__ = [
    "ACTION_DIM", "MAX_BODY_RATE_RPY", "OBS_DIM", "STEP_RATE_HZ",
    "PolicyInfo", "get_obs_vector", "load_policy", "scale_action", "warm_up",
    "zero_observation",
]
