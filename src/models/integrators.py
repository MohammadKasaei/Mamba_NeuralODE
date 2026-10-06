"""Differentiable fixed-step integration on [0,1]; no adaptive tolerances."""
NFE_PER_STEP = {"euler": 1, "heun": 2, "rk4": 4}


def integrate(field, state, steps=1, method="euler", return_trajectory=False):
    if steps < 1 or method not in NFE_PER_STEP:
        raise ValueError("Positive steps and euler/heun/rk4 required")
    dt = 1.0 / steps
    trajectory = [state] if return_trajectory else None
    for _ in range(steps):
        k1 = field(state)
        if method == "euler":
            state = state + dt * k1
        elif method == "heun":
            k2 = field(state + dt * k1)
            state = state + (dt / 2) * (k1 + k2)
        else:
            k2 = field(state + dt * k1 / 2)
            k3 = field(state + dt * k2 / 2)
            k4 = field(state + dt * k3)
            state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        if trajectory is not None:
            trajectory.append(state)
    return (state, trajectory) if return_trajectory else state
