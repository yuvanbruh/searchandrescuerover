"""
Sanity / calibration script for the training simulator.

  python check_sim.py                 # v2 world + random worlds, 2 reference policies
  python check_sim.py --episodes 20 --png

'mission2_like' is a re-implementation of the Mission 2 decision rule INSIDE the
simulator. It is NOT used for training. It exists only so you can compare the
simulator's numbers (coverage, distance, time, inspected, abandoned) against
your real Gazebo runs and tune SimConfig until they look alike.
"""
import argparse
import time
import numpy as np
from sar_rl import SarExplorationEnv, SimConfig


def random_policy(obs, rng):
    return int(rng.choice(np.where(obs["mask"] > 0)[0]))


def mission2_like(obs, rng, alpha=3.0, beta=0.25):
    cand, mask, glob = obs["cand"], obs["mask"] > 0, obs["glob"]
    is_t = cand[:, 0] > 0.5
    tm = mask & is_t
    if tm.any():                                   # targets first, shortest Nav2 path
        idx = np.where(tm)[0]
        return int(idx[np.argmin(cand[idx, 1])])
    beta_eff = beta * (1.0 + (1.0 - glob[2]))      # battery-urgency scaling
    u = alpha * cand[:, 3] - beta_eff * cand[:, 2]
    u[~mask] = -1e9
    return int(np.argmax(u))


def run(policy, layout, episodes, nominal, seed0=0, png=None):
    cfg = SimConfig(domain_randomize=not nominal)
    env = SarExplorationEnv(cfg, layout=layout)
    rows, t0, decisions = [], time.time(), 0
    for ep in range(episodes):
        obs, info = env.reset(seed=seed0 + ep)
        rng = np.random.default_rng(ep)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs, rng))
            decisions += 1
            done = term or trunc
        rows.append(info["metrics"])
        if png and ep == 0:
            env.save_png(png)
    dt = time.time() - t0
    keys = ["duration_sec", "distance_traveled_m", "final_coverage_pct", "true_targets",
            "true_detected", "true_inspected", "db_entries_detected", "db_entries_inspected",
            "db_entries_abandoned", "stuck_events", "decisions"]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    mean["completed_frac"] = float(np.mean([r["completed"] for r in rows]))
    return mean, dt, decisions


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    for layout in ("v2", "random"):
        for name, pol in (("random", random_policy), ("mission2_like", mission2_like)):
            png = f"sim_{layout}_{name}.png" if a.png else None
            m, dt, n = run(pol, layout, a.episodes, nominal=(layout == "v2"), png=png)
            print(f"\n[{layout} | {name}]  {a.episodes} eps, {dt:.1f}s wall, "
                  f"{1000*dt/max(n,1):.1f} ms/decision")
            print("  " + "  ".join(f"{k}={v:.2f}" for k, v in m.items()))
