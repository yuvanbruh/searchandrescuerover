"""
After training: compare the trained RL policy with the Mission-2 rule INSIDE the
simulator (quick preview of the real Gazebo comparison).

    python eval_sim.py --ckpt runs/run1/best.pt --episodes 50
"""
import argparse
import numpy as np
import torch
from sar_rl import SarExplorationEnv, SimConfig
from sar_rl.policy import CandidatePolicy
from check_sim import mission2_like

KEYS = ["duration_sec", "distance_traveled_m", "final_coverage_pct", "true_inspected",
        "true_targets", "stuck_events", "db_entries_abandoned"]


def run(policy_fn, layout, nominal, episodes):
    env = SarExplorationEnv(SimConfig(domain_randomize=not nominal), layout=layout)
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=50_000 + ep)   # same seeds for both policies
        rng = np.random.default_rng(ep)
        done = False
        while not done:
            obs, r, te, tr, info = env.step(policy_fn(obs, rng))
            done = te or tr
        rows.append(info["metrics"])
    out = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in KEYS}
    out["completed"] = (float(np.mean([r["completed"] for r in rows])), 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    a = ap.parse_args()
    model = CandidatePolicy()
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])

    @torch.no_grad()
    def rl(obs, rng):
        logits, _ = model(torch.as_tensor(obs["cand"])[None], torch.as_tensor(obs["mask"] > 0)[None],
                          torch.as_tensor(obs["glob"])[None])
        return int(logits.argmax(-1))

    for layout, nominal in (("v2", True), ("v2", False), ("random", False)):
        print(f"\n=== {layout} ({'nominal' if nominal else 'randomised'} physics), {a.episodes} episodes ===")
        res = {n: run(f, layout, nominal, a.episodes) for n, f in (("mission2_like", mission2_like), ("RL", rl))}
        print(f"{'metric':24s}{'mission2_like':>22s}{'RL':>22s}")
        for k in KEYS + ["completed"]:
            m, r = res["mission2_like"][k], res["RL"][k]
            print(f"{k:24s}{m[0]:>14.2f} ±{m[1]:<6.1f}{r[0]:>14.2f} ±{r[1]:<6.1f}")


if __name__ == "__main__":
    main()
