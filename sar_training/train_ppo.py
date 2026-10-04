"""
Train the RL policy from scratch with PPO in the 2D simulator.

    pip install torch gymnasium numpy scipy matplotlib
    python train_ppo.py --name run1

Leave it running. Checkpoints / logs go to runs/<name>/ :
    log.csv          progress every update
    ckpt_latest.pt   resume point (python train_ppo.py --name run1 --resume)
    best.pt          best policy on a FIXED validation set of random layouts
                     (sar_building_v2 is never used for training or selection)
"""
import argparse
import csv
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from sar_rl import SarExplorationEnv, SimConfig
from sar_rl.policy import CandidatePolicy
from sar_rl.vec_env import VecSarEnv


class RunningStd:
    """Running std of the discounted return, used to scale rewards."""
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x):
        bm, bv, bc = float(np.mean(x)), float(np.var(x)), x.size
        d = bm - self.mean
        tot = self.count + bc
        self.mean += d * bc / tot
        m2 = self.var * self.count + bv * bc + d * d * self.count * bc / tot
        self.var, self.count = m2 / tot, tot

    @property
    def std(self):
        return float(np.sqrt(self.var + 1e-8))


def to_tensors(obs, device):
    return (torch.as_tensor(obs["cand"], dtype=torch.float32, device=device),
            torch.as_tensor(obs["mask"] > 0, dtype=torch.bool, device=device),
            torch.as_tensor(obs["glob"], dtype=torch.float32, device=device))


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    last = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


@torch.no_grad()
def evaluate(model, device, n_eps=24, seed0=10_000):
    """Deterministic policy on fixed random layouts + fixed physics seeds."""
    env = SarExplorationEnv(SimConfig(domain_randomize=True), layout="random")
    rets, insp, dur, comp = [], [], [], []
    for i in range(n_eps):
        obs, _ = env.reset(seed=seed0 + i)
        done, ret = False, 0.0
        while not done:
            c, m, g = to_tensors({k: v[None] for k, v in obs.items()}, device)
            logits, _ = model(c, m, g)
            obs, r, te, tr, info = env.step(int(logits.argmax(-1)))
            ret += r
            done = te or tr
        mt = info["metrics"]
        rets.append(ret)
        insp.append(mt["true_inspected_frac"])
        dur.append(mt["duration_sec"])
        comp.append(float(mt["completed"]))
    return dict(ret=float(np.mean(rets)), insp=float(np.mean(insp)),
                dur=float(np.mean(dur)), comp=float(np.mean(comp)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run1")
    ap.add_argument("--total-steps", type=int, default=1_500_000, help="decisions to train for")
    ap.add_argument("--n-envs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--n-steps", type=int, default=128, help="decisions per env per update")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-start", type=float, default=0.02)
    ap.add_argument("--ent-end", type=float, default=0.002)
    ap.add_argument("--eval-every", type=int, default=20, help="updates between validations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device(a.device)
    out = os.path.join("runs", a.name)
    os.makedirs(out, exist_ok=True)

    model = CandidatePolicy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    ret_rms = RunningStd()
    steps_done, update, best = 0, 0, -1e9
    ckpt_path = os.path.join(out, "ckpt_latest.pt")
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        ret_rms.mean, ret_rms.var, ret_rms.count = ck["rms"]
        steps_done, update, best = ck["steps"], ck["update"], ck["best"]
        print(f"Resumed from {ckpt_path} at {steps_done} decisions")

    venv = VecSarEnv(a.n_envs, {}, seed=a.seed * 7919 + update)
    obs = venv.reset()
    N, T = a.n_envs, a.n_steps
    K, Fd = obs["cand"].shape[1:]
    G = obs["glob"].shape[1]
    ret_acc = np.zeros(N, dtype=np.float64)
    recent = deque(maxlen=100)
    total_updates = max(1, a.total_steps // (N * T))
    log_new = not os.path.exists(os.path.join(out, "log.csv"))
    logf = open(os.path.join(out, "log.csv"), "a", newline="")
    logw = csv.writer(logf)
    if log_new:
        logw.writerow(["update", "decisions", "minutes", "ep_return", "inspected_frac", "duration_s",
                       "stuck", "completed", "entropy", "policy_loss", "value_loss", "val_return",
                       "val_inspected", "val_duration", "val_completed"])
    t_start = time.time()
    print(f"Training {a.name}: {a.n_envs} envs, {total_updates} updates "
          f"(~{total_updates * N * T} decisions)")

    while update < total_updates:
        frac = 1.0 - update / total_updates
        for g in opt.param_groups:
            g["lr"] = a.lr * max(frac, 0.05)
        ent_coef = a.ent_end + (a.ent_start - a.ent_end) * frac

        b_cand = np.zeros((T, N, K, Fd), np.float32)
        b_mask = np.zeros((T, N, K), np.float32)
        b_glob = np.zeros((T, N, G), np.float32)
        b_act = np.zeros((T, N), np.int64)
        b_logp = np.zeros((T, N), np.float32)
        b_val = np.zeros((T, N), np.float32)
        b_rew = np.zeros((T, N), np.float32)
        b_done = np.zeros((T, N), np.float32)

        for t in range(T):
            b_cand[t], b_mask[t], b_glob[t] = obs["cand"], obs["mask"], obs["glob"]
            with torch.no_grad():
                c, m, g = to_tensors(obs, device)
                logits, val = model(c, m, g)
                dist = Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)
            b_act[t] = act.cpu().numpy()
            b_logp[t] = logp.cpu().numpy()
            b_val[t] = val.cpu().numpy()
            obs, rew, done, infos = venv.step(b_act[t])
            ret_acc = ret_acc * a.gamma + rew
            ret_rms.update(ret_acc)
            ret_acc[done] = 0.0
            b_rew[t] = np.clip(rew / ret_rms.std, -10, 10)
            b_done[t] = done.astype(np.float32)
            for inf in infos:
                if inf is not None:
                    recent.append(inf)
        steps_done += N * T

        with torch.no_grad():
            c, m, g = to_tensors(obs, device)
            _, last_val = model(c, m, g)
        adv, ret = compute_gae(b_rew, b_val, b_done, last_val.cpu().numpy(), a.gamma, a.lam)

        B = T * N
        f_cand = torch.as_tensor(b_cand.reshape(B, K, Fd), device=device)
        f_mask = torch.as_tensor(b_mask.reshape(B, K) > 0, device=device)
        f_glob = torch.as_tensor(b_glob.reshape(B, G), device=device)
        f_act = torch.as_tensor(b_act.reshape(B), device=device)
        f_logp = torch.as_tensor(b_logp.reshape(B), device=device)
        f_adv = torch.as_tensor(adv.reshape(B), device=device)
        f_ret = torch.as_tensor(ret.reshape(B), device=device)

        pl_sum = vl_sum = ent_sum = 0.0
        n_mb = 0
        for _ in range(a.epochs):
            perm = torch.randperm(B, device=device)
            for s in range(0, B, a.minibatch):
                idx = perm[s:s + a.minibatch]
                logits, val = model(f_cand[idx], f_mask[idx], f_glob[idx])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(f_act[idx])
                ent = dist.entropy().mean()
                ratio = torch.exp(new_logp - f_logp[idx])
                ad = f_adv[idx]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pg = torch.max(-ad * ratio, -ad * torch.clamp(ratio, 1 - a.clip, 1 + a.clip)).mean()
                vl = 0.5 * F.mse_loss(val, f_ret[idx])
                loss = pg + 0.5 * vl - ent_coef * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                pl_sum += pg.item()
                vl_sum += vl.item()
                ent_sum += ent.item()
                n_mb += 1
        update += 1

        def avg(k):
            return float(np.mean([e[k] for e in recent])) if recent else float("nan")

        val = dict(ret=float("nan"), insp=float("nan"), dur=float("nan"), comp=float("nan"))
        if update % a.eval_every == 0 or update == total_updates:
            val = evaluate(model, device)
            if val["ret"] > best:
                best = val["ret"]
                torch.save({"model": model.state_dict()}, os.path.join(out, "best.pt"))
        mins = (time.time() - t_start) / 60
        row = [update, steps_done, round(mins, 1), round(avg("ep_return"), 2),
               round(avg("true_inspected_frac"), 3), round(avg("duration_sec"), 0),
               round(avg("stuck_events"), 2), round(avg("completed"), 2),
               round(ent_sum / n_mb, 3), round(pl_sum / n_mb, 4), round(vl_sum / n_mb, 4),
               round(val["ret"], 2), round(val["insp"], 3), round(val["dur"], 0), round(val["comp"], 2)]
        logw.writerow(row)
        logf.flush()
        msg = (f"[{update}/{total_updates}] {steps_done} dec | {mins:.1f} min | "
               f"train ret {row[3]} insp {row[4]} dur {row[5]}s stuck {row[6]} | ent {row[8]}")
        if not np.isnan(val["ret"]):
            msg += f" || VAL ret {row[11]} insp {row[12]} dur {row[13]}s (best {best:.2f})"
        print(msg, flush=True)

        if update % 5 == 0 or update == total_updates:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "rms": (ret_rms.mean, ret_rms.var, ret_rms.count),
                        "steps": steps_done, "update": update, "best": best}, ckpt_path)

    torch.save({"model": model.state_dict()}, os.path.join(out, "final.pt"))
    venv.close()
    print("Done. Best validation return:", round(best, 2), "-> runs/%s/best.pt" % a.name)


if __name__ == "__main__":
    main()
