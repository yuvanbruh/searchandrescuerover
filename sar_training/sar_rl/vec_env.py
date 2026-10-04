"""Tiny multiprocessing vector env (no torch dependency)."""
import multiprocessing as mp
import numpy as np
from .sar_env import SarExplorationEnv, SimConfig


def _worker(remote, cfg_kwargs, seed, layout):
    env = SarExplorationEnv(SimConfig(**cfg_kwargs), layout=layout, seed=seed)
    obs, _ = env.reset()
    ep_ret = 0.0
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs, r, term, trunc, info = env.step(data)
            ep_ret += r
            done = term or trunc
            ep_info = None
            if done:
                ep_info = dict(info["metrics"])
                ep_info["ep_return"] = ep_ret
                ep_ret = 0.0
                obs, _ = env.reset()
            remote.send((obs, r, done, ep_info))
        elif cmd == "reset":
            obs, _ = env.reset()
            ep_ret = 0.0
            remote.send(obs)
        elif cmd == "close":
            remote.close()
            break


def _stack(obs_list):
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


class VecSarEnv:
    def __init__(self, n_envs, cfg_kwargs=None, seed=0, layout="random"):
        ctx = mp.get_context("spawn")
        self.n = n_envs
        self.remotes, self.procs = [], []
        for i in range(n_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, cfg_kwargs or {}, seed + 1000 * i, layout),
                            daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return _stack([r.recv() for r in self.remotes])

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", int(a)))
        res = [r.recv() for r in self.remotes]
        obs = _stack([x[0] for x in res])
        return (obs, np.array([x[1] for x in res], dtype=np.float32),
                np.array([x[2] for x in res], dtype=bool), [x[3] for x in res])

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=2)
