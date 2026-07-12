'''
Process pool for stepping independent envs in parallel.

The Oceananigans trainers are CPU-bound in the Python env step (SwarmSwIM tick +
scipy field interpolators, ~3 ms/env-step) while the policy forward pass is
negligible, so stepping N envs in N worker processes parallelizes ~95% of the
rollout wall-clock. This pool is the multi-agent-friendly counterpart of
gymnasium's AsyncVectorEnv: it passes each env's step/reset tuples through
UNCHANGED (whatever their shapes), so it serves both the raw single-agent env
(scalar reward/term/trunc) and the flattened PettingZoo-parallel multi-agent env
((N,) rewards, per-agent term/trunc arrays, info["global_state"]) — which the
gym vector API cannot represent. No autoreset: the trainer decides when an env
resets (via reset_at), exactly like the previous in-process loop.

AsyncEnvPool and SyncEnvPool expose the same API, so trainers switch between
them with a flag:
    reset(seeds)              -> [(obs, info), ...] for every env
    reset_at(i, seed=None)    -> (obs, info) for env i
    step(actions)             -> [(obs, r, term, trunc, info), ...] for every env
    step_where(indices, acts) -> results for a subset (used by the parallel eval,
                                 where episodes finish at different times)
    attr(i, name)             -> getattr(env_i, name) (spaces, target values, ...)
    close()

Workers use the "spawn" start context (fork is unsafe on macOS and with
threaded parents) and are built from picklable zero-arg factories — use
functools.partial over a module-level function with plain-dict config (see
src/envs/oceananigans_factory.py), NOT a closure over a trainer dataclass:
unpickling the latter would import the trainer module (and torch) in every
worker, costing hundreds of MB per process.
'''
import multiprocessing as mp


def _worker(pipe, env_fn):
    '''Worker loop: owns ONE env, executes (cmd, data) requests off the pipe.'''
    env = env_fn()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == "step":
                pipe.send(env.step(data))
            elif cmd == "reset":
                pipe.send(env.reset(seed=data))
            elif cmd == "attr":
                pipe.send(getattr(env, data))
            elif cmd == "close":
                pipe.send(None)
                break
            else:
                raise RuntimeError(f"unknown command {cmd!r}")
    finally:
        env.close()
        pipe.close()


class AsyncEnvPool:
    '''One spawn-context process per env; requests are sent to every involved
    worker first and only then awaited, so the envs actually step concurrently.'''

    def __init__(self, env_fns):
        ctx = mp.get_context("spawn")
        self.num_envs = len(env_fns)
        self._pipes = []
        self._procs = []
        self._closed = False
        for fn in env_fns:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, fn), daemon=True)
            proc.start()
            child.close()
            self._pipes.append(parent)
            self._procs.append(proc)

    def reset_where(self, indices, seeds=None):
        indices = list(indices)
        seeds = list(seeds) if seeds is not None else [None] * len(indices)
        for i, s in zip(indices, seeds):
            self._pipes[i].send(("reset", s))
        return [self._pipes[i].recv() for i in indices]

    def reset(self, seeds=None):
        return self.reset_where(range(self.num_envs), seeds)

    def reset_at(self, index, seed=None):
        return self.reset_where([index], [seed])[0]

    def step_where(self, indices, actions):
        indices = list(indices)
        for i, a in zip(indices, actions):
            self._pipes[i].send(("step", a))
        return [self._pipes[i].recv() for i in indices]

    def step(self, actions):
        return self.step_where(range(self.num_envs), actions)

    def attr(self, index, name):
        self._pipes[index].send(("attr", name))
        return self._pipes[index].recv()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for pipe in self._pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for pipe in self._pipes:
            try:
                pipe.recv()
            except (EOFError, OSError):
                pass
            pipe.close()
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


class SyncEnvPool:
    '''In-process fallback with the identical API — the previous sequential
    behavior, for debugging and single-core machines (--no-async-envs).'''

    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]

    def reset_where(self, indices, seeds=None):
        indices = list(indices)
        seeds = list(seeds) if seeds is not None else [None] * len(indices)
        return [self.envs[i].reset(seed=s) for i, s in zip(indices, seeds)]

    def reset(self, seeds=None):
        return self.reset_where(range(self.num_envs), seeds)

    def reset_at(self, index, seed=None):
        return self.reset_where([index], [seed])[0]

    def step_where(self, indices, actions):
        return [self.envs[i].step(a) for i, a in zip(list(indices), actions)]

    def step(self, actions):
        return self.step_where(range(self.num_envs), actions)

    def attr(self, index, name):
        return getattr(self.envs[index], name)

    def close(self):
        for env in self.envs:
            env.close()
