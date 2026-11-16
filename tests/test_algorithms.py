"""Behavior tests for the algorithm code inside the notebooks and a3c_training.py.

Each test runs the notebook's own definition (the code cell that defines the function or
class, executed on its own) on a case small enough to work out by hand. They cover bugs that
were fixed in 1.0.0, so each one fails on the code before it.
"""

from __future__ import annotations

import inspect
import json
import queue
import re
import sys
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.multiprocessing as mp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _namespace() -> dict:
    """The names the notebooks import in their first cell."""
    import copy
    import math
    import random
    import time
    from collections import defaultdict, deque, namedtuple
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union

    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical, Normal
    from torch.nn.utils import parameters_to_vector, vector_to_parameters

    return dict(
        copy=copy, math=math, random=random, time=time, defaultdict=defaultdict, deque=deque,
        namedtuple=namedtuple, Any=Any, Callable=Callable, Dict=Dict, List=List, Optional=Optional,
        Tuple=Tuple, Union=Union, np=np, torch=torch, nn=nn, F=F, optim=optim, mp=mp,
        Categorical=Categorical, Normal=Normal, parameters_to_vector=parameters_to_vector,
        vector_to_parameters=vector_to_parameters, device=torch.device("cpu"),
    )


def _cells_defining(notebook: str, name: str) -> list[str]:
    nb = json.loads((ROOT / notebook).read_text(encoding="utf-8"))
    pattern = re.compile(rf"^(def|class) {name}\b", re.M)
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    found = [s for s in sources if pattern.search(s)]
    assert found, f"{notebook} has no cell defining {name}"
    return found


def _load(notebook: str, *names: str, **extra) -> dict:
    """Execute the cells that define `names` (in order) and return their namespace."""
    ns = _namespace()
    ns.update(extra)
    for name in names:
        exec(compile(_cells_defining(notebook, name)[-1], f"{notebook}:{name}", "exec"), ns)
    return ns


# --- Q-learning: no bootstrap from a terminal state ---------------------------------------

@pytest.mark.parametrize("which", [0, 1], ids=["typed", "compact"])
def test_q_learning_update_does_not_bootstrap_from_a_terminal_state(which: int) -> None:
    source = _cells_defining("02_q_learning.ipynb", "update_q_value")[which]
    ns = _namespace()
    exec(source, ns)
    q = {(0, 0): {"up": 0.0}, (0, 1): {"up": 5.0}}
    ns["update_q_value"](q, (0, 0), "up", 1.0, (0, 1), 0.5, 0.9, ["up"], done=True)
    # Halfway to the target r = 1. Bootstrapping would add 0.9 * 5 to the target.
    assert q[(0, 0)]["up"] == pytest.approx(0.5)
    ns["update_q_value"](q, (0, 0), "up", 1.0, (0, 1), 0.5, 0.9, ["up"], done=False)
    assert q[(0, 0)]["up"] == pytest.approx(0.5 + 0.5 * (1.0 + 0.9 * 5.0 - 0.5))


# --- GAE: the recursion stops where an episode was cut short ------------------------------

def _gae(notebook: str, *args, **kwargs):
    if notebook == "08_a2c.ipynb":
        ns = _load(notebook, "compute_gae_and_returns")
        advantages, returns = ns["compute_gae_and_returns"](*args, standardize_adv=False, **kwargs)
        return advantages, returns
    ns = _load(notebook, "compute_gae")
    return ns["compute_gae"](*args, standardize=False, **kwargs), None


@pytest.mark.parametrize("notebook", ["07_ppo.ipynb", "08_a2c.ipynb", "12_trpo.ipynb"])
def test_gae_does_not_run_across_a_truncated_episode(notebook: str) -> None:
    # Step 0 is the last step of an episode cut by the step limit: s_1 is not terminal, so it
    # bootstraps from V(s_1) = 2, but step 1 belongs to the next episode.
    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.0, 0.0])
    next_values = torch.tensor([2.0, 0.0])
    dones = torch.tensor([0.0, 0.0])
    ends = torch.tensor([1.0, 0.0])
    advantages, returns = _gae(notebook, rewards, values, next_values, dones, 1.0, 1.0, episode_ends=ends)
    torch.testing.assert_close(advantages, torch.tensor([3.0, 1.0]))  # not 3 + 1 for step 0
    if returns is not None:
        torch.testing.assert_close(returns, advantages + values)


@pytest.mark.parametrize("notebook", ["07_ppo.ipynb", "08_a2c.ipynb", "12_trpo.ipynb"])
def test_gae_drops_the_bootstrap_at_a_terminal_state(notebook: str) -> None:
    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.0, 0.0])
    next_values = torch.tensor([2.0, 0.0])
    dones = torch.tensor([1.0, 0.0])
    advantages, _ = _gae(notebook, rewards, values, next_values, dones, 1.0, 1.0, episode_ends=dones)
    torch.testing.assert_close(advantages, torch.tensor([1.0, 1.0]))


# --- TRPO: shapes, and the KL the update reports -------------------------------------------

def _trpo() -> dict:
    return _load("12_trpo.ipynb", "PolicyNetwork", "calculate_kl_divergence", "fisher_vector_product",
                 "conjugate_gradient", "backtracking_line_search", "update_policy_trpo")


def test_trpo_reports_the_kl_and_gain_of_the_update_it_made() -> None:
    ns = _trpo()
    torch.manual_seed(0)
    actor = ns["PolicyNetwork"](2, 4)
    states = torch.rand(64, 2)
    with torch.no_grad():
        dist = actor(states)
        actions = dist.sample()
        old_log_probs = dist.log_prob(actions)
    advantages = torch.randn(64)
    advantages = (advantages - advantages.mean()) / advantages.std()
    stats = ns["update_policy_trpo"](actor, states, actions, advantages, old_log_probs, 0.01, 10, 0.1, 0.8, 10)
    assert stats["status"] == "success"
    assert 0.0 < stats["kl"] <= 0.01 + 1e-6  # within the trust region, and not KL(new || new) = 0
    assert stats["improvement"] > 0.0


def test_trpo_refuses_log_probs_that_would_broadcast() -> None:
    ns = _trpo()
    actor = ns["PolicyNetwork"](2, 4)
    states = torch.rand(8, 2)
    actions = torch.zeros(8, dtype=torch.long)
    with pytest.raises(AssertionError):
        ns["update_policy_trpo"](actor, states, actions, torch.ones(8), torch.zeros(8, 1), 0.01, 10, 0.1, 0.8, 10)


# --- MCTS: the rewards of moves inside the tree are backed up ------------------------------

class _NoActionsEnv:
    def get_possible_actions(self, state=None):
        return []

    def is_terminal(self, state=None):
        return False


def test_mcts_backup_adds_the_move_reward_and_discounts() -> None:
    ns = _load("17_mcts.ipynb", "MCTSNode", "backpropagate", mcts_env=_NoActionsEnv())
    root = ns["MCTSNode"]((0, 0))
    child = ns["MCTSNode"]((0, 1), parent=root, action_that_led_here=3, reward=10.0)
    # The move into the child reached the goal: its rollout is empty and returns 0.
    ns["backpropagate"](child, 0.0, 0.9)
    assert child.visit_count == root.visit_count == 1
    assert child.total_value == pytest.approx(10.0)
    assert root.total_value == pytest.approx(0.9 * 10.0)


# --- A3C: n-step targets and the worker's gradients ----------------------------------------

def test_a3c_notebook_returns_for_a_one_step_rollout() -> None:
    ns = _load("09_a3c.ipynb", "compute_n_step_returns_advantages")
    returns, advantages = ns["compute_n_step_returns_advantages"](
        [1.0], [torch.tensor([[0.5]])], torch.tensor([[2.0]]), [0.0], 0.9)
    torch.testing.assert_close(returns, torch.tensor([1.0 + 0.9 * 2.0]))
    torch.testing.assert_close(advantages, torch.tensor([1.0 + 0.9 * 2.0 - 0.5]))


class _ConstantEnv:
    """The same state and reward at every step, never done: every rollout is identical."""

    def __init__(self, rows: int = 10, cols: int = 10) -> None:
        pass

    def reset(self):
        return torch.tensor([0.5, 0.5])

    def step(self, action: int):
        return torch.tensor([0.5, 0.5]), 1.0, False


class _GradientRecorder:
    """Stands in for the shared optimizer: records the gradients handed to it, changes nothing."""

    def __init__(self, model) -> None:
        self.model = model
        self.steps: list[list] = []

    def zero_grad(self) -> None:
        for p in self.model.parameters():
            p.grad = None

    def step(self) -> None:
        self.steps.append([p.grad.clone() for p in self.model.parameters()])


def _assert_two_equal_updates(recorder: _GradientRecorder) -> None:
    # With one action, a constant environment and parameters that never change, both rollouts
    # have the same loss. Gradients left over in the local model would double the second one.
    assert len(recorder.steps) == 2
    for first, second in zip(*recorder.steps):
        torch.testing.assert_close(second, first)


def test_a3c_script_worker_hands_over_each_rollouts_own_gradient(monkeypatch) -> None:
    sys.path.insert(0, str(ROOT))
    try:
        import a3c_training as a3c
    finally:
        sys.path.remove(str(ROOT))
    monkeypatch.setattr(a3c, "GridEnvironment", _ConstantEnv)
    monkeypatch.setattr(a3c, "n_actions_custom", 1)
    # Clipping would rescale a doubled gradient back to the same norm and hide the difference.
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *args, **kwargs: torch.tensor(0.0))
    global_model = a3c.ActorCriticNetwork(a3c.n_observations_custom, 1)
    recorder = _GradientRecorder(global_model)
    a3c.worker(0, global_model, recorder, mp.Value("i", 0), 2, 10, 10, 1, 0.99, 0.5, 0.01, 200,
               queue.Queue(), threading.Event())
    _assert_two_equal_updates(recorder)


def test_a3c_notebook_worker_hands_over_each_rollouts_own_gradient() -> None:
    ns = _load("09_a3c.ipynb", "ActorCriticNetwork", "compute_n_step_returns_advantages", "worker",
               GridEnvironment=_ConstantEnv, n_observations_custom=2, n_actions_custom=1, seed=0,
               MAX_STEPS_PER_EPISODE_A3C=200)
    global_model = ns["ActorCriticNetwork"](2, 1)
    recorder = _GradientRecorder(global_model)
    ns["worker"](0, global_model, recorder, mp.Value("i", 0), 2, 10, 10, 1, 0.99, 0.5, 0.01, queue.Queue())
    _assert_two_equal_updates(recorder)


# --- HAC: both levels get their experience ------------------------------------------------

def _hac(rows: int = 4, cols: int = 4) -> tuple[dict, object]:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    action_to_goal_map = dict(enumerate(cells))
    ns = _load("16_hac.ipynb", "GridEnvironmentTabular", "GridEnvironmentHAC", "GoalConditionedQNetwork",
               "HindsightReplayBuffer", "soft_update", "HACAgent",
               action_to_goal_map=action_to_goal_map,
               goal_to_action_map={cell: i for i, cell in action_to_goal_map.items()})
    env = ns["GridEnvironmentHAC"](rows=rows, cols=cols)
    kwargs = dict(env=env, num_levels=2, H=3, k_hindsight=4, buffer_size=1000, batch_size=8, gamma=0.98,
                  tau=0.01, actor_lr=1e-3, epsilon_start=1.0, epsilon_end=0.05, device=torch.device("cpu"))
    if "epsilon_decay_steps" in inspect.signature(ns["HACAgent"]).parameters:
        kwargs["epsilon_decay_steps"] = 1000  # Older signature, so old code reaches the checks below
    return ns, ns["HACAgent"](**kwargs)


def test_hac_high_level_run_returns_both_levels_transitions() -> None:
    ns, agent = _hac()
    ns["random"].seed(0)
    agent.env.reset()
    transitions, _, _, _ = agent.run_level(1, agent.env.get_normalized_state(), agent.final_goal_norm)
    assert any(t["level"] == 0 for t in transitions), "the low level's steps are missing"
    # Every high-level transition covers the low-level segment just before it: it starts where
    # the segment started, ends where it ended, and its (hindsight) action is the cell it reached.
    segment = []
    for t in transitions:
        if t["level"] == 0:
            segment.append(t)
            continue
        assert segment
        assert (t["state"] == segment[0]["state"]).all()
        assert (t["next_state"] == segment[-1]["next_state"]).all()
        assert t["action"] == ns["goal_to_action_map"][agent._cell(segment[-1]["next_state"])]
        assert all(len(s["future_goals"]) == len(segment) - j for j, s in enumerate(segment))
        segment = []


def test_hac_hindsight_goals_come_from_later_in_the_attempt() -> None:
    ns, _ = _hac()
    np = ns["np"]
    s0, s1, s2 = (np.array(cell, dtype=np.float32) / 3 for cell in [(0, 0), (0, 1), (0, 2)])
    far_goal = np.array([1.0, 1.0], dtype=np.float32)
    # The attempt went s0 -> s1 -> s2 while aiming at far_goal. Relabeled with s2, the first
    # step has not reached its goal yet and the second one has.
    for state, next_state, reward, done in [(s0, s1, -1.0, 0.0), (s1, s2, 0.0, 1.0)]:
        buffer = ns["HindsightReplayBuffer"](10, hindsight_prob=1.0)
        buffer.push(state, 3, -1.0, next_state, far_goal, False, 0, future_goals=[s2])
        _, _, rewards, _, goals, dones = buffer.sample(1)
        torch.testing.assert_close(goals[0], torch.from_numpy(s2))
        assert rewards.item() == reward and dones.item() == done
