# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-16

The first tagged release. Every notebook was re-run from the top after the fixes below, so each output, plot and analysis comes from the code next to it.

### Added

- **Notebook 19, multi-armed bandits for dynamic pricing** ([#1](https://github.com/FareedKhan-dev/all-rl-algorithms/issues/1)): ε-greedy, UCB1 and Thompson Sampling choosing prices for a simulated shop, compared by the revenue they give up over 20 runs. Thompson Sampling gives up the least. Thanks to @MR-HOLO for the question and @MenokoOG for suggesting Thompson Sampling.
- **`tests/test_notebooks.py`**: every notebook ran from the top on its current code with no error output, its math renders, and it imports only declared dependencies.
- **`tests/test_algorithms.py`**: the notebooks' own functions on small cases worked out by hand, covering the fixes below. Every test fails on the 0.x code.
- **CI** on Python 3.10, 3.12 and 3.13, plus a CPU job that runs the algorithm tests.
- `a3c_training.py` ends with a summary of the last 50 episodes and a greedy test of the trained policy.
- A bandits section in the cheat sheet.

### Fixed

- **Q-learning (02):** episodes could start on a goal and the update bootstrapped from terminal states (the old outputs show Q((3,3), up) = 19.23 on a terminal state). Cliff Walking reused the 4x4 world's terminal states, charged nothing for the cliff and paid +10 every time the agent stepped back onto the goal; it now has the textbook rules, and Q-learning learns the 13-move path along the edge.
- **PPO (07) and A2C (08):** V(s<sub>t+1</sub>) came from the next stored state, which after a step-limit cut is the next episode's reset state, and GAE ran on across those cuts. PPO also stepped the environment an extra time per rollout, and its critic was trained on standardized advantages: its value loss sat at 1.0 for the whole run and now falls to 0.01. A2C logged episodes that crossed an iteration as shorter than they were.
- **TRPO (12):** old log-probabilities of shape (N, 1) against new ones of shape (N,) broadcast the ratio to (N, N), so the line search tested a meaningless objective and rejected most updates, and the logged KL compared the new policy with itself. All 150 updates are now accepted, each inside the 0.01 trust region.
- **A3C (09 and `a3c_training.py`):** the local model's gradients were never zeroed, so every update carried the sum of all earlier ones. Episodes cut by the step limit bootstrapped from 0 (script) or from the next episode's start state (notebook), and one-step rollouts crashed the notebook's worker. With correct gradients the old 5-step returns no longer find a goal 18 moves away, so the returns are now 20 steps and training runs 100,000 steps; the greedy policy takes the 18-step shortest path.
- **DDPG (10) and SAC (11):** stored `terminated or truncated` as the done flag, so the critic treated Pendulum's 200-step time limit as a terminal state with value 0.
- **MCTS (17):** the backup ignored the rewards of the moves inside the tree, including the +10 for stepping into the goal. Episodes now take 13 to 16 moves on average instead of 26 to 59.
- **HAC (16):** the low level's transitions were dropped, so it never trained: rewards fell from -17 to -85 as exploration decayed and every test episode timed out. With hindsight goals, hindsight actions and one update per transition, it now reaches the goal by the shortest path in every test episode.
- **MADDPG (14) and QMIX (15):** exploration schedules far longer than training (epsilon ended near 0.7), MADDPG's "greedy" test sampled from the policies, and QMIX's hard-update mode copied the targets after every update. MADDPG's actor now uses the other agents' actions from the replay buffer and the exact expectation over its own four actions.
- **PlaNet (18):** logged only the second half of each episode's reward, which is where the old rewards close to 0 came from.
- **Simple RL (01):** `plt.cm.get_cmap` is gone in Matplotlib 3.11, so the plot cell failed; the goal reward was read by stepping the live environment.
- **SARSA (03):** greedy ties always went to 'up'; they are now broken at random.
- **97 LaTeX expressions** in eight notebooks were broken by single backslashes in the JSON (`\theta` rendered as a tab and "heta").
- **Analysis text:** every analysis cell was checked against the new outputs and rewritten where it no longer matched. Among the old errors: DQN described a CartPole run it does not contain, REINFORCE and DQN gave the wrong state scaling, and DDPG read its actor loss as the Q-value when it is minus the Q-value.
- **Cheat sheet:** every notebook link pointed to a file that does not exist (`2_q_learning.ipynb` for `02_q_learning.ipynb`), and the A3C, Q-learning, SARSA, MADDPG and HAC snippets repeated the bugs above.

### Changed

- `pyproject.toml` accepts Python 3.10 and newer (it required 3.13), declares the notebooks' dependencies, and has a dev group with Jupyter and pytest; `uv.lock` is regenerated.
- README: installation with `uv sync` or pip, notebook 19, corrected algorithm descriptions, and how to contribute a notebook change.
- DDPG, SAC and PlaNet no longer suggest installing the retired `gym` package, and SAC no longer calls the deprecated `torch.set_default_tensor_type`.

### Removed

- `main.py`, the stub left by `uv init`.

[1.0.0]: https://github.com/FareedKhan-dev/all-rl-algorithms/releases/tag/v1.0.0
