<!-- omit in toc -->
# Reinforcement Learning Cheat Sheet ✨

To avoid the need to constantly refer to the original notebooks, here is a cheat sheet summarizing the most important concepts and algorithms in reinforcement learning (RL).

---

<!-- omit in toc -->
## Table of Contents

- [1. Core Concepts](#1-core-concepts)
  - [1.1 Agent-Environment Loop](#11-agent-environment-loop)
  - [1.2 Markov Decision Process (MDP)](#12-markov-decision-process-mdp)
  - [1.3 Value Functions \& Bellman Equations](#13-value-functions--bellman-equations)
  - [1.4 Exploration vs. Exploitation](#14-exploration-vs-exploitation)
- [2. Basic / Tabular Methods](#2-basic--tabular-methods)
  - [2.1 Simple Exploration Bot](#21-simple-exploration-bot)
  - [2.2 Q-Learning](#22-q-learning)
  - [2.3 SARSA](#23-sarsa)
  - [2.4 Expected SARSA](#24-expected-sarsa)
  - [2.5 Dyna-Q](#25-dyna-q)
- [3. Policy Gradient Methods](#3-policy-gradient-methods)
  - [3.1 REINFORCE (Monte Carlo Policy Gradient)](#31-reinforce-monte-carlo-policy-gradient)
  - [3.2 Trust Region Policy Optimization (TRPO)](#32-trust-region-policy-optimization-trpo)
- [4. Actor-Critic Methods](#4-actor-critic-methods)
  - [4.1 Advantage Actor-Critic (A2C)](#41-advantage-actor-critic-a2c)
  - [4.2 Asynchronous Advantage Actor-Critic (A3C)](#42-asynchronous-advantage-actor-critic-a3c)
  - [4.3 Deep Deterministic Policy Gradient (DDPG)](#43-deep-deterministic-policy-gradient-ddpg)
  - [4.4 Soft Actor-Critic (SAC)](#44-soft-actor-critic-sac)
  - [4.5 Proximal Policy Optimization (PPO)](#45-proximal-policy-optimization-ppo)
- [5. Value-Based Deep Methods](#5-value-based-deep-methods)
  - [5.1 Deep Q-Networks (DQN)](#51-deep-q-networks-dqn)
- [6. Multi-Agent RL (MARL)](#6-multi-agent-rl-marl)
  - [6.1 Multi-Agent Deep Deterministic Policy Gradient (MADDPG)](#61-multi-agent-deep-deterministic-policy-gradient-maddpg)
  - [6.2 QMIX (Monotonic Value Function Factorization)](#62-qmix-monotonic-value-function-factorization)
- [7. Hierarchical RL (HRL)](#7-hierarchical-rl-hrl)
  - [7.1 Hierarchical Actor-Critic (HAC)](#71-hierarchical-actor-critic-hac)
- [8. Planning \& Model-Based Methods](#8-planning--model-based-methods)
  - [8.1 Monte Carlo Tree Search (MCTS)](#81-monte-carlo-tree-search-mcts)
  - [8.2 PlaNet (Deep Planning Network)](#82-planet-deep-planning-network)
- [9. Bandits](#9-bandits)
  - [9.1 Multi-Armed Bandits for Dynamic Pricing](#91-multi-armed-bandits-for-dynamic-pricing)
- [Key Insights \& Takeaways](#key-insights--takeaways)

---

## 1. Core Concepts

### 1.1 Agent-Environment Loop
The fundamental interaction cycle in RL:
1.  Agent observes state $s_t$.
2.  Agent selects action $a_t$ based on policy $\pi(a_t|s_t)$.
3.  Environment transitions to next state $s_{t+1}$.
4.  Environment provides reward $r_t$.
5.  Agent updates policy/values based on $(s_t, a_t, r_t, s_{t+1})$.

### 1.2 Markov Decision Process (MDP)
Formal framework for RL problems, defined by $(S, A, P, R, \gamma)$:
-   $S$: Set of states.
-   $A$: Set of actions.
-   $P(s'|s, a)$: Transition probability function.
-   $R(s, a, s')$: Reward function.
-   $\gamma$: Discount factor ($0 \le \gamma \le 1$).

### 1.3 Value Functions & Bellman Equations
-   **State-Value Function ($`V^\pi(s)`$):** Expected return starting from state $s$ and following policy $\pi$.

```math
V^\pi(s) = \mathbb{E}_\pi \left[ G_t | S_t=s \right] = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k r_{t+k+1} | S_t=s \right]
```

-   **Action-Value Function ($`Q^\pi(s, a)`$):** Expected return starting from state $s$, taking action $a$, and following policy $\pi$.

```math
Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t | S_t=s, A_t=a \right] = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k r_{t+k+1} | S_t=s, A_t=a \right]
```

-   **Bellman Expectation Equation for $`V^\pi`$:**

```math
V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s', r} P(s', r | s, a) \left[r + \gamma V^\pi(s')\right] 
```

-   **Bellman Optimality Equation for $`Q^*`$ (used by Q-Learning):**

```math
Q^*(s, a) = \sum_{s', r} P(s', r | s, a) \left[r + \gamma \max_{a'} Q^*(s', a')\right]
```

### 1.4 Exploration vs. Exploitation
-   **Exploration:** Trying new actions to discover better rewards.
-   **Exploitation:** Choosing the action currently known to yield the best expected reward.
-   **$\epsilon$-Greedy:** Common strategy: With probability $\epsilon$, explore (random action); with probability $1-\epsilon$, exploit (greedy action). $\epsilon$ often decays over time.

---

## 2. Basic / Tabular Methods

### 2.1 Simple Exploration Bot
([01_simple_rl.ipynb](01_simple_rl.ipynb))
-   **Core Idea:** Demonstrates the basic agent-environment loop. Agent remembers immediate rewards for state-action pairs and uses a simple epsilon-greedy policy based on average *immediate* rewards. **Does not perform true RL value learning.**
-   **Mathematical Formulation:** No Bellman updates. Policy based on:
```math
\text{AvgR}(s, a) = \frac{\sum \text{rewards observed after } (s, a)}{\text{count of } (s, a)}
```
-   **Pseudocode:**
    1. Initialize memory `mem[s][a] -> [rewards]`
    2. For each episode:
       - Reset env to get `s`
       - For each step:
         - Choose `a` using $\epsilon$-greedy on `AvgR(s, a)`
         - Take action `a`, get `r`, `s'`
         - Store `r` in `mem[s][a]`
         - `s = s'`
-   **Code Snippet:**
    ```python
    # Choosing action based on average immediate reward
    avg_rewards = []
    for a in range(n_actions):
        rewards = memory[state][a]
        avg_rewards.append(np.mean(rewards) if rewards else 0)
    best_action = np.random.choice(np.where(avg_rewards == np.max(avg_rewards))[0])

    # Updating memory
    memory[state][action].append(reward)
    ```
-   **Key Hyperparameters:** `epsilon`, `epsilon_decay`.
-   **Pros:** Simple illustration of interaction loop and memory.
-   **Cons:** Does not learn long-term values, only immediate rewards. Not true RL. Inefficient memory.
-   **Use Cases:** Educational demonstration of basic agent structure.

### 2.2 Q-Learning
([02_q_learning.ipynb](02_q_learning.ipynb))
-   **Core Idea:** Learns the optimal action-value function ($Q^*$) **off-policy** using Temporal Difference (TD) updates.
-   **Mathematical Formulation:** Bellman Optimality update:

```math
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]
```

- $\alpha$: Learning rate
- $\gamma$: Discount factor
- $\max_{a'} Q(s_{t+1}, a')$: Max Q-value in next state (greedy estimate of future value), 0 when $s_{t+1}$ is terminal

-   **Pseudocode:**
    1. Initialize Q-table `Q(s, a)` to zeros.
    2. For each episode:
       - Initialize state `s`.
       - For each step:
         - Choose action `a` from `s` using policy derived from Q (e.g., $\epsilon$-greedy).
         - Take action `a`, observe `r`, `s'`.
         - Update `Q(s, a)` using the Q-learning rule.
         - `s = s'`
-   **Code Snippet:**
    ```python
    # Q-Learning update (a terminal next state has no future value)
    current_q = q_table[state][action]
    max_next_q = 0.0 if done else max(q_table[next_state].values())
    td_target = reward + gamma * max_next_q
    td_error = td_target - current_q
    q_table[state][action] += alpha * td_error
    ```
-   **Key Hyperparameters:** `alpha` (learning rate), `gamma` (discount factor), `epsilon` (exploration rate), `epsilon_decay`.
-   **Pros:** Off-policy (can learn optimal policy while exploring), simple concept, guaranteed convergence under conditions.
-   **Cons:** Tabular form doesn't scale to large state spaces, can suffer from maximization bias (addressed by Double Q-learning).
-   **Common Pitfalls:** Tuning $\alpha$ and $\epsilon$. Ensuring sufficient exploration.
-   **Use Cases:** Small, discrete state/action spaces, foundational understanding.

### 2.3 SARSA
([03_sarsa.ipynb](03_sarsa.ipynb))
-   **Core Idea:** Learns the action-value function ($Q^\pi$) for the policy currently being followed (**on-policy**) using TD updates.
-   **Mathematical Formulation:** Update uses the *next action* chosen by the policy:

```math
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]
```

-   $a_{t+1}$ is the action chosen in state $s_{t+1}$ by the current policy (e.g., $\epsilon$-greedy).
-   **Pseudocode:**
    1. Initialize Q-table `Q(s, a)`.
    2. For each episode:
       - Initialize `s`.
       - Choose `a` from `s` using policy derived from Q (e.g., $\epsilon$-greedy).
       - For each step:
         - Take action `a`, observe `r`, `s'`.
         - Choose *next action* `a'` from `s'` using policy derived from Q.
         - Update `Q(s, a)` using `r`, `s'`, `a'`.
         - `s = s'`, `a = a'`
-   **Code Snippet:**

    ```python
    # SARSA update
    current_q = q_table[state][action]
    next_q = 0.0 if done else q_table[next_state][next_action] # Q-value of the *next* action taken
    td_target = reward + gamma * next_q
    td_error = td_target - current_q
    q_table[state][action] += alpha * td_error
    ```

-   **Key Hyperparameters:** `alpha`, `gamma`, `epsilon`, `epsilon_decay`.
-   **Pros:** On-policy (learns value of the exploration policy), often more stable/conservative in risky environments than Q-learning.
-   **Cons:** Tabular, can be slower to converge to optimal if exploration persists, sensitive to policy changes.
-   **Common Pitfalls:** Ensuring the *next* action `a'` is chosen correctly before the update.
-   **Use Cases:** When evaluating the current policy is important, safer exploration needed.

### 2.4 Expected SARSA
([04_expected_sarsa.ipynb](04_expected_sarsa.ipynb))
-   **Core Idea:** Like SARSA, but updates using the *expected* value over next actions, weighted by policy probabilities, reducing variance. Still **on-policy**.
-   **Mathematical Formulation:**

```math
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \mathbb{E}_{\pi}[Q(s_{t+1}, A')] - Q(s_t, a_t)]
```

```math
\mathbb{E}_{\pi}[Q(s', A')] = \sum_{a'} \pi(a'|s') Q(s', a')
```

For $\epsilon$-greedy:

```math
\mathbb{E}_{\pi}\left[Q(s', A')\right] = (1 - \epsilon) \max_{a''} Q(s', a'') + \epsilon \frac{\sum_{a'} Q(s', a')}{|\mathcal{A}|} 
```

-   **Pseudocode:**
    1. Initialize Q-table `Q(s, a)`.
    2. For each episode:
       - Initialize `s`.
       - For each step:
         - Choose `a` from `s` using policy derived from Q (e.g., $\epsilon$-greedy).
         - Take action `a`, observe `r`, `s'`.
         - Calculate expected Q-value $E[Q(s', A')]$ based on policy $\pi$ in state `s'`.
         - Update `Q(s, a)` using `r` and $E[Q(s', A')]$.
         - `s = s'`
-   **Code Snippet:**

    ```python
    # Expected SARSA update (assuming epsilon-greedy)
    current_q = q_table[state][action]
    if not done: # A terminal next state has no future value
        q_values_next = q_table[next_state]
        max_q_next = max(q_values_next.values())
        num_actions = len(action_space)
        expected_q_next = (1.0 - epsilon) * max_q_next + (epsilon / num_actions) * sum(q_values_next.values())
    else:
        expected_q_next = 0.0
    td_target = reward + gamma * expected_q_next
    td_error = td_target - current_q
    q_table[state][action] += alpha * td_error
    ```

-   **Key Hyperparameters:** `alpha`, `gamma`, `epsilon`, `epsilon_decay`.
-   **Pros:** On-policy, lower variance than SARSA, often more stable, same computational cost as Q-learning per update.
-   **Cons:** Tabular, slightly more complex update calculation than SARSA.
-   **Use Cases:** Where SARSA is applicable but stability/variance is an issue.

### 2.5 Dyna-Q
([05_dyna_q.ipynb](05_dyna_q.ipynb))
-   **Core Idea:** Integrates **model-free learning (Q-learning)** with **model-based planning**. Learns a model of the environment from real experience and uses it to perform extra "planning" updates on the Q-table using simulated experience.
-   **Mathematical Formulation:**
    -   **Direct RL:** Standard Q-learning update on real transition $(s, a, r, s')$.
    -   **Model Learning:** $Model(s, a) \leftarrow (r, s')$ (for deterministic env).
    -   **Planning:** For $k$ steps: Sample $(s_p, a_p)$ from previously experienced pairs. Get $(r_p, s'_p) = Model(s_p, a_p)$. Apply Q-learning update to $Q(s_p, a_p)$ using $(s_p, a_p, r_p, s'_p)$.
-   **Pseudocode:**
    1. Initialize `Q(s, a)` and `Model(s, a)`.
    2. For each episode:
       - Initialize `s`.
       - For each step:
         - Choose `a` using policy based on Q.
         - Take action `a`, observe `r`, `s'`.
         - Update `Q(s, a)` with $(s, a, r, s')$ (Direct RL).
         - Update `Model(s, a)` with $(r, s')$.
         - Repeat `k` times (Planning):
           - Sample previously seen $(s_p, a_p)$.
           - Get $(r_p, s'_p)$ from `Model(s_p, a_p)`.
           - Update `Q(s_p, a_p)` with $(s_p, a_p, r_p, s'_p)$.
         - `s = s'`
-   **Code Snippet:**

    ```python
    # Direct RL Update (same as Q-Learning)
    # ... q_learning_update(q_table, state, action, reward, next_state, ...)
    
    # Model Update
    model[(state, action)] = (reward, next_state)
    if (state, action) not in observed_pairs:
        observed_pairs.append((state, action))
        
    # Planning Step
    for _ in range(planning_steps_k):
        if not observed_pairs: break
        s_p, a_p = random.choice(observed_pairs)
        r_p, s_prime_p = model[(s_p, a_p)]
        q_learning_update(q_table, s_p, a_p, r_p, s_prime_p, ...)
    ```

-   **Key Hyperparameters:** `alpha`, `gamma`, `epsilon`, `k` (number of planning steps).
-   **Pros:** Improves sample efficiency compared to pure Q-learning by reusing experience via the model. Simple integration of learning and planning.
-   **Cons:** Tabular. Effectiveness depends heavily on model accuracy. Assumes deterministic model in simple form.
-   **Common Pitfalls:** Poor model accuracy can lead to suboptimal policy. Choosing `k`.
-   **Use Cases:** Environments where interaction is costly but computation is cheap. Simple planning tasks.

---

## 3. Policy Gradient Methods

### 3.1 REINFORCE (Monte Carlo Policy Gradient)
([06_reinforce.ipynb](06_reinforce.ipynb))
-   **Core Idea:** Directly learns a parameterized policy $\pi(a|s; \theta)$ by increasing the probability of actions that led to high *cumulative* episode returns ($G_t$). **On-policy**, **Monte Carlo**.
-   **Mathematical Formulation:** Updates policy parameters $\theta$ via gradient ascent on $J(\theta) = \mathbb{E}[G_t]$.
    $$\nabla_\theta J(\theta) \approx \sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi(a_t | s_t; \theta) $$
    Loss function (for minimization): $L(\theta) = -\sum_{t=0}^{T-1} G_t \log \pi(a_t | s_t; \theta)$
    -   $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}$: Discounted return from step $t$.
-   **Pseudocode:**
    1. Initialize policy network $\pi(a|s; \theta)$.
    2. For each episode:
       - Generate trajectory $(s_0, a_0, r_1, ...)$ by sampling actions $a_t \sim \pi(a|s_t; \theta)$. Store log probs $\log \pi(a_t|s_t; \theta)$ and rewards $r_{t+1}$.
       - Calculate discounted returns $G_t$ for all steps $t$.
       - Compute loss $L$.
       - Update $\theta$ using gradient descent on $L$.
-   **Code Snippet:**

    ```python
    # Calculate returns (backward loop)
    returns = []
    G = 0.0
    for r in reversed(episode_rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    # Standardize returns (optional but recommended)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Calculate loss
    log_probs_tensor = torch.stack(episode_log_probs)
    loss = -torch.sum(returns * log_probs_tensor)
    
    # Update policy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ```

-   **Key Hyperparameters:** `learning_rate`, `gamma`. Network architecture.
-   **Pros:** Simple policy gradient concept, works with discrete/continuous actions, learns stochastic policies.
-   **Cons:** High variance due to Monte Carlo returns, episodic updates (waits until episode end), on-policy sample inefficiency.
-   **Common Pitfalls:** High variance leading to unstable training, requires careful learning rate tuning.
-   **Use Cases:** Simple benchmarks, conceptual understanding, basis for actor-critic.

### 3.2 Trust Region Policy Optimization (TRPO)
([12_trpo.ipynb](12_trpo.ipynb))
-   **Core Idea:** Improves policy gradient updates by constraining the change in the policy (measured by KL divergence) at each step, ensuring more stable and monotonic improvement. **On-policy**.
-   **Mathematical Formulation:** Solves a constrained optimization problem (approximately):

```math
\max_{\theta} \quad \mathbb{E}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t \right]
\text{s.t.} \quad \mathbb{E}_t [D_{KL}(\pi_{\theta_{old}}(\cdot|s_t) || \pi_{\theta}(\cdot|s_t))] \le \delta
```

Solved using Conjugate Gradient (to find direction $\approx F^{-1}g$) and Line Search (to satisfy constraint). $F$ is the Fisher Information Matrix.
-   **Pseudocode:**
    1. Initialize actor $\pi_\theta$, critic $V_\phi$.
    2. For each iteration:
       - Collect trajectories using $\pi_{\theta_{old}}$. Store states, actions, rewards, log probs.
       - Compute advantages $`\hat{A}_t`$ (using GAE with $`V_\phi`$).
       - Compute policy gradient $g$.
       - Use Conjugate Gradient + Fisher-Vector Products to find step direction $s \approx F^{-1}g$.
       - Perform line search to find step size $\beta \alpha$ satisfying KL constraint $\delta$ and improving surrogate objective.
       - Update actor: $\theta_{new} \leftarrow \theta_{old} + \beta \alpha s$.
       - Update critic $V_\phi$ using collected data.
-   **Code Snippet:** (Focus on conceptual update call)

    ```python
    # Conceptual TRPO update call
    policy_gradient = calculate_policy_gradient(...)
    step_direction = conjugate_gradient(fisher_vector_product_func, policy_gradient, ...)
    initial_step_size = calculate_initial_step_size(step_direction, policy_gradient, max_kl, ...)
    final_update, success = backtracking_line_search(actor, ..., step_direction, initial_step_size, max_kl, ...)
    if success:
        apply_update(actor, final_update)
    update_critic(...)
    ```

-   **Key Hyperparameters:** `delta` (KL constraint), `gamma`, `lambda` (GAE), CG iterations, CG damping, line search parameters.
-   **Pros:** Provides theoretical monotonic improvement guarantee (under approximations), very stable updates.
-   **Cons:** Complex implementation (FVP, CG, line search), computationally expensive per update, on-policy.
-   **Common Pitfalls:** Implementing FVP and CG correctly, tuning trust region $\delta$.
-   **Use Cases:** Continuous control, situations requiring high stability, benchmark for simpler algorithms like PPO.

---

## 4. Actor-Critic Methods

### 4.1 Advantage Actor-Critic (A2C)
([08_a2c.ipynb](08_a2c.ipynb))
-   **Core Idea:** A **synchronous**, simpler version of A3C. Uses an actor (policy) and a critic (value function) trained on batches of experience collected by the actor. Reduces variance compared to REINFORCE by using advantage estimates. **On-policy**.
-   **Mathematical Formulation:**
    -   Actor Loss (minimize): $L_{actor} = - \mathbb{E}_t [ \log \pi(a_t | s_t; \theta) \hat{A}_t^{\text{detached}} + c_e H(\pi) ]$
    -   Critic Loss (minimize): $L_{critic} = \mathbb{E}_t [ (R_t - V(s_t; \phi))^2 ]$
    -   $\hat{A}_t = R_t - V(s_t; \phi)$: Advantage estimate (often using n-step returns or GAE for $R_t$).
-   **Pseudocode:**
    1. Initialize shared actor $`\pi_\theta`$ and critic $`V_\phi`$.
    2. Loop for iterations:
       - Collect batch of N steps of experience $`(s_t, a_t, r_{t+1}, s_{t+1}, d_t)`$ using $`\pi_\theta`$.
       - Compute n-step returns $R_t$ and advantages $`\hat{A}_t`$ using $`V_\phi`$.
       - Compute actor loss (policy gradient + entropy) and critic loss (MSE).
       - Compute gradients for actor and critic based on the batch.
       - Apply synchronous gradient update to $\theta$ and $\phi$.
-   **Code Snippet:**

    ```python
    # Calculate Advantage and Returns (e.g., using GAE)
    advantages, returns_to_go = compute_gae_and_returns(...)
    
    # Evaluate current policy and value
    policy_dist = actor(states)
    log_probs = policy_dist.log_prob(actions)
    entropy = policy_dist.entropy().mean()
    values_pred = critic(states).squeeze()
    
    # Losses
    policy_loss = -(log_probs * advantages.detach()).mean() - entropy_coeff * entropy
    value_loss = F.mse_loss(values_pred, returns_to_go.detach())
    
    # Optimize Actor
    actor_optimizer.zero_grad()
    policy_loss.backward()
    actor_optimizer.step()
    
    # Optimize Critic
    critic_optimizer.zero_grad()
    (value_loss_coeff * value_loss).backward()
    critic_optimizer.step()
    ```

-   **Key Hyperparameters:** `learning_rates` (actor/critic), `gamma`, `lambda` (GAE), `n_steps` (rollout length), `value_loss_coeff`, `entropy_coeff`.
-   **Pros:** More stable than REINFORCE, simpler than A3C/TRPO/PPO, good baseline, utilizes GPUs well.
-   **Cons:** On-policy (sample inefficient), updates can still have variance, performance sometimes lower than PPO.
-   **Common Pitfalls:** Balancing actor/critic learning rates, choosing `n_steps`.
-   **Use Cases:** Discrete/continuous control benchmarks, simpler alternative to A3C/PPO.

### 4.2 Asynchronous Advantage Actor-Critic (A3C)
([09_a3c.ipynb](09_a3c.ipynb) & [a3c_training.py](a3c_training.py))
-   **Core Idea:** Uses multiple parallel workers, each with a local copy of the actor-critic network and an environment instance. Workers compute gradients locally based on n-step returns and asynchronously update a shared global network. **On-policy**.
-   **Mathematical Formulation:** Same loss function as A2C (per worker), but updates are applied asynchronously to global parameters $\theta_{global}, \phi_{global}$ using gradients computed from local parameters $\theta', \phi'$.
-   **Pseudocode (Worker):**
    1. Initialize local network, sync with global.
    2. Loop:
       - Reset local gradients. Sync with global.
       - Collect n-steps of experience using local policy.
       - Calculate n-step returns $R_t$ and advantages $\hat{A}_t$.
       - Compute gradients for actor and critic losses based on the n-step experience.
       - Apply gradients asynchronously to the global network using a shared optimizer.
       - If episode done, reset environment.
-   **Code Snippet:** (Conceptual - see `a3c_training.py`)

    ```python
    # Inside worker loop
    local_model.load_state_dict(global_model.state_dict())
    # ... collect n-steps data ...
    returns, advantages = compute_n_step_returns_advantages(...)
    policy_loss = -(log_probs * advantages.detach()).mean() - entropy_coeff * entropy
    value_loss = F.mse_loss(values_pred, returns.detach())
    total_loss = policy_loss + value_loss_coeff * value_loss
    
    global_optimizer.zero_grad()
    local_model.zero_grad() # backward() adds to .grad, and the local model is reused for every rollout
    total_loss.backward() # Calculates grad on local model
    # Transfer gradients to global model
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        if global_param.grad is not None: global_param.grad.data.zero_() # Optional safety zero
        if local_param.grad is not None:
             global_param.grad = local_param.grad.clone()
    global_optimizer.step() # Updates global model
    ```

-   **Key Hyperparameters:** `num_workers`, `n_steps`, learning rates, `gamma`, coefficients $c_v, c_e$. Optimizer details (e.g., shared Adam).
-   **Pros:** No replay buffer needed, decorrelates data via parallelism, efficient on multi-core CPUs.
-   **Cons:** Complex implementation (multiprocessing, shared memory, async updates), potential for stale gradients, often less GPU-efficient than A2C.
-   **Common Pitfalls:** Race conditions with shared optimizer/gradients, worker synchronization.
-   **Use Cases:** Historically significant for Atari/continuous control, CPU-based parallel training.

### 4.3 Deep Deterministic Policy Gradient (DDPG)
([10_ddpg.ipynb](10_ddpg.ipynb))
-   **Core Idea:** An **off-policy** actor-critic algorithm primarily for **continuous action spaces**. Learns a deterministic policy (actor) alongside a Q-function (critic). Uses ideas from DQN (replay buffer, target networks) for stability.
-   **Mathematical Formulation:**
    -   Critic Update (minimize loss): $L(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} [ (y - Q(s, a; \phi))^2 ]$ where $y = r + \gamma Q'(s', \mu'(s'; \theta'); \phi')$.
    -   Actor Update (maximize objective via gradient ascent, often minimize negative): $L(\theta) = - \mathbb{E}_{s \sim \mathcal{D}} [ Q(s, \mu(s; \theta); \phi) ]$.
    -   $\mu, Q$: Main networks; $\mu', Q'$: Target networks.
-   **Pseudocode:**
    1. Initialize actor $\mu_\theta$, critic $Q_\phi$, target networks $`\mu'_{\theta'}, Q'_{\phi'}`$, replay buffer $`\mathcal{D}`$.
    2. For each step:
       - Select action $a = \mu(s; \theta) + \text{Noise}$.
       - Execute $a$, get $r, s'$. Store $(s, a, r, s')$ in $\mathcal{D}$.
       - Sample mini-batch from $\mathcal{D}$.
       - Update critic $Q_\phi$ using TD error derived from target networks.
       - Update actor $\mu_\theta$ using gradient from critic's output $Q(s, \mu(s))$.
       - Soft-update target networks: $\theta' \leftarrow \tau \theta + (1-\tau)\theta'$, $\phi' \leftarrow \tau \phi + (1-\tau)\phi'$.
-   **Code Snippet:**

    ```python
    # Critic Update
    # done_batch marks terminal states only. An episode cut by a time limit (gymnasium's
    # `truncated`) still has a future, so it is stored as not done and bootstraps.
    with torch.no_grad():
        next_actions = target_actor(next_state_batch)
        target_q = target_critic(next_state_batch, next_actions)
        y = reward_batch + gamma * (1 - done_batch) * target_q
    current_q = critic(state_batch, action_batch)
    critic_loss = F.mse_loss(current_q, y)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
    # Actor Update
    actor_actions = actor(state_batch)
    q_for_actor = critic(state_batch, actor_actions) # No detach! Grad flows from critic
    actor_loss = -q_for_actor.mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    # Soft Updates
    soft_update(target_critic, critic, tau)
    soft_update(target_actor, actor, tau)
    ```

-   **Key Hyperparameters:** `buffer_size`, `batch_size`, `gamma`, `tau` (soft update rate), actor/critic learning rates, exploration noise parameters.
-   **Pros:** Off-policy sample efficiency, handles continuous actions directly.
-   **Cons:** Sensitive to hyperparameters, can suffer from Q-value overestimation, exploration can be tricky.
-   **Common Pitfalls:** Learning rates, noise scale/decay, target update rate $\tau$.
-   **Use Cases:** Continuous control (robotics, physics simulation).

### 4.4 Soft Actor-Critic (SAC)
([11_sac.ipynb](11_sac.ipynb))
-   **Core Idea:** An **off-policy** actor-critic algorithm for **continuous actions** based on the **maximum entropy** framework. Learns a stochastic policy that maximizes both expected return and policy entropy, leading to improved exploration and robustness.
-   **Mathematical Formulation:** Objective includes entropy term: $J(\pi) = \mathbb{E}_{\tau \sim \pi} [\sum \gamma^t (R_t + \alpha H(\pi(\cdot|s_t)))]$. Uses twin Q-critics, target critics, and often auto-tunes entropy coefficient $\alpha$.
    -   Critic Update (minimize loss for $Q_1, Q_2$): $L(\phi_i) = \mathbb{E} [ (Q_i(s,a) - y)^2 ]$ where $y = r + \gamma (1-d) [\min_{j=1,2} Q'_j(s', a') - \alpha \log \pi(a'|s')]$, $a' \sim \pi(\cdot|s')$.
    -   Actor Update (minimize loss): $`L(\theta) = \mathbb{E}_{s, a \sim \pi} \left[ \alpha \log \pi(a|s) - \min_{j=1,2} Q_j(s, a) \right]`$.
    -   Alpha Update (minimize loss): $L(\log \alpha) = \mathbb{E}_{a \sim \pi} [ -\log \alpha (\log \pi(a|s) + \bar{H}) ]$ (where $\bar{H}$ is target entropy).
-   **Pseudocode:**
    1. Initialize actor $`\pi_\theta`$, twin critics $`Q_{\phi_1}, Q_{\phi_2}`$, target critics $`Q'_{\phi'_1}, Q'_{\phi'_2}`$, replay buffer $`\mathcal{D}`$, $`\log \alpha`$.
    2. For each step:
       - Select action $a \sim \pi(\cdot|s; \theta)$ (sampling).
       - Execute $a$, get $r, s'$. Store $(s, a, r, s')$ in $\mathcal{D}$.
       - Sample mini-batch from $\mathcal{D}$.
       - Update critics $Q_{\phi_1}, Q_{\phi_2}$ using soft TD target (min of target Q's minus scaled log prob).
       - Update actor $\pi_\theta$ using gradient based on min Q and log prob.
       - Update $\alpha$ (if auto-tuning) based on policy entropy.
       - Soft-update target critics.
-   **Code Snippet:**

    ```python
    # Critic Target Calculation
    # done_batch marks terminal states only. An episode cut by a time limit (gymnasium's
    # `truncated`) still has a future, so it is stored as not done and bootstraps.
    with torch.no_grad():
        next_action, next_log_prob = actor(next_state_batch)
        q1_target_next, q2_target_next = target_critic(next_state_batch, next_action)
        q_target_next = torch.min(q1_target_next, q2_target_next)
        alpha = torch.exp(log_alpha).detach()
        soft_target = q_target_next - alpha * next_log_prob
        y = reward_batch + gamma * (1.0 - done_batch) * soft_target
    # ... Critic Update (MSE loss) ...
    
    # Actor Update
    pi_action, pi_log_prob = actor(state_batch)
    q1_pi, q2_pi = critic(state_batch, pi_action) # Grads enabled for critic here
    min_q_pi = torch.min(q1_pi, q2_pi)
    actor_loss = (alpha * pi_log_prob - min_q_pi).mean()
    # ... Actor Optimizer Step ...
    
    # Alpha Update
    alpha_loss = -(log_alpha * (pi_log_prob.detach() + target_entropy)).mean()
    # ... Alpha Optimizer Step ...
    
    # Soft Updates ...
    ```

-   **Key Hyperparameters:** `buffer_size`, `batch_size`, `gamma`, `tau`, learning rates (actor, critic, alpha), initial `alpha`, `target_entropy` (if auto-tuning).
-   **Pros:** State-of-the-art sample efficiency and performance on continuous control, robust, good exploration.
-   **Cons:** More complex than DDPG/PPO, requires careful implementation (especially squashing correction).
-   **Common Pitfalls:** Correct log prob calculation (tanh squashing correction), alpha tuning stability, target entropy choice.
-   **Use Cases:** Continuous control (robotics, benchmarks), situations needing robust exploration.

### 4.5 Proximal Policy Optimization (PPO)
([07_ppo.ipynb](07_ppo.ipynb))
-   **Core Idea:** An **on-policy** actor-critic method that simplifies TRPO's constrained update using a **clipped surrogate objective**. Allows multiple epochs of updates on collected data for better sample efficiency.
-   **Mathematical Formulation:**
    -   Ratio: $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
    -   Clipped Objective (minimize negative): $L^{CLIP}(\theta) = -\mathbb{E}_t [ \min( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t ) ]$
    -   Often includes value loss $L^{VF}$ and entropy bonus $S$: $L = L^{CLIP} + c_1 L^{VF} - c_2 S$.
-   **Pseudocode:**
    1. Initialize actor $\pi_\theta$, critic $V_\phi$.
    2. For each iteration:
       - Collect batch of trajectories using $\pi_{\theta_{old}}$. Store states, actions, rewards, dones, old log probs.
       - Compute advantages $\hat{A}_t$ (GAE) and returns $R_t$.
       - For K epochs:
         - For each mini-batch in collected data:
           - Calculate policy ratio $r_t(\theta)$.
           - Compute clipped surrogate loss $L^{CLIP}$.
           - Compute value loss $L^{VF}$.
           - Compute entropy bonus $S$.
           - Compute combined loss $L$.
           - Update $\theta$ and $\phi$ using gradient descent on $L$.
-   **Code Snippet:**

    ```python
    # Inside PPO update loop (for one epoch/minibatch)
    policy_dist = actor(states)
    log_probs_new = policy_dist.log_prob(actions)
    entropy = policy_dist.entropy().mean()
    values_pred = critic(states).squeeze()
    
    # Calculate ratio
    ratio = torch.exp(log_probs_new - log_probs_old)
    
    # Calculate policy loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - ppo_clip_epsilon, 1.0 + ppo_clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean() - entropy_coeff * entropy
    
    # Calculate value loss. returns_to_go = raw GAE + V(s): standardize only the advantages
    # the policy sees, never the critic's target.
    value_loss = F.mse_loss(values_pred, returns_to_go)
    
    # Update networks (typically combined loss or separate updates)
    # ... optimizer steps ...
    ```

-   **Key Hyperparameters:** `clip_epsilon`, `gamma`, `lambda` (GAE), learning rates, `num_epochs`, `mini_batch_size`, `value_loss_coeff`, `entropy_coeff`.
-   **Pros:** Simpler than TRPO, stable updates, good performance (often SOTA or near-SOTA), relatively sample efficient for an on-policy method.
-   **Cons:** Still on-policy (less efficient than off-policy), performance sensitive to implementation details and hyperparameters.
-   **Common Pitfalls:** Advantage/observation normalization, learning rate schedule, choice of $\epsilon$.
-   **Use Cases:** Default choice for many discrete/continuous control tasks, RLHF for LLMs.

---

## 5. Value-Based Deep Methods

### 5.1 Deep Q-Networks (DQN)
([13_dqn.ipynb](13_dqn.ipynb))
-   **Core Idea:** Combines Q-learning with a deep neural network to approximate $Q(s, a; \theta)$. Uses **Experience Replay** and **Target Networks** for stability. **Off-policy**.
-   **Mathematical Formulation:** Minimizes TD error using target network $Q'$:

```math
L(\theta) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} [ (y - Q(s, a; \theta))^2 ] 

y = r + \gamma (1-d) \max_{a'} Q'(s', a'; \theta^{-})
```

-   $`\mathcal{D}`$: Replay buffer. $`\theta^{-}`$: Target network parameters.
-   **Pseudocode:**
    1. Initialize Q-network $Q_\theta$, target network $Q'_{\theta^-}$, replay buffer $\mathcal{D}$.
    2. For each episode:
       - For each step:
         - Choose action `a` using $\epsilon$-greedy on $Q_\theta$.
         - Execute `a`, get `r`, `s'`. Store $(s, a, r, s', done)$ in $\mathcal{D}$.
         - Sample mini-batch from $\mathcal{D}$.
         - Compute target `y` using $Q'_{\theta^-}$.
         - Update $Q_\theta$ by minimizing loss $L(\theta)$.
         - Periodically update target network: $\theta^- \leftarrow \theta$.
         - `s = s'`
-   **Code Snippet:**

    ```python
    # DQN Optimization Step
    non_final_mask = torch.tensor(...) # Mask for non-terminal next states
    non_final_next_states = torch.cat(...)
    state_batch, action_batch, reward_batch, done_batch = ... # From replay buffer
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ```

-   **Key Hyperparameters:** `buffer_size`, `batch_size`, `gamma`, `tau` or `target_update_freq`, `learning_rate`, `epsilon` schedule.
-   **Pros:** Handles high-dimensional states (e.g., pixels), off-policy sample efficiency, stable due to replay/target nets.
-   **Cons:** Primarily for discrete actions, can overestimate Q-values, sensitive to hyperparameters.
-   **Common Pitfalls:** Target network updates, buffer management, hyperparameter tuning.
-   **Use Cases:** Atari games from pixels, tasks with discrete actions and large state spaces.

---

## 6. Multi-Agent RL (MARL)

### 6.1 Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
([14_maddpg.ipynb](14_maddpg.ipynb))
-   **Core Idea:** Extends DDPG to multi-agent settings using the "centralized training, decentralized execution" paradigm. Each agent has an actor and a *centralized* critic that observes joint states/observations and actions. **Off-policy**.
-   **Mathematical Formulation:**
    -   Centralized Critic $Q_i(x, a_1, ..., a_N; \phi_i)$ for agent $i$. $x = (o_1, ..., o_N)$.
    -   Critic Update: Minimize $L(\phi_i) = \mathbb{E} [ (y_i - Q_i(x, \mathbf{a}))^2 ]$ where $y_i = r_i + \gamma Q'_i(x', \mu'_1(o'_1), ..., \mu'_N(o'_N))$.
    -   Actor Update: Minimize $L(\theta_i) = - \mathbb{E}_{x, \mathbf{a} \sim \mathcal{D}} [ Q_i(x, a_1, ..., \mu_i(o_i), ..., a_N) ]$ (using main critic $Q_i$; the other agents' actions $a_j$ come from the buffer). The notebook's actions are discrete, so it takes the exact expectation over agent $i$'s actions instead: $\sum_a \pi_i(a|o_i) Q_i(x, a, a_{-i})$.
-   **Pseudocode:**
    1. Initialize actors $\mu_i$, critics $Q_i$, targets $\mu'_i, Q'_i$, replay buffer $\mathcal{D}$.
    2. For each step:
       - Each agent $i$ chooses $a_i = \mu_i(o_i) + \text{Noise}$.
       - Execute joint action $\mathbf{a}$, get $r=(r_1,...), o'=(o'_1,...)$. Store $(o, \mathbf{a}, r, o')$ in $\mathcal{D}$.
       - Sample mini-batch from $\mathcal{D}$.
       - For each agent $i$: Update critic $Q_i$ and actor $\mu_i$.
       - Soft-update all target networks.
-   **Code Snippet:** (Conceptual - Update involves joint info)

    ```python
    # Critic Update (Agent i)
    with torch.no_grad():
        target_actions_next = [target_actors[j](next_obs_batch[:, j]) for j in range(num_agents)]
        target_q_next = target_critics[i](joint_next_obs_batch, torch.cat(target_actions_next, dim=1))
        y_i = rewards_batch[:, i] + gamma * (1 - dones_batch[:, i]) * target_q_next
    current_q_i = critics[i](joint_obs_batch, joint_actions_batch) # Actions from buffer
    critic_loss_i = F.mse_loss(current_q_i, y_i)
    # ... optimize critic i ...
    
    # Actor Update (Agent i): the other agents' actions come from the replay buffer, as in the
    # paper; only agent i's action is recomputed by its actor, so only it carries a gradient
    actions = list(actions_batch.unbind(dim=1)) # Per-agent actions from the buffer
    actions[i] = actors[i](obs_batch[:, i])
    actor_loss_i = -critics[i](joint_obs_batch, torch.cat(actions, dim=1)).mean()
    # ... optimize actor i ...
    ```

-   **Key Hyperparameters:** Similar to DDPG, but potentially per-agent. Buffer size, batch size, $\gamma, \tau$, learning rates, noise.
-   **Pros:** Addresses non-stationarity in MARL, decentralized execution, handles mixed cooperative/competitive settings.
-   **Cons:** Centralized critic scales poorly with many agents, credit assignment can be hard in cooperative settings.
-   **Use Cases:** Multi-robot coordination, predator-prey, cooperative navigation.

### 6.2 QMIX (Monotonic Value Function Factorization)
([15_qmix.ipynb](15_qmix.ipynb))
-   **Core Idea:** A value-based MARL algorithm for **cooperative** tasks. Learns individual agent Q-functions $Q_i$ and mixes them **monotonically** using a mixing network conditioned on the global state to produce $Q_{tot}$. **Off-policy**, **centralized training, decentralized execution**.
-   **Mathematical Formulation:**
    -   $Q_{tot}(x, \mathbf{a}) = f_{mix}(Q_1(o_1, a_1), ..., Q_N(o_N, a_N); x)$
    -   Constraint: $\frac{\partial Q_{tot}}{\partial Q_i} \ge 0$ (enforced by non-negative mixer weights, often via hypernetworks).
    -   Loss: Minimize TD error on $Q_{tot}$: $L = \mathbb{E} [ (y - Q_{tot}(x, \mathbf{a}))^2 ]$ where $y = r + \gamma Q'_{tot}(x', \mathbf{a}')$ with $a'_i = \arg\max_a Q'_i(o'_i, a)$.
-   **Pseudocode:**
    1. Initialize agent networks $`Q_i`$, target networks $`Q'_i`$, mixer $`f_{mix}`$, target mixer $`f'_{mix}`$, replay buffer $`\mathcal{D}`$.
    2. For each step:
       - Each agent $i$ chooses $a_i$ using $\epsilon$-greedy on $Q_i(o_i)$.
       - Execute $\mathbf{a}$, get $r$, $o'$, $x'$. Store $(o, \mathbf{a}, r, o', x, x', done)$ in $\mathcal{D}$.
       - Sample mini-batch.
       - Calculate target $`y`$ using target networks $`Q'_i`$ and target mixer $`f'_{mix}`$.
       - Calculate current $`Q_{tot}`$ using main networks $`Q_i`$ and main mixer $`f_{mix}`$.
       - Compute loss $L$.
       - Update all $Q_i$ and $f_{mix}$ parameters via gradient descent on $L$.
       - Soft-update target networks.
-   **Code Snippet:**

    ```python
    # Calculate Target Q_tot'
    with torch.no_grad():
        # ... Get max Q'_i for each agent i in next state ...
        target_agent_qs = torch.cat(...) # Shape (batch, num_agents)
        q_tot_target = target_mixer(target_agent_qs, next_global_state_batch)
        y = reward_batch + gamma * (1 - done_batch) * q_tot_target

    # Calculate Current Q_tot
    # ... Get Q_i for the action *taken* by each agent i in current state ...
    current_agent_qs = torch.cat(...) # Shape (batch, num_agents)
    q_tot_current = mixer(current_agent_qs, global_state_batch)
    
    # Loss and Optimize
    loss = F.mse_loss(q_tot_current, y)
    optimizer.zero_grad()
    loss.backward() # Gradients flow back to all agent nets and mixer
    optimizer.step()
    # ... Soft update targets ...
    ```

-   **Key Hyperparameters:** Like DQN, plus mixing network architecture, hypernetwork details.
-   **Pros:** Good for cooperative tasks, satisfies the IGM principle (Individual-Global-Max: the agents' individually greedy actions form the greedy joint action), scales better in action space than joint Q-learning.
-   **Cons:** Limited representational power due to monotonicity, requires global state for mixer.
-   **Use Cases:** Cooperative MARL (e.g., SMAC benchmark), resource allocation.

---

## 7. Hierarchical RL (HRL)

### 7.1 Hierarchical Actor-Critic (HAC)
([16_hac.ipynb](16_hac.ipynb))
-   **Core Idea:** Learns policies at multiple levels of abstraction. High levels set subgoals (as "actions") for lower levels, which execute primitive actions to achieve them within a time limit $H$. Uses **intrinsic rewards** and **hindsight** for learning subgoals. **Off-policy**.
-   **Mathematical Formulation (Conceptual):**
    -   Level $i$: Policy $\pi_i(a_i | s, g_i)$, Q-function $Q_i(s, a_i, g_i)$. $a_i$ is subgoal for level $i-1$.
    -   Low Level (0): Learns using intrinsic reward $r_{int}$ (success/fail to reach $g_0$).
    -   High Level (1): Learns using environment reward $R = \sum r_{env}$.
    -   Hindsight: Relabel transitions with *achieved* states as goals, granting artificial success.
-   **Pseudocode (2-Level):**
    1. Initialize networks $Q_0, Q'_0, Q_1, Q'_1$, buffers $D_0, D_1$.
    2. For episode:
       - `s = env.reset()`
       - While overall goal `G` not reached:
         - High level chooses subgoal `g_0 = select_action(level=1, state=s, goal=G)`.
         - `transitions = []`, `total_env_reward = 0`
         - `s_start = s`
         - For `h` from 1 to `H`:
           - Low level chooses primitive action `a = select_action(level=0, state=s, goal=g_0)`.
           - Take `a`, get `r_env`, `s_next`, `env_done`.
           - `total_env_reward += r_env`
           - `r_int = get_intrinsic_reward(s_next, g_0)`
           - Store low-level tuple `(s, a, r_int, s_next, g_0, env_done or test_goal(s_next,g_0))` in `transitions`.
           - `s = s_next`
           - If `test_goal(s, g_0)` or `env_done`: break inner loop.
         - `s_end = s`
         - Store high-level tuple `(s_start, s_end, total_env_reward, s_end, G, env_done)` in `transitions`, with the state actually reached `s_end` as the action (hindsight action).
         - Give each low-level transition the states reached later in the same attempt, for hindsight goals.
         - Add `transitions` to buffers with hindsight relabeling.
         - Update $Q_0, Q_1$ from buffers. Update targets.
         - If `env_done`: break outer loop.
-   **Code Snippet:** (Focus on hindsight and intrinsic reward)

    ```python
    # Inside low-level execution loop
    # ... execute action a, get next_state_norm, env_done ...
    intrinsic_reward = -1.0 # Default failure
    subgoal_achieved = self._test_goal(next_state_norm, goal_norm)
    if subgoal_achieved:
        intrinsic_reward = 0.0 # Success reward
        
    # Store original transition; once the attempt is over, it gets future_goals, the states
    # the attempt reached from this step on
    buffer.push(..., reward=intrinsic_reward, goal=goal_norm, done=env_done or subgoal_achieved,
                future_goals=later_states, env_done=env_done, level=0)
    
    # Hindsight goals are applied in buffer.sample(): with probability k / (k + 1) the goal is
    # replaced by one of future_goals and the reward and done flag are recomputed for it
    ```

-   **Key Hyperparameters:** Number of levels, time limit `H`, learning rates, `gamma`, `tau`, `epsilon` schedule, buffer sizes, hindsight probability `p`.
-   **Pros:** Can solve long-horizon/sparse reward tasks, structured exploration, potential for skill reuse.
-   **Cons:** Very complex implementation, sensitive to goal definition, time limits, and hyperparameters, potential for suboptimal subgoal setting. The notebook simplifies the paper: two levels, discrete subgoals, Q-learning instead of DDPG, and no subgoal-testing transitions.
-   **Common Pitfalls:** Hindsight logic, intrinsic reward definition, goal feasibility, tuning `H`.
-   **Use Cases:** Complex robotics tasks, long-horizon planning, navigation.

---

## 8. Planning & Model-Based Methods

### 8.1 Monte Carlo Tree Search (MCTS)
([17_mcts.ipynb](17_mcts.ipynb))
-   **Core Idea:** An **online planning** algorithm that builds a search tree using simulated trajectories (rollouts) from the current state. Uses statistics (visit counts, values) and UCT to balance exploration/exploitation within the search tree. **Requires a simulator/model**.
-   **Mathematical Formulation:**
    -   Tree Nodes store state $s$, visit count $N(s)$, total value $W(s)$.
    -   Edges store action $a$, count $N(s, a)$, value $W(s, a)$.
    -   Selection policy (UCT): Choose action $a$ maximizing $Q(s, a) + C \sqrt{\frac{\ln N(s)}{N(s, a)}}$.
-   **Pseudocode (Single MCTS step for action selection):**
    1. Initialize tree with root node = current state $s_{root}$.
    2. Repeat for `N` simulations:
       - `node = root_node`
       - **Selection:** While `node` is fully expanded and not terminal, `node = select_best_child_uct(node)`.
       - **Expansion:** If `node` not fully expanded and not terminal, expand one child `node = expand_node(node)`.
       - **Simulation:** Run rollout from `node`'s state using default policy, get reward `R`.
       - **Backpropagation:** Update `N` and `W` for nodes/edges from `node` back up to root using `R`. With a reward on every move (as in the notebook's grid), each level up adds the reward of the move into the node and discounts the rest: $G \leftarrow r + \gamma G$, then $W \leftarrow W + G$.
    3. Choose best action from root based on visit counts (or values).
-   **Code Snippet:** (UCT Selection)

    ```python
    # Inside select_best_child_uct
    best_score = -float('inf')
    best_child = None
    for action, child in node.children.items():
        if child.visit_count == 0:
            uct_score = float('inf')
        else:
            exploit = child.total_value / child.visit_count
            explore = exploration_constant * math.sqrt(math.log(node.visit_count) / child.visit_count)
            uct_score = exploit + explore
        # ... update best_child ...
    return best_child
    ```

-   **Key Hyperparameters:** `num_simulations` (budget per step), `exploration_constant C`, `rollout_depth`, `gamma` (for rollouts).
-   **Pros:** Anytime algorithm, handles large state/action spaces, no explicit value function needed for search, asymmetric tree growth.
-   **Cons:** Requires a simulator/model, computationally intensive per step, rollout policy quality affects performance.
-   **Common Pitfalls:** Tuning `C`, efficient implementation of steps, quality of rollouts.
-   **Use Cases:** Game playing (Go, Chess), planning problems with simulators.

### 8.2 PlaNet (Deep Planning Network)
([18_planet.ipynb](18_planet.ipynb))
-   **Core Idea:** A **model-based** RL agent that learns a **latent dynamics model** (often an RSSM) directly from experience (potentially high-dimensional observations). It then performs **planning directly in the latent space** using algorithms like CEM to select actions. **Off-policy**.
-   **Mathematical Formulation (Conceptual):**
    -   Learns models: $p(s_{t+1}|s_t, a_t)$ (transition), $p(r_t|s_t)$ (reward), possibly $p(o_t|s_t)$ (observation/reconstruction), $q(s_t|...)$ (encoder).
    -   Model Loss: Maximize data likelihood (often via ELBO, including reconstruction, reward prediction, KL regularization terms). Simplified: MSE on next state & reward.
    -   Planning (CEM): Optimize $`\mathbb{E} \left[ \sum_{k=t}^{t+H-1} \gamma^{k-t} \hat{r}_k \right]`$ over action sequences $`a_t..a_{t+H-1}`$ using the learned latent model $`(\hat{p}, \hat{r})`$.
-   **Pseudocode:**
    1. Initialize latent dynamics model, replay buffer $\mathcal{D}$ (stores sequences).
    2. Loop:
       - **Interact:** Observe $s_t$. Plan action $a_t$ using CEM in latent space with current model. Execute $a_t$, get $r_t, s_{t+1}$. Store $(s_t, a_t, r_t, s_{t+1})$ in $\mathcal{D}$.
       - **Train Model:** Sample sequences from $\mathcal{D}$. Update model parameters to minimize prediction/reconstruction losses.
-   **Code Snippet:** (CEM Planning Call)

    ```python
    # Inside main loop
    # state = current latent state representation
    action = cem_planner(
        dynamics_model, state, 
        horizon=PLANNING_HORIZON, 
        num_candidates=CEM_CANDIDATES, 
        num_elites=CEM_ELITES, 
        num_iterations=CEM_ITERATIONS, 
        gamma=CEM_GAMMA, ...)
    # Execute action in real env...
    # Train model...
    ```

-   **The notebook's version:** an MLP that predicts Pendulum's next state vector and reward, trained on single steps, with CEM planning on top. There is no latent RSSM and no image encoder.
-   **Key Hyperparameters:** Model architecture (latent size, hidden dims), model learning rate, buffer size, sequence length, planning horizon `H`, CEM parameters (`J`, `M`, iterations).
-   **Pros:** Very sample efficient (especially from images), learns compact world representation, effective planning.
-   **Cons:** Complex model training, planning can be computationally expensive, model inaccuracies can lead to poor plans (compounding errors).
-   **Common Pitfalls:** Model convergence, tuning planning horizon vs. model accuracy, computational cost of planning.
-   **Use Cases:** Control from pixels, sample-constrained robotics tasks, model-based benchmarks.

---

## 9. Bandits

### 9.1 Multi-Armed Bandits for Dynamic Pricing
([19_bandits_dynamic_pricing.ipynb](19_bandits_dynamic_pricing.ipynb))
-   **Core Idea:** RL with a single state. Each price is an arm, each visitor is one pull, and the reward is the revenue of that sale. The agent has to keep testing prices it is unsure about while charging the one that looks best, which is the exploration-exploitation trade-off with nothing else around it.
-   **Mathematical Formulation:** Regret after $T$ visitors, against always charging the best price:

```math
\text{Regret}(T) = T \mu^* - \sum_{t=1}^{T} \mu_{a_t}, \qquad \mu_a = p_a \, P(\text{buy} \mid p_a)
```

-   **UCB1:** charge $\arg\max_a \left[ \hat{\mu}_a + c \sqrt{2 \ln t / n_a} \right]$, with $c$ the largest possible reward (the highest price).
-   **Thompson Sampling:** keep a $\text{Beta}(\alpha_a, \beta_a)$ belief about each price's purchase probability, draw $\theta_a$ from each, charge $\arg\max_a p_a \theta_a$, and add the sale or no-sale to that price's belief.
-   **Code Snippet:**

    ```python
    # Thompson Sampling for pricing
    theta = rng.beta(alpha, beta)          # one guess of every purchase probability
    arm = int(np.argmax(prices * theta))   # the price with the best guessed revenue
    bought, revenue = env.visit(arm)       # one visitor: did they buy at that price?
    alpha[arm] += bought                   # update the belief about that price only
    beta[arm] += 1 - bought
    ```

-   **Key Hyperparameters:** $\epsilon$ for $\epsilon$-greedy, $c$ for UCB1, the prior for Thompson Sampling.
-   **Pros:** Simple, sample-efficient, and well understood (regret bounds).
-   **Cons:** No state: today's price does not change tomorrow's situation. Pricing with inventory or a selling season needs a full MDP.
-   **Use Cases:** Price and offer testing, ad and recommendation selection, A/B tests that adapt while they run.

---

## Key Insights & Takeaways

> **Pro Tip**: Standardizing advantages (in policy gradient/actor-critic methods) or returns (in REINFORCE) often significantly stabilizes training. This involves subtracting the mean and dividing by the standard deviation within each batch or episode. Standardize only the copy the policy sees: a critic trained on standardized targets learns a quantity with no meaning.

> **Algorithm Combinations**: Many state-of-the-art algorithms combine multiple ideas. PPO and A2C use actor-critic with policy gradients. SAC combines actor-critic with maximum entropy RL. PlaNet integrates model-based learning with CEM planning.

> **Implementation Details Matter**: Small implementation details can dramatically affect performance. Pay attention to network initialization, normalization techniques, and update frequencies.

> **Environment Design**: The choice of state representation, action space, and reward function often matters more than the specific algorithm used.

> **Sample Efficiency vs Stability Trade-off**: Off-policy methods (SAC, DDPG) are typically more sample efficient, while on-policy methods (PPO, A2C) are often more stable and easier to tune.

---

👉 This cheat sheet provides a high-level overview. For detailed implementation and nuances, refer to the specific notebooks in the repository and the original research papers. Good luck learning!