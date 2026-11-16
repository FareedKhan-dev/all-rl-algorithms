# -*- coding: utf-8 -*-
"""
Runnable Python script for Asynchronous Advantage Actor-Critic (A3C) 
training on a custom Grid Environment. (Corrected Gradient Handling)
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple
from itertools import count
from typing import List, Tuple, Dict, Optional, Callable
import time
import queue # Standard queue, not needed if using mp.Queue

# Import PyTorch and multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp # Use torch multiprocessing

# Set up device (Workers likely run on CPU, global model might be GPU but requires care)
# For simplicity, let's assume CPU for this example to avoid GPU sharing complexities.
# Global model needs to be accessible by workers, share_memory() often works best with CPU tensors.
device = torch.device("cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility in the main process
# Note: workers will need their own seeding if full reproducibility is needed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# -----------------------------------------------------------------------------
# Custom Grid World Environment (Identical)
# -----------------------------------------------------------------------------
class GridEnvironment:
    def __init__(self, rows: int = 10, cols: int = 10) -> None:
        self.rows: int = rows
        self.cols: int = cols
        self.start_state: Tuple[int, int] = (0, 0)
        self.goal_state: Tuple[int, int] = (rows - 1, cols - 1)
        self.state: Tuple[int, int] = self.start_state
        self.state_dim: int = 2
        self.action_dim: int = 4
        # Action map: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_map: Dict[int, Tuple[int, int]] = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def reset(self) -> torch.Tensor:
        self.state = self.start_state
        return self._get_state_tensor(self.state)

    def _get_state_tensor(self, state_tuple: Tuple[int, int]) -> torch.Tensor:
        # Normalize state to be between 0 and 1
        norm_row = state_tuple[0] / (self.rows - 1) if self.rows > 1 else 0.0
        norm_col = state_tuple[1] / (self.cols - 1) if self.cols > 1 else 0.0
        normalized_state: List[float] = [norm_row, norm_col]
        # Ensure tensor is created on the correct device (CPU for workers)
        # Force CPU here as workers explicitly use CPU
        return torch.tensor(normalized_state, dtype=torch.float32, device=torch.device("cpu"))

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        if self.state == self.goal_state:
            # Should not happen if done is checked correctly, but good practice
            return self._get_state_tensor(self.state), 0.0, True

        if action not in self.action_map:
            raise ValueError(f"Invalid action: {action}. Action must be in {list(self.action_map.keys())}")

        dr, dc = self.action_map[action]
        current_row, current_col = self.state
        next_row, next_col = current_row + dr, current_col + dc

        reward: float = -0.1 # Small penalty for each step

        # Check boundaries
        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
            # Stay in the same state if boundary hit
            next_row, next_col = current_row, current_col
            reward = -1.0 # Larger penalty for hitting a wall

        self.state = (next_row, next_col)
        next_state_tensor: torch.Tensor = self._get_state_tensor(self.state)
        done: bool = (self.state == self.goal_state)

        if done:
            reward = 10.0 # Large reward for reaching the goal

        return next_state_tensor, reward, done

    def get_action_space_size(self) -> int:
        return self.action_dim

    def get_state_dimension(self) -> int:
        return self.state_dim

# Instantiate once to get dims for network setup
temp_env = GridEnvironment(rows=10, cols=10)
n_actions_custom = temp_env.get_action_space_size()
n_observations_custom = temp_env.get_state_dimension()
del temp_env # No longer needed

# -----------------------------------------------------------------------------
# A3C Actor-Critic Network
# -----------------------------------------------------------------------------
class ActorCriticNetwork(nn.Module):
    """ Combined Actor-Critic network for A3C """
    def __init__(self, n_observations: int, n_actions: int):
        super(ActorCriticNetwork, self).__init__()
        # Shared layers
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)

        # Actor head (outputs action logits)
        self.actor_head = nn.Linear(128, n_actions)

        # Critic head (outputs state value)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """
        Forward pass, returns action distribution and state value.

        Parameters:
        - x (torch.Tensor): Input state tensor. Should be on the correct device.

        Returns:
        - Tuple[Categorical, torch.Tensor]:
            - Action distribution (Categorical).
            - State value estimate (Tensor).
        """
        # Ensure input is a FloatTensor
        if not isinstance(x, torch.Tensor):
             # Assume input needs conversion, place on model's device implicitly? No, safer to manage explicitly.
             # Let's assume x is already a tensor on the correct device (CPU for workers)
             raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        elif x.dtype != torch.float32:
             x = x.to(dtype=torch.float32)

        # Add batch dimension if missing (e.g., single state input)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Shared layers
        x = F.relu(self.layer1(x))
        shared_features = F.relu(self.layer2(x))

        # Actor head
        action_logits = self.actor_head(shared_features)
        # Ensure logits are on the same device before creating Categorical
        action_dist = Categorical(logits=action_logits.to(x.device))

        # Critic head
        state_value = self.critic_head(shared_features)

        # If input had no batch dim, remove it from output value
        if x.shape[0] == 1 and state_value.dim() > 0: # Check state_value dim > 0 before squeeze
            state_value = state_value.squeeze(0)

        return action_dist, state_value

# -----------------------------------------------------------------------------
# N-Step Return and Advantage Calculation
# -----------------------------------------------------------------------------
def compute_n_step_returns_advantages(rewards: List[float],
                                      values: List[torch.Tensor],
                                      bootstrap_value: torch.Tensor,
                                      dones: List[float],
                                      gamma: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes n-step returns (targets for critic) and advantages for actor.

    Parameters:
    - rewards (List[float]): List of rewards from the n-step rollout.
    - values (List[torch.Tensor]): List of value estimates V(s_t) for the rollout steps (as tensors, with grad history).
    - bootstrap_value (torch.Tensor): Value estimate V(s_{t+n}) for bootstrapping. Should be detached.
    - dones (List[float]): List of done flags (0.0 for not done, 1.0 for done).
    - gamma (float): Discount factor.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]:
        - n_step_returns: Target values for the critic (on CPU).
        - advantages: Advantage estimates for the actor (on CPU).
    """
    n_steps = len(rewards)
    # Detach values *here* for calculation, they retain grad_fn upstream for loss
    values_tensor = torch.cat([v.detach() for v in values]).squeeze().to(torch.device("cpu"))

    # Detach bootstrap value as well and ensure it's on CPU
    R = bootstrap_value.detach().to(torch.device("cpu"))

    # Initialize tensors on CPU (as workers run on CPU)
    returns = torch.zeros(n_steps, dtype=torch.float32, device=torch.device("cpu"))
    advantages = torch.zeros(n_steps, dtype=torch.float32, device=torch.device("cpu"))

    # Calculate backwards from the last step
    for t in reversed(range(n_steps)):
        # R is the discounted return from step t onwards
        # If done[t] is 1.0, the state t+1 was terminal, so its value is 0.
        R = rewards[t] + gamma * R * (1.0 - dones[t])
        returns[t] = R

        # Advantage A_t = n_step_return(R_t) - V(s_t) (using detached value here)
        # Ensure values_tensor has the right shape if n_steps=1
        value_t = values_tensor if values_tensor.dim() == 0 else values_tensor[t]
        advantages[t] = R - value_t

    # Standardization of advantages is often helpful but omitted here for simplicity
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


# -----------------------------------------------------------------------------
# Worker Process Logic (Corrected)
# -----------------------------------------------------------------------------
def worker(worker_id: int,
           global_model: ActorCriticNetwork,
           global_optimizer: optim.Optimizer,
           global_counter: mp.Value, # Shared counter for total steps
           max_global_steps: int,
           env_rows: int,
           env_cols: int,
           n_steps: int, # N-step rollout length
           gamma: float,
           value_loss_coeff: float,
           entropy_coeff: float,
           max_episode_steps: int, # Max steps per episode in env
           result_queue: mp.Queue, # Queue for sending results (reward, length, errors, progress)
           stop_event: mp.Event) -> None: # Event to signal workers to stop
    """
    Function executed by each A3C worker process. (Corrected gradient handling)
    """
    worker_device = torch.device("cpu") # Workers run on CPU
    print(f"Worker {worker_id} started on CPU.")

    # Worker-specific seeding
    torch.manual_seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

    # Create local environment and model
    local_env = GridEnvironment(rows=env_rows, cols=env_cols)
    local_model = ActorCriticNetwork(n_observations_custom, n_actions_custom).to(worker_device)

    # Initial sync with global model
    local_model.load_state_dict(global_model.state_dict())
    # Keep model in train() mode as default unless BatchNorm/Dropout are used
    local_model.train()

    state = local_env.reset() # Initial state tensor (on CPU)

    episode_reward = 0.0
    episode_length = 0
    episode_count = 0 # Track episodes completed by this worker

    try:
        while global_counter.value < max_global_steps and not stop_event.is_set():
            # --- Sync local model with global model before each rollout ---
            local_model.load_state_dict(global_model.state_dict())
            local_model.train() # Ensure it's in train mode

            # Storage for n-step rollout data
            log_probs_list: List[torch.Tensor] = []
            values_list: List[torch.Tensor] = [] # Store tensors with grad_fn
            rewards_list: List[float] = []
            dones_list: List[float] = []
            entropies_list: List[torch.Tensor] = [] # Store tensors with grad_fn

            # --- Rollout Phase (collect n steps or until episode ends) ---
            for step_idx in range(n_steps):
                # Ensure state tensor is on the correct device (CPU)
                state_tensor = state.to(worker_device)

                # >>>>> Perform forward pass *without* torch.no_grad() <<<<<
                action_dist, value_pred = local_model(state_tensor)

                action = action_dist.sample() # Sample action
                log_prob = action_dist.log_prob(action) # Log probability - has grad_fn
                entropy = action_dist.entropy() # Entropy - has grad_fn

                # Interact with environment
                next_state, reward, done = local_env.step(action.item())

                # Store transition data (log_prob, value_pred, entropy now have grad history)
                log_probs_list.append(log_prob)
                values_list.append(value_pred) # value_pred has grad_fn
                rewards_list.append(reward)
                dones_list.append(float(done))
                entropies_list.append(entropy) # entropy has grad_fn

                episode_reward += reward
                episode_length += 1

                # Update state. Keep the state the rollout stopped in for the bootstrap below:
                # when the episode ends, `state` is replaced by the reset state of the next one.
                state = next_state
                rollout_end_state = next_state

                # Check for termination conditions (a terminal state, or the step limit)
                episode_done = done or (episode_length >= max_episode_steps)

                # Increment global step counter safely
                current_global_step = 0 # Initialize
                with global_counter.get_lock():
                    global_counter.value += 1
                    current_global_step = global_counter.value

                # Optional: Send periodic progress update
                if current_global_step > 0 and current_global_step % 5000 == 0:
                    result_queue.put(("progress", worker_id, current_global_step))

                # If episode ended, reset env and log results
                if episode_done:
                    episode_count += 1
                    result_queue.put(("episode_end", worker_id, episode_reward, episode_length))
                    state = local_env.reset()
                    episode_reward = 0.0
                    episode_length = 0
                    break # End inner rollout loop

                # Check global step limit inside inner loop too
                if current_global_step >= max_global_steps:
                     stop_event.set() # Signal stop if limit reached
                     break

            # --- Prepare for Gradient Calculation ---
            if not rewards_list: # Skip if rollout was empty
                continue

            # Calculate bootstrap value V(s_{t+n}): 0 after a terminal state, otherwise the critic's
            # estimate for the state the rollout stopped in. That includes an episode cut by the
            # step limit: it did not reach a terminal state, so its value is not 0.
            R = torch.tensor([0.0], dtype=torch.float32, device=worker_device)
            if not dones_list[-1]:
                # Use torch.no_grad() here, as this is a target value
                with torch.no_grad():
                    _, R = local_model(rollout_end_state.to(worker_device))

            # Compute n-step returns and advantages (function expects detached bootstrap value R)
            # values_list contains tensors with grad_fn, compute_n_step will detach them internally for advantage calc
            returns_tensor, advantages_tensor = compute_n_step_returns_advantages(
                rewards_list, values_list, R, dones_list, gamma
            )

            # Move targets to the correct device (they are created on CPU)
            returns_tensor = returns_tensor.to(worker_device)
            advantages_tensor = advantages_tensor.to(worker_device)

            # --- Calculate Losses ---
            # Convert lists of tensors (which have grad_fn) to single tensors
            log_probs_tensor = torch.cat(log_probs_list)
            values_pred_tensor = torch.cat(values_list) # These are the V(s_t) predictions with grad_fn
            entropies_tensor = torch.cat(entropies_list)

            # Squeeze tensors ONLY if they have extra dims (e.g., shape [N, 1] -> [N])
            # Ensure the dimensions match for calculation.
            # Common shapes: log_probs [N], values [N], returns [N], advantages [N], entropies [N]
            log_probs_tensor = log_probs_tensor.squeeze()
            values_pred_tensor = values_pred_tensor.squeeze()
            returns_tensor = returns_tensor.squeeze()
            advantages_tensor = advantages_tensor.squeeze()
            entropies_tensor = entropies_tensor.squeeze()

            # Ensure tensors are at least 1D after squeeze (handles n_steps=1 case)
            if log_probs_tensor.dim() == 0: log_probs_tensor = log_probs_tensor.unsqueeze(0)
            if values_pred_tensor.dim() == 0: values_pred_tensor = values_pred_tensor.unsqueeze(0)
            if returns_tensor.dim() == 0: returns_tensor = returns_tensor.unsqueeze(0)
            if advantages_tensor.dim() == 0: advantages_tensor = advantages_tensor.unsqueeze(0)
            if entropies_tensor.dim() == 0: entropies_tensor = entropies_tensor.unsqueeze(0)

            # Final shape check before loss (helpful for debugging)
            # print(f"Worker {worker_id} Shapes: logp={log_probs_tensor.shape}, vals={values_pred_tensor.shape}, ret={returns_tensor.shape}, adv={advantages_tensor.shape}, ent={entropies_tensor.shape}")
            # assert log_probs_tensor.shape == values_pred_tensor.shape == returns_tensor.shape == advantages_tensor.shape == entropies_tensor.shape, "Shape mismatch before loss calculation!"


            # Actor loss (Policy Gradient) - detach advantages
            policy_loss = -(log_probs_tensor * advantages_tensor.detach()).mean()

            # Critic loss (Value Function) - MSE between *predicted* values and n-step returns
            value_loss = F.mse_loss(values_pred_tensor, returns_tensor.detach())

            # Entropy bonus - mean entropy over the batch
            entropy_loss = -entropies_tensor.mean()

            # Combined loss
            total_loss = policy_loss + value_loss_coeff * value_loss + entropy_coeff * entropy_loss

            # --- Compute Gradients and Update Global Network ---
            # Ensure model is in training mode for backward pass
            local_model.train()

            # Zero the gradients of both models. backward() adds to .grad and the local model is
            # reused for every rollout, so without this each update would also carry the
            # gradients of all earlier rollouts.
            global_optimizer.zero_grad()
            local_model.zero_grad()

            # Calculate gradients for the local model based on the total loss
            total_loss.backward()

            # Optional: Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=0.5)

            # Transfer gradients from local model to global model
            for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                if global_param.grad is not None:
                    # This shouldn't happen if zero_grad was called correctly, but indicates potential issues
                    # print(f"Warning: Worker {worker_id} - Global grad already exists for {name}")
                    # For safety, let's zero it before copying, although ideally optimizer.zero_grad() handles this.
                    global_param.grad = None

                if local_param.grad is not None:
                    # Clone grad from local to global param's .grad attribute
                    global_param.grad = local_param.grad.clone().to(global_param.device) # Ensure grad is on global model's device

            # Apply the gradients using the shared optimizer (updates global model)
            global_optimizer.step()

            # Check if max global steps reached after update
            if global_counter.value >= max_global_steps and not stop_event.is_set():
                print(f"Worker {worker_id} reached max global steps after update.")
                stop_event.set() # Signal others to stop


    except Exception as e:
        print(f"!!! Worker {worker_id} encountered an error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(("error", worker_id, str(e)))
        stop_event.set() # Signal others to stop

    finally:
        print(f"Worker {worker_id} finished. Total episodes: {episode_count}, Final Global steps: {global_counter.value}")
        result_queue.put(("finished", worker_id))

# -----------------------------------------------------------------------------
# Main Execution Block
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # --- Hyperparameter Setup ---
    GAMMA_A3C = 0.99             # Discount factor
    LR_A3C = 1e-4                # Learning rate
    N_STEPS = 20                 # Steps per update (n-step rollout length). The goal is 18 moves
                                 # from the start; with 5-step returns its +10 barely reaches the
                                 # early states, and the agent settles for staying off the walls
    VALUE_LOSS_COEFF_A3C = 0.5   # Coefficient for value loss term
    ENTROPY_COEFF_A3C = 0.01     # Coefficient for entropy bonus term

    NUM_WORKERS = mp.cpu_count() # Use number of available CPU cores
    # NUM_WORKERS = 4 # Or set manually
    MAX_GLOBAL_STEPS_A3C = 100000 # Total training steps across all workers
    MAX_STEPS_PER_EPISODE_A3C = 200 # Max steps per episode before reset (timeout)

    ENV_ROWS = 10
    ENV_COLS = 10

    # Set multiprocessing start method (important for some OS like macOS/Windows)
    # 'spawn' is generally safer than 'fork' with CUDA/threading
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"Could not set start method to 'spawn': {e}. Using default.")


    # --- Initialization ---
    # Initialize Global Network (on CPU, as share_memory works best)
    global_model_a3c = ActorCriticNetwork(n_observations_custom, n_actions_custom).to(device)
    # Crucial step: Ensure model parameters are shared across processes
    global_model_a3c.share_memory()
    print(f"Global model initialized on {device} and set to shared memory.")

    # Initialize Optimizer (acts on the shared global model's parameters)
    # Adam is common, but RMSprop was used in the original A3C paper
    # global_optimizer_a3c = optim.RMSprop(global_model_a3c.parameters(), lr=LR_A3C, alpha=0.99, eps=1e-5)
    global_optimizer_a3c = optim.Adam(global_model_a3c.parameters(), lr=LR_A3C)
    print(f"Global optimizer initialized: {type(global_optimizer_a3c).__name__}")

    # Shared counter for total steps taken across all workers
    global_step_counter = mp.Value('i', 0) # 'i' for integer, starts at 0

    # Manager for shared queue (needed for Queue to work across processes)
    manager = mp.Manager()
    result_queue = manager.Queue() # Queue for workers to send back results
    stop_event = manager.Event()   # Event to signal workers to stop

    # Lists for plotting overall progress (collected from queue)
    a3c_episode_rewards = []
    a3c_episode_lengths = []
    # all_worker_stats = {i: {"rewards": [], "lengths": []} for i in range(NUM_WORKERS)} # Optional detailed tracking

    print(f"\nStarting A3C Training with {NUM_WORKERS} workers...")
    print(f"Target Max Global Steps: {MAX_GLOBAL_STEPS_A3C}")
    print(f"Environment: Grid {ENV_ROWS}x{ENV_COLS}")
    print(f"Hyperparameters: Gamma={GAMMA_A3C}, LR={LR_A3C}, N_Steps={N_STEPS}, V_Coeff={VALUE_LOSS_COEFF_A3C}, E_Coeff={ENTROPY_COEFF_A3C}")

    start_time = time.time()

    # --- Create and Start Worker Processes ---
    workers = []
    for i in range(NUM_WORKERS):
        worker_process = mp.Process(target=worker,
                                    args=(i, global_model_a3c, global_optimizer_a3c,
                                          global_step_counter, MAX_GLOBAL_STEPS_A3C,
                                          ENV_ROWS, ENV_COLS, N_STEPS, GAMMA_A3C,
                                          VALUE_LOSS_COEFF_A3C, ENTROPY_COEFF_A3C,
                                          MAX_STEPS_PER_EPISODE_A3C, result_queue,
                                          stop_event))
        workers.append(worker_process)
        worker_process.start()
        print(f"Worker {i} process started.")

    # --- Monitor Queue and Collect Results ---
    finished_workers = 0
    progress_updates = 0
    error_occurred = False

    while finished_workers < NUM_WORKERS:
        try:
            # Get result from the queue (blocks until item available)
            # Add a timeout to prevent hanging indefinitely if something goes wrong
            result = result_queue.get(timeout=120) # Increased timeout

            if isinstance(result, tuple):
                message_type = result[0]
                worker_id = result[1]

                if message_type == "episode_end":
                    ep_reward = result[2]
                    ep_length = result[3]
                    a3c_episode_rewards.append(ep_reward)
                    a3c_episode_lengths.append(ep_length)

                    # Print progress periodically based on total episodes collected
                    if len(a3c_episode_rewards) % 50 == 0:
                         avg_r = np.mean(a3c_episode_rewards[-50:]) if len(a3c_episode_rewards) >= 50 else np.mean(a3c_episode_rewards)
                         avg_l = np.mean(a3c_episode_lengths[-50:]) if len(a3c_episode_lengths) >= 50 else np.mean(a3c_episode_lengths)
                         print(f" > Steps: {global_step_counter.value}, Episodes: {len(a3c_episode_rewards)}, Avg Reward (last 50): {avg_r:.2f}, Avg Length (last 50): {avg_l:.1f}")

                elif message_type == "progress":
                    current_step = result[2]
                    progress_updates += 1
                    # Optional: Print less frequently
                    # if progress_updates % (NUM_WORKERS * 10) == 0:
                    #      print(f"   Progress Update: Global Steps ~{current_step}")

                elif message_type == "error":
                    error_msg = result[2]
                    print(f"!!! Received error from worker {worker_id}: {error_msg}")
                    error_occurred = True
                    if not stop_event.is_set():
                        print("   Signaling other workers to stop due to error.")
                        stop_event.set() # Signal all other workers to stop

                elif message_type == "finished":
                    print(f"Worker {worker_id} signaled completion.")
                    finished_workers += 1

            else:
                 print(f"Warning: Received unexpected item from queue: {result}")

        except queue.Empty:
            print("Warning: Result queue timed out. Checking worker status...")
            # Check if workers are still alive, maybe break if they are not
            alive_workers = sum(p.is_alive() for p in workers)
            print(f"Active workers: {alive_workers}/{NUM_WORKERS}")
            if alive_workers == 0 and finished_workers < NUM_WORKERS:
                 print("Error: All workers seem to have exited unexpectedly.")
                 if not stop_event.is_set(): stop_event.set() # Ensure stop is signaled
                 break # Exit monitoring loop
            if stop_event.is_set():
                 print("Stop event is set, likely due to error, max steps, or manual stop. Waiting for workers to finish.")
                 # Continue waiting for "finished" signals or timeout join later
                 # Break here might prevent collecting final "finished" signals
                 # Let the loop continue until finished_workers == NUM_WORKERS or join times out

        # Check if max steps reached globally, even if workers haven't signaled finish yet
        if not stop_event.is_set() and global_step_counter.value >= MAX_GLOBAL_STEPS_A3C:
            print(f"Max global steps ({MAX_GLOBAL_STEPS_A3C}) reached. Signaling workers to stop.")
            stop_event.set()

    # --- Wait for all Worker Processes to Finish ---
    print("\nWaiting for worker processes to join...")
    active_workers_final_check = []
    for i, p in enumerate(workers):
        p.join(timeout=30) # Add a reasonable timeout for join
        if p.is_alive():
            print(f"Warning: Worker {i} did not join cleanly after timeout. Terminating.")
            p.terminate() # Forcefully terminate if stuck
            active_workers_final_check.append(i)
        # else:
        #      print(f"Worker {i} joined successfully. Exit code: {p.exitcode}")


    end_time = time.time()
    print(f"\n--- Custom Grid World Training Finished (A3C) ---")
    print(f"Total global steps reached: {global_step_counter.value}")
    print(f"Total episodes completed: {len(a3c_episode_rewards)}")
    print(f"Training time: {end_time - start_time:.2f} seconds")
    if error_occurred:
        print("Training finished DUE TO AN ERROR in one or more workers.")
    if active_workers_final_check:
        print(f"Workers {active_workers_final_check} had to be terminated.")

    # --- Plotting Results ---
    if a3c_episode_rewards:
        plt.figure(figsize=(12, 5))

        # Plot Episode Rewards
        plt.subplot(1, 2, 1)
        plt.plot(a3c_episode_rewards, label='Episode Reward', alpha=0.6)
        # Add a moving average
        if len(a3c_episode_rewards) >= 50:
            moving_avg_rewards = np.convolve(a3c_episode_rewards, np.ones(50)/50, mode='valid')
            plt.plot(np.arange(len(moving_avg_rewards)) + 49, moving_avg_rewards,
                     label='Moving Avg (50 ep)', color='red', linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("A3C Episode Rewards over Time")
        plt.legend()
        plt.grid(True)

        # Plot Episode Lengths
        plt.subplot(1, 2, 2)
        plt.plot(a3c_episode_lengths, label='Episode Length', color='orange', alpha=0.6)
         # Add a moving average
        if len(a3c_episode_lengths) >= 50:
            moving_avg_lengths = np.convolve(a3c_episode_lengths, np.ones(50)/50, mode='valid')
            plt.plot(np.arange(len(moving_avg_lengths)) + 49, moving_avg_lengths,
                     label='Moving Avg (50 ep)', color='blue', linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.title("A3C Episode Lengths over Time")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("a3c_custom_grid_training_progress.png")
        print("\nPlot saved to a3c_custom_grid_training_progress.png")
        # plt.show() # Uncomment to display plot directly
    else:
        print("\nNo episode data collected, skipping plotting.")

    # --- Summary and a greedy test of the trained global model ---
    if a3c_episode_rewards:
        last = min(50, len(a3c_episode_rewards))
        print(f"Average over the last {last} episodes: reward {np.mean(a3c_episode_rewards[-last:]):.2f}, "
              f"length {np.mean(a3c_episode_lengths[-last:]):.1f} steps")
    test_env = GridEnvironment(rows=ENV_ROWS, cols=ENV_COLS)
    state, test_reward, reached_goal = test_env.reset(), 0.0, False
    for test_step in range(1, MAX_STEPS_PER_EPISODE_A3C + 1):
        with torch.no_grad():
            action_dist, _ = global_model_a3c(state)
        state, reward, reached_goal = test_env.step(int(action_dist.probs.argmax()))
        test_reward += reward
        if reached_goal:
            break
    print(f"Greedy policy from the start: {'reached the goal' if reached_goal else 'did not reach the goal'} "
          f"after {test_step} steps, reward {test_reward:.2f}")

    print("Script finished.")