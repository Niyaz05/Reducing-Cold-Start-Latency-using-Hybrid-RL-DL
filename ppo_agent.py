"""
ppo_agent.py
------------
PPO Actor-Critic agent, closely matching the CASR paper's DRL design.

Architecture (from CASR paper, Table 2):
    Input layer:  21 dimensions (state vector)
    Hidden layer: 1 layer, 128 neurons, Tanh activation
    Output layer:
        Actor  → 27-class softmax (action probabilities)
        Critic → 1 scalar (state value)

PPO-clip objective (Eq. 8 in CASR paper):
    L_clip(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

Entropy bonus (Eq. 11-12 in CASR paper):
    H(π(·|s_t)) = E[-log π(a_t|s_t)]
    L(θ) = -L_clip(θ) + ρ·H(π(·|s_t))

Hyperparameters from CASR Table 2:
    γ = 0.63       (discount factor - low to avoid considering irrelevant future)
    ε = 0.2        (clip parameter)
    ρ = 0.01       (entropy coefficient)
    lr = 0.001     (learning rate for both actor and critic)
    hidden = 128   (neurons per hidden layer)
    activation = tanh
    epochs = 10    (updates per batch)
    batch_size = 20 (mini-batch size)
    buffer = 1000  (replay buffer capacity in steps)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Neural network modules
# ──────────────────────────────────────────────────────────────────────────────

class _ActorNet(nn.Module):
    """
    Policy network.
    Input: state (21-dim)  →  Output: probability distribution over 27 actions
    """
    def __init__(self, state_dim: int = 21, action_dim: int = 27, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        # Orthogonal initialisation (recommended in CASR paper reference [39])
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


class _CriticNet(nn.Module):
    """
    Value network.
    Input: state (21-dim)  →  Output: scalar value V(s)
    """
    def __init__(self, state_dim: int = 21, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Rollout buffer
# ──────────────────────────────────────────────────────────────────────────────

class _RolloutBuffer:
    """Stores trajectories collected during environment interaction."""

    def __init__(self, capacity: int = 1000, state_dim: int = 21):
        self.capacity = capacity
        self.state_dim = state_dim
        self._clear()

    def _clear(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    @property
    def count(self) -> int:
        return len(self.rewards)

    def store(
        self, state, action, log_prob, reward, next_state, done, value
    ):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.values.append(value)

    def get_tensors(self, device: torch.device) -> dict:
        return {
            "states": torch.tensor(
                np.array(self.states), dtype=torch.float32, device=device
            ),
            "actions": torch.tensor(
                self.actions, dtype=torch.long, device=device
            ),
            "log_probs": torch.tensor(
                self.log_probs, dtype=torch.float32, device=device
            ),
            "rewards": torch.tensor(
                self.rewards, dtype=torch.float32, device=device
            ),
            "dones": torch.tensor(
                self.dones, dtype=torch.float32, device=device
            ),
            "values": torch.tensor(
                self.values, dtype=torch.float32, device=device
            ),
        }

    def clear(self):
        self._clear()


# ──────────────────────────────────────────────────────────────────────────────
# PPO Agent
# ──────────────────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    Proximal Policy Optimisation agent (PPO-clip with entropy bonus).

    Hyperparameters match CASR Table 2 by default.
    """

    def __init__(
        self,
        state_dim: int = 21,
        action_dim: int = 27,
        hidden: int = 128,
        lr_actor: float = 0.001,
        lr_critic: float = 0.001,
        gamma: float = 0.63,          # CASR: 0.63 (low discount for fast-changing workloads)
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,   # ρ in CASR
        epochs: int = 10,
        mini_batch_size: int = 20,
        max_grad_norm: float = 0.5,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[PPOAgent] Using device: {self.device}")

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm

        # Networks
        self.actor = _ActorNet(state_dim, action_dim, hidden).to(self.device)
        self.critic = _CriticNet(state_dim, hidden).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = _RolloutBuffer(capacity=1000, state_dim=state_dim)

        # Logging
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []
        self.entropies: List[float] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Action selection
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Select action and return (action, log_prob, value).
        Called during environment interaction (no gradient needed).
        """
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.actor(s)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(s)
        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Store transitions
    # ──────────────────────────────────────────────────────────────────────────

    def store_transition(self, state, action, log_prob, reward, next_state, done, value):
        self.buffer.store(state, action, log_prob, reward, next_state, done, value)

    # ──────────────────────────────────────────────────────────────────────────
    # PPO update
    # ──────────────────────────────────────────────────────────────────────────

    def update(self) -> dict:
        """
        Perform PPO update on collected buffer.
        Returns dict of mean losses for logging.
        """
        if self.buffer.count == 0:
            return {}

        tensors = self.buffer.get_tensors(self.device)
        states = tensors["states"]
        actions = tensors["actions"]
        old_log_probs = tensors["log_probs"]
        rewards = tensors["rewards"]
        dones = tensors["dones"]
        old_values = tensors["values"]

        # ── Compute returns and advantages (GAE) ─────────────────────────────
        advantages, returns = self._compute_gae(rewards, old_values, dones)

        # Normalise advantages (helps training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── PPO epochs ────────────────────────────────────────────────────────
        n = len(states)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        update_count = 0

        for _ in range(self.epochs):
            # Shuffle for mini-batch sampling
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.mini_batch_size):
                idx = indices[start: start + self.mini_batch_size]
                if len(idx) < 2:
                    continue

                s_batch = states[idx]
                a_batch = actions[idx]
                old_lp_batch = old_log_probs[idx]
                adv_batch = advantages[idx]
                ret_batch = returns[idx]

                # New policy evaluation
                probs = self.actor(s_batch)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(a_batch)
                entropy = dist.entropy().mean()

                # PPO-clip ratio
                ratio = torch.exp(new_log_probs - old_lp_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_batch

                # Actor loss: -L_clip + ρ·H (Eq. 12 in CASR paper)
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                # Critic loss: MSE
                value_pred = self.critic(s_batch)
                critic_loss = nn.functional.mse_loss(value_pred, ret_batch)

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                update_count += 1

        self.buffer.clear()

        mean_actor_loss = total_actor_loss / max(1, update_count)
        mean_critic_loss = total_critic_loss / max(1, update_count)
        mean_entropy = total_entropy / max(1, update_count)

        self.actor_losses.append(mean_actor_loss)
        self.critic_losses.append(mean_critic_loss)
        self.entropies.append(mean_entropy)

        return {
            "actor_loss": mean_actor_loss,
            "critic_loss": mean_critic_loss,
            "entropy": mean_entropy,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # GAE calculation
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalised Advantage Estimation (GAE-λ).
        Returns (advantages, discounted_returns).
        """
        n = len(rewards)
        advantages = torch.zeros(n, device=self.device)
        gae = 0.0

        # Bootstrap value for last step = 0 (assuming episode ends or truncated)
        next_value = 0.0

        for t in reversed(range(n)):
            mask = 1.0 - dones[t].item()
            delta = rewards[t].item() + self.gamma * next_value * mask - values[t].item()
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            next_value = values[t].item()

        returns = advantages + values
        return advantages, returns

    # ──────────────────────────────────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
        }, path)
        print(f"[PPOAgent] Saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_optimizer.load_state_dict(ckpt["actor_opt"])
        self.critic_optimizer.load_state_dict(ckpt["critic_opt"])
        print(f"[PPOAgent] Loaded ← {path}")
