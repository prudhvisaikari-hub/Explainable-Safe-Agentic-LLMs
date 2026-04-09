"""Safe Reinforcement Learning Agent with constraint-aware policies.
Implements safe RL for emergency vehicle allocation.
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional


class SafeRLAgent:
    """
    A safe reinforcement learning agent that learns policies
    while respecting safety constraints.
    Uses a combination of PPO/SAC with safety layers.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        safety_weight: float = 0.5
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.safety_weight = safety_weight

        # Policy network parameters (placeholder - would use neural net in practice)
        self.policy_weights = np.random.randn(state_dim, action_dim) * 0.1
        self.value_weights = np.random.randn(state_dim, 1) * 0.1

        # Training buffers
        self.state_buffer: List[np.ndarray] = []
        self.action_buffer: List[int] = []
        self.reward_buffer: List[float] = []
        self.next_state_buffer: List[np.ndarray] = []
        self.done_buffer: List[bool] = []
        self.safety_violation_buffer: List[bool] = []

        # Statistics
        self.episode_rewards: List[float] = []
        self.episode_safety_violations: List[int] = []
        self.total_steps = 0
        self.total_episodes = 0

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        q_values = state @ self.policy_weights
        return int(np.argmax(q_values))

    def select_safe_action(
        self,
        state: np.ndarray,
        safe_actions: List[int],
        epsilon: float = 0.1
    ) -> int:
        """Select action from safe action set only."""
        if not safe_actions:
            return self.select_action(state, epsilon)

        if np.random.random() < epsilon:
            return np.random.choice(safe_actions)

        q_values = state @ self.policy_weights
        safe_q_values = {a: q_values[a] for a in safe_actions}
        return max(safe_q_values, key=safe_q_values.get)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        safety_violation: bool = False
    ):
        """Store a transition in the replay buffer."""
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.next_state_buffer.append(next_state)
        self.done_buffer.append(done)
        self.safety_violation_buffer.append(safety_violation)
        self.total_steps += 1

    def compute_safe_reward(
        self,
        reward: float,
        safety_violation: bool
    ) -> float:
        """Compute reward with safety penalty."""
        if safety_violation:
            return reward - self.safety_weight * 10.0
        return reward

    def update_policy(self, batch_size: int = 64) -> Dict[str, float]:
        """Update policy using batch of experiences."""
        if len(self.state_buffer) < batch_size:
            return {'loss': 0.0, 'safety_violations': 0}

        indices = np.random.choice(
            len(self.state_buffer),
            size=min(batch_size, len(self.state_buffer)),
            replace=False
        )

        states = np.array([self.state_buffer[i] for i in indices])
        actions = np.array([self.action_buffer[i] for i in indices])
        rewards = np.array([self.reward_buffer[i] for i in indices])
        next_states = np.array([self.next_state_buffer[i] for i in indices])
        dones = np.array([self.done_buffer[i] for i in indices])
        safety_violations = np.array([self.safety_violation_buffer[i] for i in indices])

        # Compute TD targets
        current_q = np.sum(states * self.policy_weights, axis=1)
        next_q = np.max(next_states @ self.policy_weights, axis=1)
        targets = rewards + self.gamma * next_q * (1 - dones)

        # Safety penalty
        targets = targets - self.safety_weight * safety_violations * 10.0

        # Policy gradient update (simplified)
        errors = targets - current_q
        grad = np.zeros_like(self.policy_weights)
        for i, idx in enumerate(indices):
            grad[:, actions[i]] += errors[i] * states[i]

        self.policy_weights += self.lr * grad / len(indices)

        # Clear buffer
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.next_state_buffer = []
        self.done_buffer = []
        self.safety_violation_buffer = []

        return {
            'loss': float(np.mean(errors ** 2)),
            'safety_violations': int(np.sum(safety_violations))
        }

    def end_episode(self, episode_reward: float, safety_violations: int):
        """Record episode statistics."""
        self.episode_rewards.append(episode_reward)
        self.episode_safety_violations.append(safety_violations)
        self.total_episodes += 1

    def get_stats(self) -> Dict[str, Any]:
        """Return training statistics."""
        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'avg_safety_violations': np.mean(self.episode_safety_violations) if self.episode_safety_violations else 0.0,
            'safety_weight': self.safety_weight,
            'gamma': self.gamma
        }

    def save_policy(self, filepath: str):
        """Save policy weights to file."""
        np.savez(
            filepath,
            policy_weights=self.policy_weights,
            value_weights=self.value_weights,
            stats=self.get_stats()
        )

    def load_policy(self, filepath: str):
        """Load policy weights from file."""
        data = np.load(filepath)
        self.policy_weights = data['policy_weights']
        self.value_weights = data['value_weights']
