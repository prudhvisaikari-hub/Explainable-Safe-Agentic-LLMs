from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent


class SafeAgent(BaseAgent):
    """Safe reinforcement learning agent with risk-aware decision making."""

    def __init__(self, state_dim: int, action_dim: int, risk_threshold: float = 0.3):
        super().__init__(state_dim, action_dim)
        self.risk_threshold = risk_threshold
        self.safety_policy = SafetyPolicy()

    def select_action(self, state: Any, training: bool = False) -> int:
        """Select a safe action based on current state."""
        action = self._select_action_internal(state, training)
        safe_action = self.safety_policy.validate_action(action, state)
        return safe_action

    def _select_action_internal(self, state: Any, training: bool) -> int:
        """Internal action selection logic."""
        if training:
            return self._exploration_action(state)
        return self._greedy_action(state)

    def _exploration_action(self, state: Any) -> int:
        """Exploration with safety constraints."""
        return self.action_sampler.sample_safe(state)

    def _greedy_action(self, state: Any) -> int:
        """Greedy action selection."""
        return self.policy_network.get_best_action(state)

    def update_policy(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        """Update the policy with a safe transition."""
        safety_cost = self.safety_policy.assess_transition(state, action, next_state)
        if safety_cost > self.risk_threshold:
            reward = reward * (1 - safety_cost)
        super().update_policy(state, action, reward, next_state, done)

    def get_risk_estimate(self, state: Any) -> float:
        """Estimate the risk level of the current state."""
        return self.safety_policy.estimate_risk(state)

    def is_state_safe(self, state: Any) -> bool:
        """Check if a state is within safe operating bounds."""
        return self.get_risk_estimate(state) <= self.risk_threshold


class SafetyPolicy:
    """Safety policy for validating agent actions."""

    def __init__(self):
        self.constraints: List[callable] = []
        self.risk_model: Optional[Any] = None

    def add_constraint(self, constraint_fn: callable):
        """Add a safety constraint."""
        self.constraints.append(constraint_fn)

    def validate_action(self, action: int, state: Any) -> int:
        """Validate and potentially modify an action."""
        for constraint in self.constraints:
            if not constraint(action, state):
                return self._get_safe_alternative(action, state)
        return action

    def _get_safe_alternative(self, action: int, state: Any) -> int:
        """Find a safe alternative action."""
        for alt_action in range(10):
            if all(c(alt_action, state) for c in self.constraints):
                return alt_action
        return 0

    def assess_transition(self, state: Any, action: int, next_state: Any) -> float:
        """Assess the safety cost of a transition."""
        risk = 0.0
        for constraint in self.constraints:
            if not constraint(action, state):
                risk += 0.1
        return min(1.0, risk)

    def estimate_risk(self, state: Any) -> float:
        """Estimate the risk of a state."""
        if self.risk_model:
            return self.risk_model.predict(state)
        return 0.5
