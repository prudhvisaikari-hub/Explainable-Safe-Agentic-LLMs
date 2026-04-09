"""Reward Shaping for Safe Agentic LLM training.
Defines shaped rewards for emergency vehicle allocation.
"""

import numpy as np
from typing import Any, Dict, Tuple


class RewardShaper:
    """
    Computes shaped rewards that guide the agent toward
    safe and efficient emergency response behaviors.
    """

    def __init__(
        self,
        time_penalty: float = -0.1,
        success_reward: float = 10.0,
        safety_bonus: float = 5.0,
        efficiency_bonus: float = 2.0
    ):
        self.time_penalty = time_penalty
        self.success_reward = success_reward
        self.safety_bonus = safety_bonus
        self.efficiency_bonus = efficiency_bonus

    def compute_reward(
        self,
        state: Dict[str, Any],
        action: int,
        info: Dict[str, Any],
        done: bool
    ) -> Tuple[float, Dict[str, float]]:
        """Compute shaped reward with component breakdown."""
        components = {}

        # Base reward components
        r_time = self.time_penalty  # Penalize each step
        r_safety = 0.0
        r_efficiency = 0.0
        r_success = 0.0

        # Safety reward based on constraint satisfaction
        safety_violation = info.get('safety_violation', False)
        if not safety_violation:
            r_safety = self.safety_bonus
        else:
            r_safety = -self.safety_bonus * 2
        components['safety'] = r_safety

        # Efficiency reward based on response time
        response_time = info.get('response_time', 100)
        if response_time < 30:
            r_efficiency = self.efficiency_bonus
        elif response_time < 60:
            r_efficiency = self.efficiency_bonus * 0.5
        else:
            r_efficiency = 0.0
        components['efficiency'] = r_efficiency

        # Success reward on episode completion
        if done:
            success = info.get('success', False)
            if success:
                r_success = self.success_reward
            else:
                r_success = -self.success_reward * 0.5
        components['success'] = r_success

        # Time step penalty
        components['time'] = r_time

        total_reward = r_time + r_safety + r_efficiency + r_success

        return total_reward, components

    def compute_potential_based_shaping(
        self,
        state: Dict[str, Any],
        info: Dict[str, Any],
        gamma: float = 0.99
    ) -> float:
        """Compute potential-based reward shaping (Ng et al. 1999)."""
        phi_state = self._state_potential(state)
        phi_next = self._state_potential(info)
        return gamma * phi_next - phi_state

    def _state_potential(self, state: Dict[str, Any]) -> float:
        """Compute potential function for a state."""
        # Potential based on proximity to goal
        distance = state.get('distance_to_goal', 100)
        severity = state.get('incident_severity', 0)
        availability = state.get('vehicle_availability', 0)

        potential = -distance * severity + availability * 10
        return potential / 100.0

    def get_reward_breakdown(self, total_reward: float, components: Dict[str, float]) -> str:
        """Generate human-readable reward breakdown."""
        breakdown = f"Total reward: {total_reward:.2f}\n"
        for key, value in components.items():
            breakdown += f"  - {key}: {value:.2f}\n"
        return breakdown
