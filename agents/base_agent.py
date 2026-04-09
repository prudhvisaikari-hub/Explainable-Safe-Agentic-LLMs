"""Base Agent class for the Multi-Agent Framework.
Defines the abstract interface for all agent types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    Each agent can perceive, reason, act, and explain its decisions.
    """

    def __init__(self, agent_id: str, name: str, role: str):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.state_history: List[Dict] = []
        self.action_history: List[Dict] = []
        self.explanation_history: List[str] = []
        self.trust_score: float = 1.0
        self.safety_violations: int = 0

    @abstractmethod
    def perceive(self, observation: np.ndarray) -> Dict[str, Any]:
        """Process raw observations into meaningful percepts."""
        pass

    @abstractmethod
    def reason(self, percepts: Dict[str, Any]) -> Dict[str, Any]:
        """Reason about percepts to form a decision."""
        pass

    @abstractmethod
    def act(self, reasoning: Dict[str, Any]) -> Any:
        """Execute an action based on reasoning."""
        pass

    @abstractmethod
    def explain(self) -> str:
        """Generate a natural language explanation for the last action."""
        pass

    def record_state(self, state: Dict[str, Any]):
        """Record the current state for history tracking."""
        self.state_history.append(state)

    def record_action(self, action: Any, reward: float):
        """Record an action and its reward."""
        self.action_history.append({
            'action': action,
            'reward': reward,
            'agent_id': self.agent_id
        })

    def update_trust(self, delta: float):
        """Update trust score based on performance."""
        self.trust_score = np.clip(self.trust_score + delta, 0.0, 1.0)

    def increment_safety_violation(self):
        """Track safety constraint violations."""
        self.safety_violations += 1

    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics."""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'role': self.role,
            'trust_score': self.trust_score,
            'safety_violations': self.safety_violations,
            'total_actions': len(self.action_history),
            'avg_reward': np.mean([a['reward'] for a in self.action_history]) if self.action_history else 0.0
        }
