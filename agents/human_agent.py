"""Human Agent for Human-in-the-Loop decision making.
Provides human oversight and override capabilities.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from .base_agent import BaseAgent


class HumanAgent(BaseAgent):
    """
    Represents the human operator in the Human-in-the-Loop system.
    Can review, approve, override, or provide feedback on agent decisions.
    """

    def __init__(self, agent_id: str = "human_001", name: str = "HumanOperator"):
        super().__init__(agent_id, name, "human_operator")
        self.approval_history: List[Dict] = []
        self.override_history: List[Dict] = []
        self.feedback_history: List[Dict] = []
        self.current_review = None
        self.response_time_threshold = 300  # seconds

    def perceive(self, observation: np.ndarray) -> Dict[str, Any]:
        """Present the situation to the human operator."""
        percepts = {
            'observation': observation,
            'pending_decisions': [],
            'alerts': [],
            'system_status': 'active'
        }
        return percepts

    def review_decision(self, agent_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Present a decision for human review."""
        self.current_review = {
            'agent_decision': agent_decision,
            'reviewed_at': None,
            'action': None,
            'feedback': None
        }
        return self.current_review

    def approve(self, feedback: str = "") -> Dict[str, Any]:
        """Approve the current decision under review."""
        if self.current_review is None:
            return {'success': False, 'error': 'No decision under review'}

        self.current_review['action'] = 'approved'
        self.current_review['feedback'] = feedback
        self.current_review['reviewed_at'] = 'now'

        self.approval_history.append(self.current_review.copy())
        self.record_action('approve', 1.0)
        return {
            'success': True,
            'action': 'approved',
            'feedback': feedback
        }

    def override(self, new_action: str, reason: str = "") -> Dict[str, Any]:
        """Override the current decision with a new action."""
        if self.current_review is None:
            return {'success': False, 'error': 'No decision under review'}

        self.current_review['action'] = 'overridden'
        self.current_review['new_action'] = new_action
        self.current_review['override_reason'] = reason
        self.current_review['reviewed_at'] = 'now'

        self.override_history.append(self.current_review.copy())
        self.record_action('override', 0.5)
        self.increment_safety_violation()  # Override indicates potential AI error

        return {
            'success': True,
            'action': 'overridden',
            'new_action': new_action,
            'reason': reason
        }

    def reject(self, reason: str = "") -> Dict[str, Any]:
        """Reject the current decision."""
        if self.current_review is None:
            return {'success': False, 'error': 'No decision under review'}

        self.current_review['action'] = 'rejected'
        self.current_review['rejection_reason'] = reason
        self.current_review['reviewed_at'] = 'now'

        self.record_action('reject', 0.0)
        return {
            'success': True,
            'action': 'rejected',
            'reason': reason
        }

    def provide_feedback(self, agent_id: str, feedback: str, rating: float) -> Dict[str, Any]:
        """Provide feedback to improve agent performance."""
        feedback_entry = {
            'agent_id': agent_id,
            'feedback': feedback,
            'rating': np.clip(rating, 0.0, 5.0),
            'timestamp': 'now'
        }
        self.feedback_history.append(feedback_entry)
        return feedback_entry

    def reason(self, percepts: Dict[str, Any]) -> Dict[str, Any]:
        """Human reasoning is simulated - in practice this is manual."""
        return {
            'human_involved': True,
            'attention_level': 'high',
            'decision_complexity': 'variable',
            'available_actions': ['approve', 'override', 'reject', 'defer']
        }

    def act(self, reasoning: Dict[str, Any]) -> Any:
        """Human action - in practice this is interactive."""
        return {
            'action': 'awaiting_human_input',
            'reasoning': reasoning,
            'human_in_loop': True
        }

    def explain(self) -> str:
        """Explain human decision making."""
        if self.current_review is None:
            return "No active review in progress."

        action = self.current_review.get('action', 'none')
        feedback = self.current_review.get('feedback', '') or self.current_review.get('override_reason', '') or self.current_review.get('rejection_reason', '')

        explanation = f"As the human operator, I have taken the action '{action}' on the pending decision."
        if feedback:
            explanation += f" Reason: {feedback}"

        self.explanation_history.append(explanation)
        return explanation

    def get_stats(self) -> Dict[str, Any]:
        """Return human operator statistics."""
        base_stats = super().get_stats()
        base_stats['approvals'] = len(self.approval_history)
        base_stats['overrides'] = len(self.override_history)
        base_stats['feedback_count'] = len(self.feedback_history)
        base_stats['response_time_threshold'] = self.response_time_threshold

        # Calculate approval rate
        total_reviews = len(self.approval_history) + len(self.override_history)
        base_stats['approval_rate'] = len(self.approval_history) / max(total_reviews, 1)

        # Average feedback rating
        if self.feedback_history:
            base_stats['avg_feedback_rating'] = np.mean([f['rating'] for f in self.feedback_history])
        else:
            base_stats['avg_feedback_rating'] = 0.0

        return base_stats
