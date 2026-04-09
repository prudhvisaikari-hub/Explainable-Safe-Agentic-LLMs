"""Specialist Agent for domain-specific decision making.
Handles specialized tasks like traffic, ambulance, fire, and police coordination.
"""

import numpy as np
from typing import Any, Dict, List, Optional
from .base_agent import BaseAgent


class SpecialistAgent(BaseAgent):
    """
    A specialist agent that focuses on a specific domain.
    Used for specialized emergency response coordination.
    """

    SPECIALIST_TYPES = ['traffic', 'ambulance', 'fire', 'police']

    def __init__(self, agent_id: str, name: str, specialist_type: str):
        if specialist_type not in self.SPECIALIST_TYPES:
            raise ValueError(f"Invalid specialist type: {specialist_type}")
        super().__init__(agent_id, name, f"specialist_{specialist_type}")
        self.specialist_type = specialist_type
        self.domain_knowledge: Dict[str, Any] = {}
        self.priority_threshold = 0.7
        self.last_decision = None
        self.last_percepts = None

    def perceive(self, observation: np.ndarray) -> Dict[str, Any]:
        """Extract domain-relevant features from observations."""
        percepts = {
            'timestamp': observation[0] if len(observation) > 0 else 0,
            'incident_severity': observation[1] if len(observation) > 1 else 0,
            'vehicle_availability': observation[2] if len(observation) > 2 else 0,
            'traffic_density': observation[3] if len(observation) > 3 else 0,
            'distance_to_incident': observation[4] if len(observation) > 4 else 0,
            'weather_conditions': observation[5] if len(observation) > 5 else 0,
            'specialist_type': self.specialist_type
        }
        self.last_percepts = percepts
        return percepts

    def reason(self, percepts: Dict[str, Any]) -> Dict[str, Any]:
        """Domain-specific reasoning based on specialist type."""
        severity = percepts.get('incident_severity', 0)
        availability = percepts.get('vehicle_availability', 0)
        distance = percepts.get('distance_to_incident', 100)

        # Calculate urgency score
        urgency = severity * (1.0 / (distance + 1)) * availability

        # Specialist-specific reasoning
        if self.specialist_type == 'traffic':
            traffic_density = percepts.get('traffic_density', 0)
            priority = urgency * (1 + traffic_density * 0.5)
            action_suggestion = self._traffic_reasoning(priority, percepts)
        elif self.specialist_type == 'ambulance':
            priority = urgency * 1.5  # Higher priority for medical
            action_suggestion = self._ambulance_reasoning(priority, percepts)
        elif self.specialist_type == 'fire':
            weather = percepts.get('weather_conditions', 0)
            priority = urgency * (1 + weather * 0.3)
            action_suggestion = self._fire_reasoning(priority, percepts)
        else:  # police
            priority = urgency * 1.2
            action_suggestion = self._police_reasoning(priority, percepts)

        reasoning = {
            'urgency_score': urgency,
            'priority': priority,
            'action_suggestion': action_suggestion,
            'confidence': min(urgency / self.priority_threshold, 1.0),
            'specialist_type': self.specialist_type
        }
        self.last_decision = reasoning
        return reasoning

    def _traffic_reasoning(self, priority: float, percepts: Dict) -> str:
        if priority > 0.8:
            return "Clear all lanes for emergency vehicle passage"
        elif priority > 0.5:
            return "Reduce traffic density in affected area"
        else:
            return "Monitor traffic conditions"

    def _ambulance_reasoning(self, priority: float, percepts: Dict) -> str:
        if priority > 0.8:
            return "Dispatch nearest ambulance with priority routing"
        elif priority > 0.5:
            return "Alert nearby hospitals for potential arrival"
        else:
            return "Stand by for medical support"

    def _fire_reasoning(self, priority: float, percepts: Dict) -> str:
        if priority > 0.8:
            return "Dispatch fire trucks with water supply"
        elif priority > 0.5:
            return "Evacuate nearby areas and alert fire department"
        else:
            return "Monitor for fire hazards"

    def _police_reasoning(self, priority: float, percepts: Dict) -> str:
        if priority > 0.8:
            return "Dispatch police units for emergency response"
        elif priority > 0.5:
            return "Coordinate with other emergency services"
        else:
            return "Maintain security perimeter"

    def act(self, reasoning: Dict[str, Any]) -> Any:
        """Execute the suggested action."""
        action = reasoning.get('action_suggestion', 'monitor')
        confidence = reasoning.get('confidence', 0.0)

        # Only act if confidence is above threshold
        if confidence >= self.priority_threshold:
            return {
                'action': action,
                'executed': True,
                'agent_id': self.agent_id,
                'specialist_type': self.specialist_type
            }
        else:
            return {
                'action': 'wait_for_higher_priority',
                'executed': False,
                'agent_id': self.agent_id,
                'reason': f"Confidence {confidence:.2f} below threshold {self.priority_threshold}"
            }

    def explain(self) -> str:
        """Generate explanation for the last decision."""
        if self.last_decision is None:
            return "No decision has been made yet."

        d = self.last_decision
        percepts = self.last_percepts or {}

        explanation = (
            f"As a {self.specialist_type} specialist, I assessed the situation "
            f"with urgency score {d.get('urgency_score', 0):.2f} and priority "
            f"{d.get('priority', 0):.2f}. Based on incident severity of "
            f"{percepts.get('incident_severity', 0):.2f} and vehicle availability "
            f"of {percepts.get('vehicle_availability', 0):.2f}, I recommend "
            f"'{d.get('action_suggestion', 'unknown')}' with confidence "
            f"{d.get('confidence', 0):.2f}."
        )
        self.explanation_history.append(explanation)
        return explanation

    def get_stats(self) -> Dict[str, Any]:
        """Return specialist-specific statistics."""
        base_stats = super().get_stats()
        base_stats['specialist_type'] = self.specialist_type
        base_stats['priority_threshold'] = self.priority_threshold
        base_stats['domain_knowledge_size'] = len(self.domain_knowledge)
        return base_stats
