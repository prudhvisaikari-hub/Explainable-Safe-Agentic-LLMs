"""Central Planner Agent for coordinating multi-agent decisions.
Aggregates specialist recommendations and produces unified actions.
"""

import numpy as np
from typing import Any, Dict, List, Optional
from .base_agent import BaseAgent


class CentralPlanner(BaseAgent):
    """
    The central coordinator that aggregates decisions from all
    specialist agents and produces a unified action plan.
    Implements risk-aware decision fusion.
    """

    def __init__(self, agent_id: str = "planner_001", name: str = "CentralPlanner"):
        super().__init__(agent_id, name, "central_planner")
        self.specialist_agents: Dict[str, Any] = {}
        self.risk_threshold = 0.3
        self.consensus_threshold = 0.6
        self.last_aggregated_state = None
        self.decision_log: List[Dict] = []

    def register_specialist(self, specialist: Any):
        """Register a specialist agent for coordination."""
        self.specialist_agents[specialist.agent_id] = specialist

    def perceive(self, observation: np.ndarray) -> Dict[str, Any]:
        """Aggregate observations from all registered specialists."""
        aggregated = {
            'global_state': observation,
            'specialist_percepts': {},
            'num_specialists': len(self.specialist_agents)
        }

        for agent_id, specialist in self.specialist_agents.items():
            try:
                percepts = specialist.perceive(observation)
                aggregated['specialist_percepts'][agent_id] = percepts
            except Exception as e:
                aggregated['specialist_percepts'][agent_id] = {'error': str(e)}

        self.last_aggregated_state = aggregated
        return aggregated

    def reason(self, percepts: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse specialist recommendations into a unified plan."""
        specialist_reasons = {}
        recommendations = []

        for agent_id, specialist in self.specialist_agents.items():
            try:
                specialist_percepts = percepts['specialist_percepts'].get(agent_id, {})
                if 'error' not in specialist_percepts:
                    reason = specialist.reason(specialist_percepts)
                    specialist_reasons[agent_id] = reason
                    recommendations.append({
                        'agent_id': agent_id,
                        'priority': reason.get('priority', 0),
                        'action': reason.get('action_suggestion', ''),
                        'confidence': reason.get('confidence', 0)
                    })
            except Exception as e:
                specialist_reasons[agent_id] = {'error': str(e)}

        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)

        # Calculate risk score
        avg_confidence = np.mean([r['confidence'] for r in recommendations]) if recommendations else 0
        risk_score = 1.0 - avg_confidence

        # Determine final action
        if risk_score > self.risk_threshold:
            final_action = 'escalate_to_human'
            confidence = 0.5
        elif recommendations:
            top_rec = recommendations[0]
            if top_rec['confidence'] >= self.consensus_threshold:
                final_action = top_rec['action']
                confidence = top_rec['confidence']
            else:
                final_action = 'wait_for_consensus'
                confidence = top_rec['confidence'] * 0.5
        else:
            final_action = 'no_recommendation'
            confidence = 0.0

        reasoning = {
            'specialist_reasons': specialist_reasons,
            'recommendations': recommendations,
            'final_action': final_action,
            'confidence': confidence,
            'risk_score': risk_score,
            'consensus_achieved': confidence >= self.consensus_threshold
        }

        self.decision_log.append(reasoning)
        return reasoning

    def act(self, reasoning: Dict[str, Any]) -> Any:
        """Execute the unified action plan."""
        action = reasoning.get('final_action', 'no_action')
        confidence = reasoning.get('confidence', 0.0)

        if action == 'escalate_to_human':
            return {
                'action': action,
                'reasoning': reasoning,
                'requires_human': True,
                'risk_score': reasoning.get('risk_score', 1.0)
            }
        elif action == 'wait_for_consensus':
            return {
                'action': action,
                'reasoning': reasoning,
                'requires_human': False
            }
        else:
            return {
                'action': action,
                'reasoning': reasoning,
                'executed': confidence >= self.consensus_threshold,
                'requires_human': False
            }

    def explain(self) -> str:
        """Explain the planning decision."""
        if not self.decision_log:
            return "No decisions have been logged yet."

        last = self.decision_log[-1]
        num_specs = len(last.get('specialist_reasons', {}))
        risk = last.get('risk_score', 0)
        confidence = last.get('confidence', 0)
        action = last.get('final_action', 'unknown')

        explanation = (
            f"The central planner aggregated inputs from {num_specs} specialist agents. "
            f"The overall risk score was {risk:.2f} with a confidence of {confidence:.2f}. "
            f"Based on this analysis, the recommended action is '{action}'. "
        )

        if risk > self.risk_threshold:
            explanation += "Due to elevated risk, human oversight is recommended."
        elif last.get('consensus_achieved'):
            explanation += "Strong consensus was achieved among specialists."
        else:
            explanation += "Consensus was partial; continued monitoring advised."

        self.explanation_history.append(explanation)
        return explanation

    def get_stats(self) -> Dict[str, Any]:
        """Return planner-specific statistics."""
        base_stats = super().get_stats()
        base_stats['num_specialists'] = len(self.specialist_agents)
        base_stats['risk_threshold'] = self.risk_threshold
        base_stats['consensus_threshold'] = self.consensus_threshold
        base_stats['decisions_logged'] = len(self.decision_log)
        return base_stats
