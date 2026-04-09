"""Central explanation engine for the multi-agent framework.

Orchestrates all XAI components to provide unified explanations
for agent decisions, safety constraints, and reward shaping.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class ExplanationType(Enum):
    """Types of explanations available in the framework."""
    DECISION = "decision"
    SAFETY = "safety"
    REWARD = "reward"
    COUNTERFACTUAL = "counterfactual"
    ATTENTION = "attention"
    FEATURE_IMPORTANCE = "feature_importance"
    NATURAL_LANGUAGE = "natural_language"


@dataclass
class Explanation:
    """Represents a single explanation object."""
    explanation_type: ExplanationType
    agent_id: str
    decision: str
    description: str
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class MultiAgentExplanation:
    """Aggregated explanation across multiple agents."""
    explanations: List[Explanation] = field(default_factory=list)
    consensus_score: float = 0.0
    disagreement_points: List[Dict[str, Any]] = field(default_factory=list)
    human_feedback: Optional[str] = None
    revision_history: List[Dict[str, Any]] = field(default_factory=list)


class ExplanationEngine:
    """Central engine for generating and managing explanations.

    This class orchestrates all XAI components including attention-based
    explanations, counterfactual reasoning, feature importance analysis,
    and natural language generation for multi-agent decisions.
    """

    def __init__(self, agents: Optional[Dict[str, Any]] = None):
        """Initialize the explanation engine.

        Args:
            agents: Dictionary of agent objects for explanation context.
        """
        self.agents = agents or {}
        self.explanation_history: List[Explanation] = []
        self.multi_agent_explanations: List[MultiAgentExplanation] = []
        self._attention_explainer = None
        self._counterfactual_explainer = None
        self._feature_importance = None
        self._nl_explainer = None

    def set_components(
        self,
        attention_explainer=None,
        counterfactual_explainer=None,
        feature_importance=None,
        nl_explainer=None
    ):
        """Set XAI component instances.

        Args:
            attention_explainer: AttentionExplainer instance.
            counterfactual_explainer: CounterfactualExplainer instance.
            feature_importance: FeatureImportance instance.
            nl_explainer: NaturalLanguageExplainer instance.
        """
        self._attention_explainer = attention_explainer
        self._counterfactual_explainer = counterfactual_explainer
        self._feature_importance = feature_importance
        self._nl_explainer = nl_explainer

    def explain_decision(
        self,
        agent_id: str,
        decision: str,
        context: Dict[str, Any],
        attention_weights: Optional[Dict[str, float]] = None
    ) -> Explanation:
        """Generate a decision explanation for a specific agent.

        Args:
            agent_id: Identifier of the agent making the decision.
            decision: The decision/action taken.
            context: Context information including state, observations, etc.
            attention_weights: Optional attention weights from the model.

        Returns:
            Explanation object with decision rationale.
        """
        evidence = {}

        # Get attention-based evidence if available
        if attention_weights and self._attention_explainer:
            evidence["attention"] = self._attention_explainer.analyze(
                attention_weights, context
            )

        # Get feature importance if available
        if self._feature_importance:
            evidence["feature_importance"] = self._feature_importance.compute(
                context, decision
            )

        explanation = Explanation(
            explanation_type=ExplanationType.DECISION,
            agent_id=agent_id,
            decision=decision,
            description=self._generate_decision_description(
                agent_id, decision, context, evidence
            ),
            confidence=self._compute_confidence(context, evidence),
            evidence=evidence,
            timestamp=context.get("timestamp", 0.0)
        )

        self.explanation_history.append(explanation)
        return explanation

    def explain_safety_violation(
        self,
        agent_id: str,
        decision: str,
        constraint_name: str,
        violation_details: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Explanation:
        """Generate explanation for a safety constraint violation.

        Args:
            agent_id: Identifier of the violating agent.
            decision: The decision that caused the violation.
            constraint_name: Name of the violated constraint.
            violation_details: Details about the violation.
            context: Context at the time of violation.

        Returns:
            Explanation object describing the safety violation.
        """
        evidence = {
            "constraint": constraint_name,
            "violation_details": violation_details,
            "decision": decision,
        }

        # Get counterfactual explanation
        if self._counterfactual_explainer:
            evidence["counterfactual"] = self._counterfactual_explainer.generate(
                context, decision, constraint_name
            )

        description = (
            f"Agent {agent_id} violated constraint '{constraint_name}' "
            f"by taking action '{decision}'. "
            f"Value exceeded threshold: {violation_details}"
        )

        explanation = Explanation(
            explanation_type=ExplanationType.SAFETY,
            agent_id=agent_id,
            decision=decision,
            description=description,
            confidence=1.0,
            evidence=evidence,
            timestamp=context.get("timestamp", 0.0)
        )

        self.explanation_history.append(explanation)
        return explanation

    def explain_reward_shaping(
        self,
        agent_id: str,
        original_reward: float,
        shaped_reward: float,
        shaping_factors: Dict[str, float],
        context: Dict[str, Any]
    ) -> Explanation:
        """Generate explanation for reward shaping applied to an agent.

        Args:
            agent_id: Identifier of the agent.
            original_reward: Original reward before shaping.
            shaped_reward: Final reward after shaping.
            shaping_factors: Dictionary of factor names and their contributions.
            context: Context information.

        Returns:
            Explanation object describing reward shaping.
        """
        evidence = {
            "original_reward": original_reward,
            "shaped_reward": shaped_reward,
            "shaping_factors": shaping_factors,
            "delta": shaped_reward - original_reward
        }

        factors_desc = ", ".join(
            f"{k}={v:.2f}" for k, v in shaping_factors.items()
        )
        description = (
            f"Reward for agent {agent_id} shaped from {original_reward:.2f} "
            f"to {shaped_reward:.2f}. Factors: {factors_desc}"
        )

        explanation = Explanation(
            explanation_type=ExplanationType.REWARD,
            agent_id=agent_id,
            decision=context.get("action", "N/A"),
            description=description,
            confidence=1.0,
            evidence=evidence,
            timestamp=context.get("timestamp", 0.0)
        )

        self.explanation_history.append(explanation)
        return explanation

    def generate_multi_agent_explanation(
        self,
        explanations: List[Explanation],
        human_feedback: Optional[str] = None
    ) -> MultiAgentExplanation:
        """Aggregate multiple agent explanations into a unified explanation.

        Args:
            explanations: List of individual agent explanations.
            human_feedback: Optional human feedback on the decision.

        Returns:
            MultiAgentExplanation with consensus analysis.
        """
        # Compute consensus score based on decision agreement
        decisions = [e.decision for e in explanations]
        if decisions:
            most_common = max(set(decisions), key=decisions.count)
            consensus_score = decisions.count(most_common) / len(decisions)
        else:
            consensus_score = 0.0

        # Identify disagreement points
        disagreement_points = []
        for exp in explanations:
            if exp.decision != most_common:
                disagreement_points.append({
                    "agent_id": exp.agent_id,
                    "decision": exp.decision,
                    "description": exp.description,
                    "confidence": exp.confidence
                })

        ma_explanation = MultiAgentExplanation(
            explanations=explanations,
            consensus_score=consensus_score,
            disagreement_points=disagreement_points,
            human_feedback=human_feedback
        )

        self.multi_agent_explanations.append(ma_explanation)
        return ma_explanation

    def get_counterfactual_explanation(
        self,
        agent_id: str,
        decision: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get counterfactual explanation for what would happen with a different decision.

        Args:
            agent_id: Agent identifier.
            decision: The actual decision taken.
            context: Current context.

        Returns:
            Counterfactual explanation dictionary or None if unavailable.
        """
        if self._counterfactual_explainer:
            return self._counterfactual_explainer.generate(context, decision)
        return None

    def get_feature_importance(
        self,
        context: Dict[str, Any],
        decision: str
    ) -> Optional[Dict[str, float]]:
        """Get feature importance scores for a decision.

        Args:
            context: Context/state features.
            decision: The decision to explain.

        Returns:
            Dictionary mapping features to importance scores.
        """
        if self._feature_importance:
            return self._feature_importance.compute(context, decision)
        return None

    def get_attention_explanation(
        self,
        attention_weights: Dict[str, float],
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get attention-based explanation.

        Args:
            attention_weights: Attention weight distribution.
            context: Context information.

        Returns:
            Attention explanation dictionary.
        """
        if self._attention_explainer:
            return self._attention_explainer.analyze(attention_weights, context)
        return None

    def get_natural_language_explanation(
        self,
        explanation: Explanation
    ) -> str:
        """Convert an explanation to natural language.

        Args:
            explanation: Explanation object to convert.

        Returns:
            Natural language description of the explanation.
        """
        if self._nl_explainer:
            return self._nl_explainer.explain(explanation)
        return explanation.description

    def _generate_decision_description(
        self,
        agent_id: str,
        decision: str,
        context: Dict[str, Any],
        evidence: Dict[str, Any]
    ) -> str:
        """Generate a description for a decision based on evidence.

        Args:
            agent_id: Agent making the decision.
            decision: The decision taken.
            context: Context information.
            evidence: Evidence supporting the decision.

        Returns:
            Human-readable description.
        """
        state_info = context.get("state_summary", "unknown state")
        return (
            f"Agent {agent_id} chose '{decision}' in state {state_info}. "
            f"This decision was influenced by factors identified through "
            f"attention and feature importance analysis."
        )

    def _compute_confidence(
        self,
        context: Dict[str, Any],
        evidence: Dict[str, Any]
    ) -> float:
        """Compute confidence score for an explanation.

        Args:
            context: Context information.
            evidence: Evidence supporting the explanation.

        Returns:
            Confidence score between 0 and 1.
        """
        # Base confidence from context
        base_confidence = context.get("confidence", 0.5)

        # Adjust based on evidence availability
        evidence_bonus = 0.0
        if "attention" in evidence:
            evidence_bonus += 0.2
        if "feature_importance" in evidence:
            evidence_bonus += 0.2
        if "counterfactual" in evidence:
            evidence_bonus += 0.1

        return min(1.0, base_confidence + evidence_bonus)

    def get_explanation_history(
        self,
        agent_id: Optional[str] = None,
        explanation_type: Optional[ExplanationType] = None
    ) -> List[Explanation]:
        """Get filtered explanation history.

        Args:
            agent_id: Filter by agent ID.
            explanation_type: Filter by explanation type.

        Returns:
            List of matching explanations.
        """
        results = self.explanation_history

        if agent_id:
            results = [e for e in results if e.agent_id == agent_id]
        if explanation_type:
            results = [e for e in results if e.explanation_type == explanation_type]

        return results

    def clear_history(self):
        """Clear all explanation history."""
        self.explanation_history.clear()
        self.multi_agent_explanations.clear()
