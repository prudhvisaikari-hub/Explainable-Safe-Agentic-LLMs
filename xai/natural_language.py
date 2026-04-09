"""Natural language explanation module for the multi-agent framework.

Converts technical explanations into human-readable natural language
descriptions suitable for human-in-the-loop decision making.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum


class ExplanationStyle(Enum):
    """Styles for natural language explanations."""
    TECHNICAL = "technical"
    CONCISE = "concise"
    DETAILED = "detailed"
    EXECUTIVE = "executive"
    EDUCATIONAL = "educational"


class NaturalLanguageExplainer:
    """Generates natural language explanations from technical data.

    Converts structured explanation objects, feature importance scores,
    attention weights, and counterfactual scenarios into human-readable
    text suitable for different audiences and purposes.
    """

    def __init__(self, style: ExplanationStyle = ExplanationStyle.DETAILED):
        """Initialize the natural language explainer.

        Args:
            style: The style of explanation to generate.
        """
        self.style = style
        self.templates = self._load_templates()

    def explain(self, explanation: Any) -> str:
        """Convert an explanation object to natural language.

        Args:
            explanation: An explanation object (Explanation, MultiAgentExplanation,
                        FeatureImportanceResult, CounterfactualScenario, etc.)

        Returns:
            Natural language explanation string.
        """
        exp_type = type(explanation).__name__

        if exp_type == "Explanation":
            return self._explain_single(explanation)
        elif exp_type == "MultiAgentExplanation":
            return self._explain_multi_agent(explanation)
        elif exp_type == "FeatureImportanceResult":
            return self._explain_feature_importance(explanation)
        elif exp_type == "CounterfactualScenario":
            return self._explain_counterfactual(explanation)
        elif exp_type == "AttentionMap":
            return self._explain_attention(explanation)
        else:
            return str(explanation)

    def _explain_single(self, exp) -> str:
        """Explain a single agent explanation."""
        style = self.style

        if style == ExplanationStyle.CONCISE:
            return f"Agent {exp.agent_id} chose '{exp.decision}' with {exp.confidence:.0%} confidence."

        elif style == ExplanationStyle.TECHNICAL:
            parts = [f"[Agent: {exp.agent_id}] [Action: {exp.decision}]"]
            parts.append(f"[Confidence: {exp.confidence:.2f}]")
            parts.append(f"[Type: {exp.explanation_type.value}]")

            if exp.evidence:
                parts.append(f"[Evidence keys: {list(exp.evidence.keys())}]")

            return " ".join(parts)

        elif style == ExplanationStyle.EXECUTIVE:
            action_desc = self._describe_action(exp.decision)
            confidence_desc = "high" if exp.confidence > 0.7 else "moderate" if exp.confidence > 0.4 else "low"
            return f"The system {action_desc} with {confidence_desc} confidence based on current risk assessment."

        else:  # DETAILED or EDUCATIONAL
            parts = []

            # Opening
            parts.append(f"Agent {exp.agent_id} decided to take the action '{exp.decision}'.")

            # Confidence
            conf_text = "high" if exp.confidence > 0.7 else "moderate" if exp.confidence > 0.4 else "low"
            parts.append(f"The confidence in this decision is {conf_text} ({exp.confidence:.0%}).")

            # Type-specific explanation
            if exp.explanation_type.value == "decision":
                parts.append(self._explain_decision_rationale(exp))
            elif exp.explanation_type.value == "safety":
                parts.append(self._explain_safety_rationale(exp))
            elif exp.explanation_type.value == "reward":
                parts.append(self._explain_reward_rationale(exp))

            # Evidence
            if exp.evidence:
                parts.append(self._summarize_evidence(exp.evidence))

            return " ".join(parts)

    def _explain_multi_agent(self, exp) -> str:
        """Explain a multi-agent explanation."""
        n_agents = len(exp.explanations)
        consensus = exp.consensus_score

        if self.style == ExplanationStyle.CONCISE:
            return f"{n_agents} agents participated. Consensus: {consensus:.0%}."

        elif self.style == ExplanationStyle.EXECUTIVE:
            consensus_desc = "strong" if consensus > 0.8 else "moderate" if consensus > 0.5 else "weak"
            decisions = [e.decision for e in exp.explanations]
            most_common = max(set(decisions), key=decisions.count) if decisions else "none"
            return f"Multi-agent consensus ({consensus_desc}, {consensus:.0%}) favors '{most_common}'."

        parts = []
        parts.append(f"A total of {n_agents} agents participated in the decision-making process.")

        # Consensus
        consensus_desc = "strong" if consensus > 0.8 else "moderate" if consensus > 0.5 else "weak"
        parts.append(f"The consensus among agents is {consensus_desc} ({consensus:.0%}).")

        # Individual decisions
        decisions = {}
        for e in exp.explanations:
            decisions[e.decision] = decisions.get(e.decision, 0) + 1

        decision_summary = ", ".join([f"'{d}' ({c} agent{'s' if c > 1 else ''})" for d, c in decisions.items()])
        parts.append(f"Decisions were: {decision_summary}.")

        # Disagreements
        if exp.disagreement_points:
            parts.append(f"There were {len(exp.disagreement_points)} points of disagreement among agents.")
            if self.style == ExplanationStyle.DETAILED:
                for dp in exp.disagreement_points[:2]:
                    parts.append(f"  - Agent {dp['agent_id']} chose '{dp['decision']}' with {dp['confidence']:.0%} confidence.")

        # Human feedback
        if exp.human_feedback:
            parts.append(f"Human feedback received: '{exp.human_feedback}'")

        return " ".join(parts)

    def _explain_feature_importance(self, result) -> str:
        """Explain feature importance results."""
        top_features = result.top_features[:5]

        if self.style == ExplanationStyle.CONCISE:
            top_names = [f[0] for f in top_features[:3]]
            return f"Key factors: {', '.join(top_names)}."

        parts = []
        parts.append(f"Using the {result.method} method, the following features most influenced the decision:")

        for i, (feature, score) in enumerate(top_features, 1):
            importance = "very important" if score > 0.2 else "important" if score > 0.1 else "somewhat important"
            parts.append(f"  {i}. {feature} ({importance}, {score:.1%})")

        if self.style == ExplanationStyle.EDUCATIONAL:
            parts.append("")
            parts.append("Feature importance helps understand which aspects of the situation")
            parts.append("the agent considered most when making its decision.")

        return " ".join(parts)

    def _explain_counterfactual(self, scenario) -> str:
        """Explain a counterfactual scenario."""
        changes = scenario.minimal_changes
        original = scenario.original_action
        counterfactual = scenario.counterfactual_action

        if self.style == ExplanationStyle.CONCISE:
            return f"If conditions changed, the agent would choose '{counterfactual}' instead of '{original}'."

        parts = []

        # Describe the changes
        if changes:
            change_descriptions = []
            for c in changes[:3]:
                change_descriptions.append(f"{c[0]} changed from {c[1]:.2f} to {c[2]:.2f}")
            parts.append(f"Consider a scenario where {', '.join(change_descriptions)}." )

        # Describe the outcome
        parts.append(f"In this alternative scenario, the agent would have chosen '{counterfactual}' instead of '{original}'.")

        # Outcome impact
        outcome = scenario.outcome_change
        if outcome.get("reward_change", 0) != 0:
            reward_dir = "improved" if outcome["reward_change"] > 0 else "decreased"
            parts.append(f"This would have {reward_dir} the reward by {abs(outcome['reward_change']):.2f}.")

        if outcome.get("safety_change", 0) != 0:
            safety_dir = "improved" if outcome["safety_change"] > 0 else "decreased"
            parts.append(f"Safety would have {safety_dir} by {abs(outcome['safety_change']):.2f}.")

        # Feasibility
        feasibility = scenario.feasibility
        parts.append(f"This scenario has {feasibility} feasibility (plausibility: {scenario.plausibility_score:.0%}).")

        return " ".join(parts)

    def _explain_attention(self, attention_map) -> str:
        """Explain an attention map."""
        top = attention_map.top_attended[:5]

        if self.style == ExplanationStyle.CONCISE:
            top_names = [t[0] for t in top[:3]]
            return f"Model attended most to: {', '.join(top_names)}."

        parts = []
        parts.append("The model's attention was distributed as follows:")

        for i, (token, weight) in enumerate(top, 1):
            focus = "primary focus" if weight > 0.2 else "secondary focus" if weight > 0.1 else "notable attention"
            parts.append(f"  {i}. '{token}' ({focus}, {weight:.1%})")

        return " ".join(parts)

    def _explain_decision_rationale(self, exp) -> str:
        """Explain the rationale behind a decision."""
        if exp.evidence.get("attention"):
            att = exp.evidence["attention"]
            top = att.get("top_attended_tokens", [])[:3]
            if top:
                return f"This choice was primarily influenced by attention to {', '.join([t[0] for t in top])}."

        if exp.evidence.get("feature_importance"):
            fi = exp.evidence["feature_importance"]
            top = fi.top_features[:3] if hasattr(fi, 'top_features') else []
            if top:
                return f"The key factors in this decision were {', '.join([t[0] for t in top])}."

        return exp.description

    def _explain_safety_rationale(self, exp) -> str:
        """Explain a safety-related rationale."""
        evidence = exp.evidence
        constraint = evidence.get("constraint", "unknown constraint")
        violation = evidence.get("violation_details", {})

        parts = []
        parts.append(f"A safety constraint '{constraint}' was violated.")

        if violation:
            parts.append(f"Violation details: {violation}.")

        if evidence.get("counterfactual"):
            parts.append("A counterfactual analysis suggests alternative safe actions were available.")

        return " ".join(parts)

    def _explain_reward_rationale(self, exp) -> str:
        """Explain a reward-related rationale."""
        evidence = exp.evidence
        original = evidence.get("original_reward", 0)
        shaped = evidence.get("shaped_reward", 0)
        factors = evidence.get("shaping_factors", {})

        delta = shaped - original
        direction = "increased" if delta > 0 else "decreased"

        parts = []
        parts.append(f"The reward was {direction} from {original:.2f} to {shaped:.2f} (delta: {delta:+.2f}).")

        if factors:
            factor_str = ", ".join([f"{k} ({v:+.2f})" for k, v in factors.items()])
            parts.append(f"Shaping factors: {factor_str}.")

        return " ".join(parts)

    def _summarize_evidence(self, evidence: Dict[str, Any]) -> str:
        """Summarize available evidence."""
        parts = []

        if "attention" in evidence:
            parts.append("Attention analysis was performed.")
        if "feature_importance" in evidence:
            parts.append("Feature importance was computed.")
        if "counterfactual" in evidence:
            parts.append("Counterfactual scenarios were generated.")

        if parts:
            return " " + " ".join(parts)
        return ""

    def _describe_action(self, action: str) -> str:
        """Provide a human-friendly description of an action."""
        descriptions = {
            "aggressive_action": "took an aggressive approach",
            "moderate_action": "took a moderate approach",
            "conservative_action": "took a conservative approach",
            "wait_and_observe": "chose to wait and observe",
            "proceed": "proceeded with the task",
            "halt": "halted the operation",
            "retry": "attempted to retry",
            "escalate": "escalated the decision to a higher authority"
        }
        return descriptions.get(action, f"chose to '{action}'")

    def _load_templates(self) -> Dict[str, str]:
        """Load explanation templates."""
        return {
            "decision": "Agent {agent} chose {action} with {confidence} confidence.",
            "safety": "Safety constraint {constraint} was violated by {action}.",
            "reward": "Reward shaped from {original} to {shaped}.",
            "multi_agent": "{n} agents reached {consensus} consensus on {decision}.",
        }

    def set_style(self, style: ExplanationStyle):
        """Set the explanation style."""
        self.style = style

    def batch_explain(self, explanations: List[Any]) -> List[str]:
        """Explain multiple objects at once."""
        return [self.explain(exp) for exp in explanations]
