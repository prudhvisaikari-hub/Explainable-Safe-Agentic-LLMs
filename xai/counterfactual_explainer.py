"""Counterfactual explanation module for the multi-agent framework.

Generates what-if scenarios and counterfactual reasoning
for agent decisions, safety violations, and policy changes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import copy


@dataclass
class CounterfactualScenario:
    """Represents a single counterfactual scenario."""
    original_state: Dict[str, Any]
    modified_state: Dict[str, Any]
    original_action: str
    counterfactual_action: str
    outcome_change: Dict[str, float]
    minimal_changes: List[Tuple[str, Any, Any]]
    plausibility_score: float = 0.0
    feasibility: str = "unknown"


class CounterfactualExplainer:
    """Generates counterfactual explanations for agent decisions.

    Explores alternative states and actions to explain why a particular
    decision was made. Supports minimal-change counterfactuals and
    constraint-based what-if analysis.
    """

    def __init__(
        self,
        max_scenarios: int = 5,
        min_change_threshold: float = 0.1
    ):
        """Initialize the counterfactual explainer.

        Args:
            max_scenarios: Maximum number of scenarios to generate.
            min_change_threshold: Minimum state change to consider significant.
        """
        self.max_scenarios = max_scenarios
        self.min_change_threshold = min_change_threshold
        self.scenarios: List[CounterfactualScenario] = []

    def generate(
        self,
        state: Dict[str, Any],
        action: str,
        constraint_name: Optional[str] = None,
        env: Optional[Any] = None
    ) -> List[CounterfactualScenario]:
        """Generate counterfactual scenarios for a decision.

        Args:
            state: Current state of the environment.
            action: The action taken.
            constraint_name: Optional constraint that was violated.
            env: Optional environment for simulating outcomes.

        Returns:
            List of CounterfactualScenario objects.
        """
        scenarios = []

        # Scenario 1: What if risk was lower?
        if "risk_level" in state:
            scenarios.append(self._generate_risk_reduction_scenario(
                state, action, constraint_name, env
            ))

        # Scenario 2: What if the agent had more information?
        scenarios.append(self._generate_information_scenario(
            state, action, constraint_name, env
        ))

        # Scenario 3: What if a different constraint applied?
        if constraint_name:
            scenarios.append(self._generate_constraint_relaxation_scenario(
                state, action, constraint_name, env
            ))

        # Scenario 4: What if the state was slightly different?
        scenarios.append(self._generate_state_perturbation_scenario(
            state, action, constraint_name, env
        ))

        # Scenario 5: What if a different agent made the decision?
        scenarios.append(self._generate_agent_swap_scenario(
            state, action, constraint_name, env
        ))

        # Filter and rank scenarios
        valid_scenarios = [
            s for s in scenarios
            if s is not None and s.counterfactual_action != action
        ]

        # Rank by plausibility and minimal change
        valid_scenarios.sort(key=lambda s: -s.plausibility_score)

        self.scenarios.extend(valid_scenarios[:self.max_scenarios])
        return valid_scenarios[:self.max_scenarios]

    def _generate_risk_reduction_scenario(
        self,
        state: Dict[str, Any],
        action: str,
        constraint_name: Optional[str],
        env: Optional[Any]
    ) -> Optional[CounterfactualScenario]:
        """Generate scenario where risk is reduced."""
        modified_state = copy.deepcopy(state)
        changes = []

        # Reduce risk level
        if "risk_level" in modified_state:
            old_risk = modified_state["risk_level"]
            new_risk = max(0, old_risk * 0.5)
            changes.append(("risk_level", old_risk, new_risk))
            modified_state["risk_level"] = new_risk

        # Reduce uncertainty
        if "uncertainty" in modified_state:
            old_unc = modified_state["uncertainty"]
            new_unc = max(0, old_unc * 0.5)
            changes.append(("uncertainty", old_unc, new_unc))
            modified_state["uncertainty"] = new_unc

        if not changes:
            return None

        # Simulate counterfactual action
        cf_action = self._simulate_action(modified_state, env)

        # Compute outcome change
        outcome_change = self._compute_outcome_change(
            state, modified_state, action, cf_action, env
        )

        return CounterfactualScenario(
            original_state=state,
            modified_state=modified_state,
            original_action=action,
            counterfactual_action=cf_action,
            outcome_change=outcome_change,
            minimal_changes=changes,
            plausibility_score=0.7,
            feasibility="high"
        )

    def _generate_information_scenario(
        self,
        state: Dict[str, Any],
        action: str,
        constraint_name: Optional[str],
        env: Optional[Any]
    ) -> Optional[CounterfactualScenario]:
        """Generate scenario with more information."""
        modified_state = copy.deepcopy(state)
        changes = []

        # Increase observation completeness
        if "observation_completeness" in modified_state:
            old_val = modified_state["observation_completeness"]
            new_val = min(1.0, old_val + 0.3)
            changes.append(("observation_completeness", old_val, new_val))
            modified_state["observation_completeness"] = new_val
        else:
            modified_state["observation_completeness"] = 0.8
            changes.append(("observation_completeness", 0.5, 0.8))

        # Reduce noise
        if "noise_level" in modified_state:
            old_val = modified_state["noise_level"]
            new_val = max(0, old_val * 0.5)
            changes.append(("noise_level", old_val, new_val))
            modified_state["noise_level"] = new_val

        cf_action = self._simulate_action(modified_state, env)
        outcome_change = self._compute_outcome_change(
            state, modified_state, action, cf_action, env
        )

        return CounterfactualScenario(
            original_state=state,
            modified_state=modified_state,
            original_action=action,
            counterfactual_action=cf_action,
            outcome_change=outcome_change,
            minimal_changes=changes,
            plausibility_score=0.6,
            feasibility="medium"
        )

    def _generate_constraint_relaxation_scenario(
        self,
        state: Dict[str, Any],
        action: str,
        constraint_name: str,
        env: Optional[Any]
    ) -> Optional[CounterfactualScenario]:
        """Generate scenario with relaxed constraint."""
        modified_state = copy.deepcopy(state)
        changes = [
            ("constraint_relaxed", False, True),
            ("constraint_name", constraint_name, None)
        ]
        modified_state["constraint_relaxed"] = True

        cf_action = self._simulate_action(modified_state, env)
        outcome_change = self._compute_outcome_change(
            state, modified_state, action, cf_action, env
        )

        return CounterfactualScenario(
            original_state=state,
            modified_state=modified_state,
            original_action=action,
            counterfactual_action=cf_action,
            outcome_change=outcome_change,
            minimal_changes=changes,
            plausibility_score=0.5,
            feasibility="low"
        )

    def _generate_state_perturbation_scenario(
        self,
        state: Dict[str, Any],
        action: str,
        constraint_name: Optional[str],
        env: Optional[Any]
    ) -> Optional[CounterfactualScenario]:
        """Generate scenario with small state perturbations."""
        modified_state = copy.deepcopy(state)
        changes = []

        # Perturb numeric features
        for key, value in state.items():
            if isinstance(value, (int, float)) and key not in ["timestamp"]:
                import random
                perturbation = random.uniform(-0.1, 0.1) * abs(value) if value != 0 else 0.1
                if abs(perturbation) >= self.min_change_threshold:
                    old_val = modified_state[key]
                    new_val = old_val + perturbation
                    changes.append((key, old_val, new_val))
                    modified_state[key] = new_val

        if not changes:
            changes = [("state_perturbation", 0, 0.1)]

        cf_action = self._simulate_action(modified_state, env)
        outcome_change = self._compute_outcome_change(
            state, modified_state, action, cf_action, env
        )

        return CounterfactualScenario(
            original_state=state,
            modified_state=modified_state,
            original_action=action,
            counterfactual_action=cf_action,
            outcome_change=outcome_change,
            minimal_changes=changes,
            plausibility_score=0.8,
            feasibility="high"
        )

    def _generate_agent_swap_scenario(
        self,
        state: Dict[str, Any],
        action: str,
        constraint_name: Optional[str],
        env: Optional[Any]
    ) -> Optional[CounterfactualScenario]:
        """Generate scenario with a different agent making the decision."""
        modified_state = copy.deepcopy(state)
        changes = [
            ("decision_maker", state.get("agent_id", "agent_0"), "alternative_agent")
        ]
        modified_state["agent_id"] = "alternative_agent"
        modified_state["risk_tolerance"] = state.get("risk_tolerance", 0.5) + 0.2

        cf_action = self._simulate_action(modified_state, env)
        outcome_change = self._compute_outcome_change(
            state, modified_state, action, cf_action, env
        )

        return CounterfactualScenario(
            original_state=state,
            modified_state=modified_state,
            original_action=action,
            counterfactual_action=cf_action,
            outcome_change=outcome_change,
            minimal_changes=changes,
            plausibility_score=0.4,
            feasibility="medium"
        )

    def _simulate_action(
        self,
        state: Dict[str, Any],
        env: Optional[Any]
    ) -> str:
        """Simulate what action would be taken in modified state."""
        risk = state.get("risk_level", 0.5)
        uncertainty = state.get("uncertainty", 0.5)
        reward_potential = state.get("reward_potential", 0.5)

        # Simple heuristic for counterfactual action
        if risk < 0.3 and reward_potential > 0.7:
            return "aggressive_action"
        elif risk > 0.7 or uncertainty > 0.7:
            return "conservative_action"
        elif reward_potential > 0.5:
            return "moderate_action"
        else:
            return "wait_and_observe"

    def _compute_outcome_change(
        self,
        original_state: Dict[str, Any],
        modified_state: Dict[str, Any],
        original_action: str,
        counterfactual_action: str,
        env: Optional[Any]
    ) -> Dict[str, float]:
        """Compute the change in outcomes between original and counterfactual."""
        # Simulate reward difference
        risk_orig = original_state.get("risk_level", 0.5)
        risk_cf = modified_state.get("risk_level", 0.5)

        reward_orig = self._estimate_reward(original_state, original_action)
        reward_cf = self._estimate_reward(modified_state, counterfactual_action)

        safety_orig = 1.0 - risk_orig
        safety_cf = 1.0 - risk_cf

        return {
            "reward_change": reward_cf - reward_orig,
            "safety_change": safety_cf - safety_orig,
            "risk_change": risk_cf - risk_orig,
            "action_changed": 1.0 if original_action != counterfactual_action else 0.0
        }

    def _estimate_reward(
        self,
        state: Dict[str, Any],
        action: str
    ) -> float:
        """Estimate the reward for a state-action pair."""
        base_reward = state.get("reward_potential", 0.5)
        risk_penalty = state.get("risk_level", 0.5) * 0.3

        action_multipliers = {
            "aggressive_action": 1.2,
            "moderate_action": 1.0,
            "conservative_action": 0.8,
            "wait_and_observe": 0.5
        }

        multiplier = action_multipliers.get(action, 1.0)
        return max(0, min(1, (base_reward * multiplier) - risk_penalty))

    def get_explanation_text(self, scenario: CounterfactualScenario) -> str:
        """Convert a counterfactual scenario to natural language."""
        changes_desc = "; ".join(
            f"{c[0]} changed from {c[1]:.2f} to {c[2]:.2f}"
            for c in scenario.minimal_changes
        )

        return (
            f"If {changes_desc}, the agent would have chosen "
            f"'{scenario.counterfactual_action}' instead of "
            f"'{scenario.original_action}'. "
            f"This would result in a reward change of "
            f"{scenario.outcome_change.get('reward_change', 0):.2f} "
            f"and a safety change of "
            f"{scenario.outcome_change.get('safety_change', 0):.2f}."
        )

    def clear_history(self):
        """Clear stored scenarios."""
        self.scenarios.clear()
