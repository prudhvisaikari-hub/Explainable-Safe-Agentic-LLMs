"""Feature importance module for the multi-agent framework.

Computes feature importance using various methods including
gradient-based, perturbation-based, and model-agnostic approaches.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
import numpy as np


@dataclass
class FeatureImportanceResult:
    """Results of feature importance computation."""
    method: str
    feature_scores: Dict[str, float]
    ranked_features: List[tuple]
    top_features: List[tuple]
    normalization: str
    confidence_interval: Optional[Dict[str, tuple]] = None


class FeatureImportance:
    """Computes feature importance for agent decisions.

    Supports multiple methods for computing how much each feature
    in the state contributes to an agent's decision. Methods include
    gradient-based, perturbation-based, and model-agnostic approaches.
    """

    def __init__(
        self,
        method: str = "perturbation",
        n_samples: int = 100,
        baseline: Optional[Dict[str, float]] = None
    ):
        """Initialize the feature importance calculator.

        Args:
            method: Method for computing importance. Options:
                'perturbation', 'gradient', 'shapley', 'random'
            n_samples: Number of samples for Monte Carlo estimation.
            baseline: Baseline state for comparison.
        """
        self.method = method
        self.n_samples = n_samples
        self.baseline = baseline or {}
        self.results: List[FeatureImportanceResult] = []

    def compute(
        self,
        state: Dict[str, Any],
        decision: str,
        model_fn: Optional[Callable] = None,
        env: Optional[Any] = None
    ) -> FeatureImportanceResult:
        """Compute feature importance for a decision.

        Args:
            state: Current state with features.
            decision: The decision/action taken.
            model_fn: Optional model function for gradient methods.
            env: Optional environment for simulation.

        Returns:
            FeatureImportanceResult with importance scores.
        """
        numeric_features = {
            k: float(v) for k, v in state.items()
            if isinstance(v, (int, float)) and k != "timestamp"
        }

        if self.method == "perturbation":
            scores = self._compute_perturbation_importance(
                numeric_features, decision, env
            )
        elif self.method == "gradient":
            scores = self._compute_gradient_importance(
                numeric_features, decision, model_fn
            )
        elif self.method == "shapley":
            scores = self._compute_shapley_importance(
                numeric_features, decision, env
            )
        else:  # random
            scores = self._compute_random_importance(numeric_features)

        ranked = sorted(scores.items(), key=lambda x: -abs(x[1]))
        top_n = min(10, len(ranked))

        result = FeatureImportanceResult(
            method=self.method,
            feature_scores=scores,
            ranked_features=ranked,
            top_features=ranked[:top_n],
            normalization="relative"
        )

        self.results.append(result)
        return result

    def _compute_perturbation_importance(
        self,
        features: Dict[str, float],
        decision: str,
        env: Optional[Any]
    ) -> Dict[str, float]:
        """Compute importance via feature perturbation."""
        scores = {}
        baseline_reward = self._estimate_reward(features, decision, env)

        for feature, value in features.items():
            # Perturb the feature
            perturbation = 0.1 * abs(value) if value != 0 else 0.1

            # Perturb up
            perturbed_up = features.copy()
            perturbed_up[feature] = value + perturbation
            reward_up = self._estimate_reward(perturbed_up, decision, env)

            # Perturb down
            perturbed_down = features.copy()
            perturbed_down[feature] = value - perturbation
            reward_down = self._estimate_reward(perturbed_down, decision, env)

            # Importance is the average absolute change
            importance = (abs(reward_up - baseline_reward) +
                         abs(reward_down - baseline_reward)) / 2
            scores[feature] = float(importance)

        # Normalize
        total = sum(abs(s) for s in scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores

    def _compute_gradient_importance(
        self,
        features: Dict[str, float],
        decision: str,
        model_fn: Optional[Callable]
    ) -> Dict[str, float]:
        """Compute importance via gradient-based method."""
        scores = {}

        if model_fn is None:
            # Fallback to perturbation if no model function
            return self._compute_perturbation_importance(features, decision, None)

        for feature, value in features.items():
            # Estimate gradient numerically
            epsilon = 1e-5
            features_plus = features.copy()
            features_plus[feature] = value + epsilon
            features_minus = features.copy()
            features_minus[feature] = value - epsilon

            try:
                grad_plus = model_fn(features_plus, decision)
                grad_minus = model_fn(features_minus, decision)
                gradient = (grad_plus - grad_minus) / (2 * epsilon)
                scores[feature] = abs(float(gradient))
            except Exception:
                scores[feature] = 0.0

        # Normalize
        total = sum(abs(s) for s in scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores

    def _compute_shapley_importance(
        self,
        features: Dict[str, float],
        decision: str,
        env: Optional[Any]
    ) -> Dict[str, float]:
        """Compute Shapley values for feature importance."""
        feature_keys = list(features.keys())
        n_features = len(feature_keys)
        scores = {k: 0.0 for k in feature_keys}

        if n_features == 0:
            return scores

        # Simplified Shapley computation using sampling
        for _ in range(min(self.n_samples, 50)):
            # Random ordering of features
            order = np.random.permutation(n_features)

            # Start with empty set
            current_features = {}
            current_reward = self._estimate_reward(current_features, decision, env)

            for idx in order:
                feature = feature_keys[idx]
                # Add feature
                new_features = current_features.copy()
                new_features[feature] = features[feature]
                new_reward = self._estimate_reward(new_features, decision, env)

                # Marginal contribution
                contribution = new_reward - current_reward
                scores[feature] += abs(contribution)

                current_features = new_features
                current_reward = new_reward

        # Average over samples
        for k in scores:
            scores[k] /= min(self.n_samples, 50)

        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores

    def _compute_random_importance(
        self,
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute random importance as baseline."""
        if not features:
            return {}

        n = len(features)
        scores = {k: np.random.random() for k in features}

        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores

    def _estimate_reward(
        self,
        features: Dict[str, float],
        decision: str,
        env: Optional[Any]
    ) -> float:
        """Estimate reward given features and decision."""
        # Use risk and reward features if available
        risk = features.get("risk_level", 0.5)
        reward_potential = features.get("reward_potential", 0.5)
        uncertainty = features.get("uncertainty", 0.5)

        base = reward_potential * (1 - risk * 0.5)
        penalty = uncertainty * 0.2

        return max(0, min(1, base - penalty))

    def compute_group_importance(
        self,
        state: Dict[str, Any],
        decision: str,
        groups: Dict[str, List[str]],
        env: Optional[Any] = None
    ) -> Dict[str, float]:
        """Compute importance for groups of features.

        Args:
            state: Current state.
            decision: The decision taken.
            groups: Dictionary mapping group names to feature lists.
            env: Optional environment.

        Returns:
            Dictionary mapping group names to aggregated importance.
        """
        # First compute individual importance
        individual_result = self.compute(state, decision, env=env)
        individual_scores = individual_result.feature_scores

        # Aggregate by group
        group_scores = {}
        for group_name, features in groups.items():
            group_total = sum(
                individual_scores.get(f, 0.0) for f in features
            )
            group_scores[group_name] = group_total

        # Normalize group scores
        total = sum(group_scores.values())
        if total > 0:
            group_scores = {k: v / total for k, v in group_scores.items()}

        return group_scores

    def compare_importance(
        self,
        state1: Dict[str, Any],
        decision1: str,
        state2: Dict[str, Any],
        decision2: str,
        env: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Compare feature importance between two states."""
        result1 = self.compute(state1, decision1, env=env)
        result2 = self.compute(state2, decision2, env=env)

        scores1 = result1.feature_scores
        scores2 = result2.feature_scores

        # Compute changes
        all_features = set(scores1.keys()) | set(scores2.keys())
        changes = {}
        for f in all_features:
            s1 = scores1.get(f, 0)
            s2 = scores2.get(f, 0)
            changes[f] = {
                "before": s1,
                "after": s2,
                "change": s2 - s1,
                "direction": "increased" if s2 > s1 else "decreased"
            }

        return {
            "result1": result1,
            "result2": result2,
            "changes": changes,
            "top_changes": sorted(
                changes.items(),
                key=lambda x: abs(x[1]["change"]),
                reverse=True
            )[:5]
        }

    def get_summary(self, result: Optional[FeatureImportanceResult] = None) -> str:
        """Get a textual summary of feature importance."""
        if result is None:
            if self.results:
                result = self.results[-1]
            else:
                return "No feature importance results available."

        top_features = result.top_features[:5]
        features_str = ", ".join([f"{f[0]} ({f[1]:.2%})" for f in top_features])

        return (
            f"Top features influencing the decision ({result.method} method): "
            f"{features_str}"
        )

    def clear_history(self):
        """Clear stored results."""
        self.results.clear()
