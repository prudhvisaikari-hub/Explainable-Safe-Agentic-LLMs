"""Attention-based explanation module for the multi-agent framework.

Provides attention weight analysis, token importance visualization,
and attention-based rationale extraction for model decisions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


@dataclass
class AttentionHead:
    """Represents a single attention head's weights."""
    layer: int
    head: int
    weights: Dict[str, float]
    importance_score: float = 0.0


@dataclass
class AttentionMap:
    """Represents an attention map for a decision."""
    query_tokens: List[str]
    key_tokens: List[str]
    attention_matrix: np.ndarray
    aggregated_weights: Dict[str, float] = field(default_factory=dict)
    top_attended: List[Tuple[str, float]] = field(default_factory=list)


class AttentionExplainer:
    """Explains model decisions through attention weight analysis.

    Analyzes attention weights from transformer models to identify
    which input tokens or features most influenced a decision.
    Supports multi-head attention analysis and cross-agent attention.
    """

    def __init__(self, top_k: int = 10, threshold: float = 0.05):
        """Initialize the attention explainer.

        Args:
            top_k: Number of top attended tokens to return.
            threshold: Minimum attention weight to consider.
        """
        self.top_k = top_k
        self.threshold = threshold
        self.attention_maps: List[AttentionMap] = []
        self.attention_heads: List[AttentionHead] = []

    def analyze(
        self,
        attention_weights: Dict[str, float],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze attention weights and generate explanation.

        Args:
            attention_weights: Dictionary mapping tokens/features to weights.
            context: Additional context for the analysis.

        Returns:
            Dictionary with attention analysis results.
        """
        # Normalize weights
        total = sum(attention_weights.values())
        if total > 0:
            normalized = {k: v / total for k, v in attention_weights.items()}
        else:
            normalized = attention_weights

        # Get top-k attended tokens
        sorted_weights = sorted(
            normalized.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.top_k]

        # Filter by threshold
        above_threshold = [
            (k, v) for k, v in sorted_weights if v >= self.threshold
        ]

        # Categorize tokens
        categories = self._categorize_tokens(
            [k for k, v in above_threshold],
            context
        )

        # Compute attention entropy
        weights_array = np.array([v for v in normalized.values()])
        weights_array = weights_array[weights_array > 0]
        entropy = -np.sum(weights_array * np.log2(weights_array)) if len(weights_array) > 0 else 0

        return {
            "top_attended_tokens": sorted_weights,
            "above_threshold": above_threshold,
            "categories": categories,
            "total_attention": sum(normalized.values()),
            "attention_entropy": float(entropy),
            "num_significant_tokens": len(above_threshold),
            "analysis_summary": self._generate_summary(sorted_weights, categories)
        }

    def build_attention_map(
        self,
        query_tokens: List[str],
        key_tokens: List[str],
        attention_matrix: np.ndarray
    ) -> AttentionMap:
        """Build an attention map from query-key attention matrix.

        Args:
            query_tokens: List of query tokens.
            key_tokens: List of key tokens.
            attention_matrix: 2D numpy array of attention weights.

        Returns:
            AttentionMap object with aggregated analysis.
        """
        # Aggregate attention per key token
        aggregated = {}
        for j, key in enumerate(key_tokens):
            aggregated[key] = float(np.mean(attention_matrix[:, j]))

        # Get top attended key tokens
        top_attended = sorted(
            aggregated.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.top_k]

        attention_map = AttentionMap(
            query_tokens=query_tokens,
            key_tokens=key_tokens,
            attention_matrix=attention_matrix,
            aggregated_weights=aggregated,
            top_attended=top_attended
        )

        self.attention_maps.append(attention_map)
        return attention_map

    def analyze_multi_head_attention(
        self,
        attention_weights: Dict[int, Dict[int, Dict[str, float]]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze multi-head attention across layers.

        Args:
            attention_weights: Nested dict: layer -> head -> token_weights.
            context: Additional context.

        Returns:
            Analysis results aggregated across heads.
        """
        head_results = []

        for layer_idx, heads in attention_weights.items():
            for head_idx, weights in heads.items():
                result = self.analyze(weights, context)
                head = AttentionHead(
                    layer=layer_idx,
                    head=head_idx,
                    weights=weights,
                    importance_score=result.get("total_attention", 0)
                )
                self.attention_heads.append(head)
                head_results.append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "analysis": result
                })

        # Aggregate across heads
        all_tokens = {}
        for hr in head_results:
            for token, weight in hr["analysis"].get("top_attended_tokens", []):
                all_tokens[token] = all_tokens.get(token, 0) + weight

        return {
            "per_head_analysis": head_results,
            "aggregated_tokens": sorted(
                all_tokens.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.top_k],
            "num_layers": len(attention_weights),
            "num_heads": sum(len(h) for h in attention_weights.values())
        }

    def compare_attention(
        self,
        weights_before: Dict[str, float],
        weights_after: Dict[str, float],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare attention weights before and after an intervention.

        Args:
            weights_before: Attention weights before intervention.
            weights_after: Attention weights after intervention.
            context: Context information.

        Returns:
            Comparison analysis results.
        """
        before_analysis = self.analyze(weights_before, context)
        after_analysis = self.analyze(weights_after, context)

        # Compute changes
        changes = []
        all_tokens = set(weights_before.keys()) | set(weights_after.keys())

        for token in all_tokens:
            before_w = weights_before.get(token, 0)
            after_w = weights_after.get(token, 0)
            change = after_w - before_w
            if abs(change) > self.threshold:
                changes.append({
                    "token": token,
                    "before": before_w,
                    "after": after_w,
                    "change": change,
                    "direction": "increased" if change > 0 else "decreased"
                })

        changes.sort(key=lambda x: abs(x["change"]), reverse=True)

        return {
            "before": before_analysis,
            "after": after_analysis,
            "changes": changes[:self.top_k],
            "num_increased": sum(1 for c in changes if c["change"] > 0),
            "num_decreased": sum(1 for c in changes if c["change"] < 0)
        }

    def get_attention_rationale(
        self,
        attention_weights: Dict[str, float],
        decision: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate a natural language rationale based on attention.

        Args:
            attention_weights: Attention weight distribution.
            decision: The decision being explained.
            context: Context information.

        Returns:
            Natural language rationale string.
        """
        analysis = self.analyze(attention_weights, context)
        top_tokens = analysis.get("top_attended_tokens", [])[:5]

        if not top_tokens:
            return "No significant attention patterns detected."

        token_list = ", ".join([f"'{t[0]}'" for t in top_tokens])
        return (
            f"The decision to '{decision}' was primarily influenced by "
            f"attention to the following elements: {token_list}. "
            f"These factors collectively received {analysis['total_attention']:.1%} "
            f"of the model's attention."
        )

    def _categorize_tokens(
        self,
        tokens: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Categorize tokens into semantic groups.

        Args:
            tokens: List of tokens to categorize.
            context: Context for categorization.

        Returns:
            Dictionary mapping categories to token lists.
        """
        categories = {
            "safety_features": [],
            "reward_features": [],
            "state_features": [],
            "other": []
        }

        safety_keywords = ["risk", "safe", "danger", "violation", "constraint"]
        reward_keywords = ["reward", "penalty", "bonus", "value"]
        state_keywords = ["state", "observation", "position", "status"]

        for token in tokens:
            token_lower = token.lower()
            if any(k in token_lower for k in safety_keywords):
                categories["safety_features"].append(token)
            elif any(k in token_lower for k in reward_keywords):
                categories["reward_features"].append(token)
            elif any(k in token_lower for k in state_keywords):
                categories["state_features"].append(token)
            else:
                categories["other"].append(token)

        return categories

    def _generate_summary(
        self,
        top_tokens: List[Tuple[str, float]],
        categories: Dict[str, List[str]]
    ) -> str:
        """Generate a summary of the attention analysis.

        Args:
            top_tokens: Top attended tokens with weights.
            categories: Categorized tokens.

        Returns:
            Summary string.
        """
        if not top_tokens:
            return "No significant attention patterns found."

        parts = []
        for cat_name, tokens in categories.items():
            if tokens:
                parts.append(f"{len(tokens)} {cat_name.replace('_', ' ')}")

        if not parts:
            parts = [f"{len(top_tokens)} general features"]

        return f"Attention focused on: {', '.join(parts)}"

    def clear_history(self):
        """Clear stored attention maps and heads."""
        self.attention_maps.clear()
        self.attention_heads.clear()
