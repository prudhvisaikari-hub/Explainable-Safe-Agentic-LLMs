"""Explainable AI (XAI) module for the multi-agent framework.

Provides attention-based explanations, counterfactual reasoning,
feature importance analysis, and natural language explanations.
"""

from .explanation_engine import ExplanationEngine
from .attention_explainer import AttentionExplainer
from .counterfactual_explainer import CounterfactualExplainer
from .feature_importance import FeatureImportance
from .natural_language import NaturalLanguageExplainer

__all__ = [
    'ExplanationEngine',
    'AttentionExplainer',
    'CounterfactualExplainer',
    'FeatureImportance',
    'NaturalLanguageExplainer',
]
