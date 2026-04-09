"""Explainable AI (XAI) module for the multi-agent framework.

Provides attention-based explanations, counterfactual reasoning,
feature importance analysis, natural language explanations, and
dashboard monitoring capabilities.
"""

from .explanation_engine import ExplanationEngine
from .attention_explainer import AttentionExplainer
from .counterfactual_explainer import CounterfactualExplainer
from .feature_importance import FeatureImportance
from .natural_language import NaturalLanguageExplainer
from .dashboard import DashboardMonitor, DashboardVisualizer, MetricsPanel

__all__ = [
    'ExplanationEngine',
    'AttentionExplainer',
    'CounterfactualExplainer',
    'FeatureImportance',
    'NaturalLanguageExplainer',
    'DashboardMonitor',
    'DashboardVisualizer',
    'MetricsPanel'
]
