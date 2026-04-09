"""Explainable-Safe-Agentic-LLMs: Multi-Agent Framework for Risk-Aware,
Human-In-The-Loop Decision-Making with Explainable and Safe Agentic LLMs.

This package provides a comprehensive multi-agent framework combining:
- Safe Reinforcement Learning agents with risk-aware decision making
- Explainable AI (XAI) for transparent agent reasoning
- Real-time monitoring dashboards
- Benchmarking and evaluation tools

Modules:
- agents: Multi-agent decision making (SafeAgent, HumanAgent, ConsensusAgent)
- env: Environment interfaces (SmartCityEnv)
- rl: Safe RL components (SafeRLAgent, reward shaping, safety constraints)
- xai: Explainability tools (explanation engine, attention, counterfactual, NLP)
- dashboard: Real-time monitoring and visualization
- experiments: Benchmarking and evaluation

Example:
    from agents import SafeAgent, HumanAgent
    from env import SmartCityEnv
    from rl import SafeRLAgent
    from xai import ExplanationEngine
    from dashboard import DashboardMonitor
    from experiments import ExperimentBenchmark, ExperimentEvaluator

For more information, see: https://github.com/prudhvisaikari-hub/Explainable-Safe-Agentic-LLMs
"""

__version__ = "0.1.0"
__author__ = "Prudhvi Saikari"
__license__ = "MIT"
