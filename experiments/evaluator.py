from typing import Dict, List, Any, Optional, Tuple
from enum import Enum


class EvaluationMetric(Enum):
    REWARD = "reward"
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    CONSISTENCY = "consistency"
    EXPLAINABILITY = "explainability"


class ExperimentEvaluator:
    """Evaluator for assessing framework performance across metrics."""

    def __init__(self):
        self.weights: Dict[EvaluationMetric, float] = {
            EvaluationMetric.REWARD: 0.25,
            EvaluationMetric.SAFETY: 0.30,
            EvaluationMetric.EFFICIENCY: 0.20,
            EvaluationMetric.CONSISTENCY: 0.15,
            EvaluationMetric.EXPLAINABILITY: 0.10
        }
        self.results: List[Dict[str, Any]] = []

    def set_weight(self, metric: EvaluationMetric, weight: float):
        """Set the weight for a metric."""
        self.weights[metric] = weight
        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total

    def evaluate_reward(self, benchmark_results: List[Dict]) -> float:
        """Evaluate reward metric."""
        if not benchmark_results:
            return 0.0
        rewards = [r.get("total_reward", 0) for r in benchmark_results]
        avg = sum(rewards) / len(rewards)
        max_possible = max(abs(r) for r in rewards) * 2 if rewards else 1
        return min(1.0, max(0.0, (avg + max_possible) / (2 * max_possible)))

    def evaluate_safety(self, benchmark_results: List[Dict], max_steps: int = 50) -> float:
        """Evaluate safety metric."""
        if not benchmark_results:
            return 0.0
        total_violations = sum(r.get("safety_violations", 0) for r in benchmark_results)
        total_steps = len(benchmark_results) * max_steps
        safety_rate = 1 - (total_violations / total_steps)
        return min(1.0, max(0.0, safety_rate))

    def evaluate_efficiency(self, benchmark_results: List[Dict]) -> float:
        """Evaluate efficiency metric."""
        if not benchmark_results:
            return 0.0
        steps = [r.get("steps", 0) for r in benchmark_results]
        avg_steps = sum(steps) / len(steps)
        max_steps = max(steps) if steps else 1
        return 1 - (avg_steps / max_steps)

    def evaluate_consistency(self, benchmark_results: List[Dict]) -> float:
        """Evaluate consistency metric."""
        if len(benchmark_results) < 2:
            return 1.0
        rewards = [r.get("total_reward", 0) for r in benchmark_results]
        mean = sum(rewards) / len(rewards)
        variance = sum((x - mean) ** 2 for x in rewards) / len(rewards)
        std = variance ** 0.5
        return max(0.0, 1 - (std / (abs(mean) + 1)))

    def evaluate_explainability(self, xai_results: Optional[List[Dict]] = None) -> float:
        """Evaluate explainability metric."""
        if not xai_results:
            return 0.5
        if not xai_results:
            return 0.5
        quality_scores = [r.get("explanation_quality", 0.5) for r in xai_results]
        return sum(quality_scores) / len(quality_scores)

    def compute_overall_score(self, benchmark_results: List[Dict],
                              xai_results: Optional[List[Dict]] = None) -> Dict[str, float]:
        """Compute overall evaluation score."""
        scores = {
            EvaluationMetric.REWARD: self.evaluate_reward(benchmark_results),
            EvaluationMetric.SAFETY: self.evaluate_safety(benchmark_results),
            EvaluationMetric.EFFICIENCY: self.evaluate_efficiency(benchmark_results),
            EvaluationMetric.CONSISTENCY: self.evaluate_consistency(benchmark_results),
            EvaluationMetric.EXPLAINABILITY: self.evaluate_explainability(xai_results)
        }
        overall = sum(self.weights[m] * scores[m] for m in scores)
        scores["overall"] = overall
        return scores

    def add_evaluation(self, evaluation: Dict[str, Any]):
        """Add an evaluation result."""
        evaluation["timestamp"] = evaluation.get("timestamp", "")
        self.results.append(evaluation)

    def get_evaluation(self, index: int) -> Optional[Dict[str, Any]]:
        """Get an evaluation by index."""
        if 0 <= index < len(self.results):
            return self.results[index]
        return None

    def get_all_evaluations(self) -> List[Dict[str, Any]]:
        """Get all evaluations."""
        return self.results.copy()

    def get_best_evaluation(self, metric: str = "overall") -> Optional[Dict[str, Any]]:
        """Get the best evaluation by a metric."""
        if not self.results:
            return None
        return max(self.results, key=lambda x: x.get(metric, 0))

    def compare_evaluations(self, eval1: Dict, eval2: Dict) -> Dict[str, Any]:
        """Compare two evaluations."""
        return {
            "eval1": eval1,
            "eval2": eval2,
            "overall_diff": eval1.get("overall", 0) - eval2.get("overall", 0),
            "better": "eval1" if eval1.get("overall", 0) > eval2.get("overall", 0) else "eval2"
        }

    def generate_report(self) -> str:
        """Generate a text report of evaluations."""
        if not self.results:
            return "No evaluations to report."

        report = ["Experiment Evaluation Report", "=" * 30]
        for i, ev in enumerate(self.results):
            report.append(f"\nEvaluation {i + 1}:")
            for k, v in ev.items():
                report.append(f"  {k}: {v}")
        return "\n".join(report)

    def clear_evaluations(self):
        """Clear all evaluations."""
        self.results.clear()
