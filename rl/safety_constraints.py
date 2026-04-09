"""Safety Constraints for Safe Agentic LLM operations.
Defines and enforces safety boundaries for decision making.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SafetyConstraint:
    """A single safety constraint."""
    name: str
    description: str
    threshold: float
    violation_penalty: float
    is_hard: bool = True


class SafetyConstraints:
    """
    Manages safety constraints for the multi-agent system.
    Ensures all agent actions remain within safe boundaries.
    """

    def __init__(self):
        self.constraints: List[SafetyConstraint] = []
        self.violation_log: List[Dict] = []
        self._init_default_constraints()

    def _init_default_constraints(self):
        """Initialize default safety constraints."""
        self.constraints = [
            SafetyConstraint(
                name='max_response_time',
                description='Emergency response must not exceed time limit',
                threshold=300.0,
                violation_penalty=10.0,
                is_hard=True
            ),
            SafetyConstraint(
                name='min_vehicle_availability',
                description='Minimum vehicles must be available for dispatch',
                threshold=0.3,
                violation_penalty=5.0,
                is_hard=True
            ),
            SafetyConstraint(
                name='max_traffic_density',
                description='Avoid routing through extremely dense traffic',
                threshold=0.9,
                violation_penalty=3.0,
                is_hard=False
            ),
            SafetyConstraint(
                name='min_confidence_threshold',
                description='Decisions must meet minimum confidence',
                threshold=0.5,
                violation_penalty=2.0,
                is_hard=False
            ),
            SafetyConstraint(
                name='max_concurrent_incidents',
                description='Limit on simultaneously handled incidents',
                threshold=10.0,
                violation_penalty=5.0,
                is_hard=True
            ),
        ]

    def add_constraint(self, constraint: SafetyConstraint):
        """Add a new safety constraint."""
        self.constraints.append(constraint)

    def check_constraints(
        self,
        state: Dict[str, Any],
        action: Any
    ) -> Tuple[bool, List[Dict]]:
        """Check all constraints against current state and action."""
        violations = []
        all_satisfied = True

        for constraint in self.constraints:
            violated, value = self._check_single_constraint(constraint, state, action)
            if violated:
                all_satisfied = False
                violations.append({
                    'constraint': constraint.name,
                    'description': constraint.description,
                    'threshold': constraint.threshold,
                    'actual_value': value,
                    'is_hard': constraint.is_hard,
                    'penalty': constraint.violation_penalty
                })
                self.violation_log.append(violations[-1])

        return all_satisfied, violations

    def _check_single_constraint(
        self,
        constraint: SafetyConstraint,
        state: Dict[str, Any],
        action: Any
    ) -> Tuple[bool, float]:
        """Check a single constraint."""
        name = constraint.name

        if name == 'max_response_time':
            value = state.get('response_time', 0)
            return value > constraint.threshold, value

        elif name == 'min_vehicle_availability':
            value = state.get('vehicle_availability', 0)
            return value < constraint.threshold, value

        elif name == 'max_traffic_density':
            value = state.get('traffic_density', 0)
            return value > constraint.threshold, value

        elif name == 'min_confidence_threshold':
            value = state.get('confidence', 1.0)
            return value < constraint.threshold, value

        elif name == 'max_concurrent_incidents':
            value = state.get('concurrent_incidents', 0)
            return value > constraint.threshold, value

        return False, 0.0

    def get_safe_actions(
        self,
        state: Dict[str, Any],
        all_actions: List[Any]
    ) -> List[Any]:
        """Filter actions to only safe ones."""
        safe_actions = []
        for action in all_actions:
            satisfied, _ = self.check_constraints(state, action)
            if satisfied:
                safe_actions.append(action)
        return safe_actions

    def compute_safety_score(self, state: Dict[str, Any]) -> float:
        """Compute an overall safety score (0 = unsafe, 1 = safe)."""
        if not self.constraints:
            return 1.0

        scores = []
        for constraint in self.constraints:
            violated, value = self._check_single_constraint(constraint, state, None)
            if violated:
                if constraint.is_hard:
                    scores.append(0.0)
                else:
                    # Soft constraint - compute degree of violation
                    margin = abs(value - constraint.threshold) / max(abs(constraint.threshold), 1e-6)
                    scores.append(max(0.0, 1.0 - margin))
            else:
                scores.append(1.0)

        return float(np.mean(scores))

    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of all constraint violations."""
        if not self.violation_log:
            return {'total_violations': 0}

        by_constraint = {}
        for v in self.violation_log:
            name = v['constraint']
            if name not in by_constraint:
                by_constraint[name] = {'count': 0, 'hard': 0, 'soft': 0}
            by_constraint[name]['count'] += 1
            if v['is_hard']:
                by_constraint[name]['hard'] += 1
            else:
                by_constraint[name]['soft'] += 1

        return {
            'total_violations': len(self.violation_log),
            'by_constraint': by_constraint,
            'hard_violations': sum(v['is_hard'] for v in self.violation_log),
            'soft_violations': len(self.violation_log) - sum(v['is_hard'] for v in self.violation_log)
        }

    def clear_violation_log(self):
        """Clear the violation log."""
        self.violation_log = []
