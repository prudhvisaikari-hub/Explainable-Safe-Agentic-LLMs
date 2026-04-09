"""Smart City Emergency Response Simulation Environment
Gymnasium-compatible environment for multi-agent emergency vehicle allocation
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class Incident:
    id: int
    severity: float
    location: Tuple[float, float]
    type: str
    demographic_group: str
    created_at: int
    resolved: bool = False

@dataclass
class Vehicle:
    id: int
    type: str
    location: Tuple[float, float]
    available: bool = True
    current_assignment: Optional[int] = None

class SmartCityEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(self, n_incidents=8, n_vehicles=4, grid_size=10,
                 max_steps=50, risk_lambda=0.3, fairness_mu=0.2):
        super().__init__()
        self.n_incidents = n_incidents
        self.n_vehicles = n_vehicles
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.risk_lambda = risk_lambda
        self.fairness_mu = fairness_mu
        self.demographic_groups = ['group_A', 'group_B', 'group_C']
        self.incident_types = ['fire', 'medical', 'accident', 'hazard']
        self.vehicle_types = ['ambulance', 'fire_truck', 'police', 'multipurpose']
        self.state_dim = (n_incidents * 4) + (n_vehicles * 3) + 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([n_incidents + 1] * n_vehicles)
        self.current_step = 0
        self.incidents = []
        self.vehicles = []
        self.history = []
        self.rewards_history = []
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.incidents = []
        self.vehicles = []
        self.history = []
        self.rewards_history = []
        for i in range(self.n_incidents):
            incident = Incident(
                id=i,
                severity=np.random.uniform(0.1, 1.0),
                location=(np.random.uniform(0, self.grid_size),
                         np.random.uniform(0, self.grid_size)),
                type=np.random.choice(self.incident_types),
                demographic_group=np.random.choice(self.demographic_groups),
                created_at=0
            )
            self.incidents.append(incident)
        for i in range(self.n_vehicles):
            vehicle = Vehicle(
                id=i,
                type=self.vehicle_types[i % len(self.vehicle_types)],
                location=(np.random.uniform(0, self.grid_size),
                         np.random.uniform(0, self.grid_size)),
                available=True
            )
            self.vehicles.append(vehicle)
        return self._get_state(), self._get_info()
    
    def _get_state(self):
        state = []
        for inc in self.incidents:
            state.append(inc.severity)
            state.append(self.incident_types.index(inc.type) / len(self.incident_types))
            state.append(inc.location[0] / self.grid_size)
            state.append(inc.location[1] / self.grid_size)
        for veh in self.vehicles:
            state.append(1.0 if veh.available else 0.0)
            state.append(veh.location[0] / self.grid_size)
            state.append(veh.location[1] / self.grid_size)
        severities = [inc.severity for inc in self.incidents if not inc.resolved]
        if len(severities) > 1:
            risk = np.var(severities)
        else:
            risk = 0.0
        state.append(risk)
        state.append(np.mean(severities) if severities else 0.0)
        state.append(np.max(severities) if severities else 0.0)
        demo_counts = {g: 0 for g in self.demographic_groups}
        for inc in self.incidents:
            if not inc.resolved:
                demo_counts[inc.demographic_group] += 1
        for g in self.demographic_groups:
            state.append(demo_counts[g] / max(1, len(self.incidents)))
        state.append((self.max_steps - self.current_step) / self.max_steps)
        return np.array(state, dtype=np.float32)
    
    def _get_info(self):
        severities = [inc.severity for inc in self.incidents if not inc.resolved]
        unresolved = [inc for inc in self.incidents if not inc.resolved]
        demo_served = {g: 0 for g in self.demographic_groups}
        demo_total = {g: 0 for g in self.demographic_groups}
        for inc in self.incidents:
            demo_total[inc.demographic_group] += 1
            if inc.resolved:
                demo_served[inc.demographic_group] += 1
        fairness_gaps = {}
        for g in self.demographic_groups:
            if demo_total[g] > 0:
                fairness_gaps[g] = demo_served[g] / demo_total[g]
            else:
                fairness_gaps[g] = 0.0
        return {
            'unresolved_count': len(unresolved),
            'total_incidents': len(self.incidents),
            'avg_severity': np.mean(severities) if severities else 0.0,
            'max_severity': np.max(severities) if severities else 0.0,
            'risk_variance': np.var(severities) if len(severities) > 1 else 0.0,
            'fairness_gaps': fairness_gaps,
            'demographic_served': demo_served,
            'demographic_total': demo_total,
            'vehicles_available': sum(1 for v in self.vehicles if v.available),
            'step': self.current_step,
            'incidents': [{'id': i.id, 'severity': i.severity, 'type': i.type,
                          'demographic': i.demographic_group, 'resolved': i.resolved}
                         for i in self.incidents],
            'vehicles': [{'id': v.id, 'type': v.type, 'available': v.available,
                         'assignment': v.current_assignment} for v in self.vehicles]
        }
    
    def step(self, action):
        self.current_step += 1
        for veh in self.vehicles:
            veh.available = True
            veh.current_assignment = None
        assignments = []
        for veh_idx, target in enumerate(action):
            if target < self.n_incidents:
                inc = self.incidents[target]
                if not inc.resolved:
                    veh = self.vehicles[veh_idx]
                    veh.available = False
                    veh.current_assignment = inc.id
                    assignments.append((veh_idx, target))
            else:
                assignments.append((veh_idx, -1))
        resolved_this_step = []
        for veh_idx, inc_idx in assignments:
            if inc_idx >= 0:
                inc = self.incidents[inc_idx]
                if not inc.resolved:
                    type_match = 1.0 if self._vehicle_type_match(
                        self.vehicles[veh_idx].type, inc.type) else 0.5
                    resolution_prob = min(1.0, type_match * inc.severity * 0.8)
                    if np.random.random() < resolution_prob:
                        inc.resolved = True
                        resolved_this_step.append(inc_idx)
        reward = self._calculate_reward(resolved_this_step, assignments)
        info = self._get_info()
        self.history.append(info)
        self.rewards_history.append(reward)
        all_resolved = all(inc.resolved for inc in self.incidents)
        terminated = all_resolved
        truncated = self.current_step >= self.max_steps
        return self._get_state(), reward, terminated, truncated, info
    
    def _vehicle_type_match(self, vehicle_type, incident_type):
        matches = {
            'ambulance': ['medical', 'accident'],
            'fire_truck': ['fire', 'hazard'],
            'police': ['accident', 'hazard'],
            'multipurpose': ['fire', 'medical', 'accident', 'hazard']
        }
        return incident_type in matches.get(vehicle_type, [])
    
    def _calculate_reward(self, resolved, assignments):
        task_reward = 0.0
        for inc_idx in resolved:
            inc = self.incidents[inc_idx]
            task_reward += inc.severity * 10.0
        remaining_severities = [inc.severity for inc in self.incidents if not inc.resolved]
        if len(remaining_severities) > 1:
            risk_penalty = self.risk_lambda * np.var(remaining_severities) * 5.0
        else:
            risk_penalty = 0.0
        demo_served = {g: 0 for g in self.demographic_groups}
        demo_total = {g: 0 for g in self.demographic_groups}
        for inc in self.incidents:
            demo_total[inc.demographic_group] += 1
            if inc.resolved:
                demo_served[inc.demographic_group] += 1
        fairness_penalty = 0.0
        for g in self.demographic_groups:
            if demo_total[g] > 0:
                rate = demo_served[g] / demo_total[g]
                avg_rate = sum(demo_served.values()) / max(1, sum(demo_total.values()))
                fairness_penalty += self.fairness_mu * abs(rate - avg_rate) * 5.0
        step_penalty = 0.1
        reward = task_reward - risk_penalty - fairness_penalty - step_penalty
        return reward
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step {self.current_step}: Unresolved={sum(1 for i in self.incidents if not i.resolved)}")
            for inc in self.incidents:
                status = "[RESOLVED]" if inc.resolved else f"[SEV:{inc.severity:.2f}]"
                print(f"  Inc {inc.id}: {status} {inc.type}")
        return ""
