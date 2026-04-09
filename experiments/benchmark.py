from typing import Dict, List, Any, Optional
from datetime import datetime
import time


class ExperimentBenchmark:
    """Benchmark suite for running framework experiments."""

    def __init__(self, name: str = "default_benchmark"):
        self.name = name
        self.results: List[Dict[str, Any]] = []
        self.config: Dict[str, Any] = {
            "num_episodes": 100,
            "max_steps": 50,
            "safety_threshold": 0.8,
            "consensus_threshold": 0.7
        }

    def set_config(self, key: str, value: Any):
        """Set a benchmark configuration option."""
        self.config[key] = value

    def get_config(self, key: str, default=None):
        """Get a benchmark configuration option."""
        return self.config.get(key, default)

    def run_episode(self, env, agent, episode_id: int) -> Dict[str, Any]:
        """Run a single episode and collect results."""
        start_time = time.time()
        state = env.reset()
        total_reward = 0
        safety_violations = 0
        steps = 0
        actions_taken = []

        for step in range(self.config.get("max_steps", 50)):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            safety_violations += info.get("safety_violation", 0)
            actions_taken.append(action)
            state = next_state
            steps += 1
            if done:
                break

        episode_time = time.time() - start_time
        result = {
            "episode_id": episode_id,
            "total_reward": total_reward,
            "safety_violations": safety_violations,
            "steps": steps,
            "actions": actions_taken,
            "duration": episode_time,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        return result

    def run_benchmark(self, env, agent, num_episodes: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run the full benchmark suite."""
        num_episodes = num_episodes or self.config.get("num_episodes", 100)
        self.results = []
        start_time = time.time()

        for i in range(num_episodes):
            self.run_episode(env, agent, i)
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_episodes} episodes")

        total_time = time.time() - start_time
        print(f"Benchmark completed in {total_time:.2f} seconds")
        return self.results

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate benchmark statistics."""
        if not self.results:
            return {}

        rewards = [r["total_reward"] for r in self.results]
        violations = [r["safety_violations"] for r in self.results]
        steps = [r["steps"] for r in self.results]
        durations = [r["duration"] for r in self.results]

        return {
            "num_episodes": len(self.results),
            "avg_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "std_reward": self._std(rewards),
            "total_violations": sum(violations),
            "avg_violations_per_episode": sum(violations) / len(violations),
            "avg_steps": sum(steps) / len(steps),
            "avg_duration": sum(durations) / len(durations),
            "safety_rate": 1 - (sum(violations) / (len(self.results) * self.config.get("max_steps", 50)))
        }

    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def compare_benchmarks(self, other: "ExperimentBenchmark") -> Dict[str, Any]:
        """Compare this benchmark with another."""
        stats1 = self.get_statistics()
        stats2 = other.get_statistics()

        return {
            "benchmark1": {"name": self.name, "stats": stats1},
            "benchmark2": {"name": other.name, "stats": stats2},
            "reward_diff": stats1.get("avg_reward", 0) - stats2.get("avg_reward", 0),
            "safety_diff": stats1.get("safety_rate", 0) - stats2.get("safety_rate", 0),
            "winner": self.name if stats1.get("avg_reward", 0) > stats2.get("avg_reward", 0) else other.name
        }

    def export_results(self, filename: str):
        """Export results to a file."""
        import json
        with open(filename, "w") as f:
            json.dump({
                "benchmark_name": self.name,
                "config": self.config,
                "statistics": self.get_statistics(),
                "results": self.results
            }, f, indent=2, default=str)
        print(f"Results exported to {filename}")

    def load_results(self, filename: str):
        """Load results from a file."""
        import json
        with open(filename, "r") as f:
            data = json.load(f)
            self.name = data.get("benchmark_name", self.name)
            self.config.update(data.get("config", {}))
            self.results = data.get("results", [])
        print(f"Results loaded from {filename}")
