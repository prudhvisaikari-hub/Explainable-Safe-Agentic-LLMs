from typing import Dict, List, Any
from datetime import datetime
import threading
import time


class DashboardMonitor:
    """Real-time monitoring component for the multi-agent framework."""

    def __init__(self):
        self.metrics_history: Dict[str, List[tuple]] = {
            "reward": [],
            "safety_violations": [],
            "agent_decisions": [],
            "xai_queries": [],
            "consensus_score": []
        }
        self.current_state: Dict[str, Any] = {}
        self.running = False
        self._lock = threading.Lock()
        self._callbacks: List[callable] = []

    def start(self, interval: float = 1.0):
        """Start the monitoring loop."""
        self.running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def stop(self):
        """Stop the monitoring loop."""
        self.running = False
        if hasattr(self, "_monitor_thread"):
            self._monitor_thread.join(timeout=2.0)

    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self.running:
            self._collect_metrics()
            self._notify_callbacks()
            time.sleep(interval)

    def _collect_metrics(self):
        """Collect current system metrics."""
        with self._lock:
            timestamp = datetime.now().isoformat()
            for key in self.metrics_history:
                value = self.current_state.get(key, 0)
                self.metrics_history[key].append((timestamp, value))

    def _notify_callbacks(self):
        """Notify registered callbacks of state changes."""
        for callback in self._callbacks:
            try:
                callback(self.current_state.copy())
            except Exception as e:
                print(f"Callback error: {e}")

    def register_callback(self, callback: callable):
        """Register a callback for state updates."""
        self._callbacks.append(callback)

    def unregister_callback(self, callback: callable):
        """Unregister a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def update_state(self, **kwargs):
        """Update the current monitored state."""
        with self._lock:
            self.current_state.update(kwargs)

    def get_metrics(self, key: str, limit: int = 100) -> List[tuple]:
        """Get recent metrics for a specific key."""
        with self._lock:
            return self.metrics_history.get(key, [])[-limit:]

    def get_all_metrics(self) -> Dict[str, List[tuple]]:
        """Get all current metrics."""
        with self._lock:
            return {k: v.copy() for k, v in self.metrics_history.items()}

    def clear_metrics(self):
        """Clear all collected metrics."""
        with self._lock:
            for key in self.metrics_history:
                self.metrics_history[key] = []

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current monitoring state."""
        with self._lock:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "state": self.current_state.copy(),
                "metrics_count": {k: len(v) for k, v in self.metrics_history.items()},
                "is_running": self.running
            }
            return summary
