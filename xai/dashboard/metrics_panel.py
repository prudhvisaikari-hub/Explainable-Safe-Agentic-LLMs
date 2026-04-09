from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricThreshold:
    """Defines thresholds for metric alerts."""

    def __init__(self, metric_name: str, min_val: Optional[float] = None,
                 max_val: Optional[float] = None, alert_level: AlertLevel = AlertLevel.WARNING):
        self.metric_name = metric_name
        self.min_val = min_val
        self.max_val = max_val
        self.alert_level = alert_level

    def check(self, value: float) -> Optional[str]:
        """Check if a value violates thresholds."""
        if self.min_val is not None and value < self.min_val:
            return f"{self.metric_name} below minimum ({value} < {self.min_val})"
        if self.max_val is not None and value > self.max_val:
            return f"{self.metric_name} above maximum ({value} > {self.max_val})"
        return None


class MetricsPanel:
    """Panel component for displaying and managing metric widgets."""

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.thresholds: Dict[str, MetricThreshold] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.widgets: List[Dict[str, Any]] = []

    def add_metric(self, name: str, value: float, unit: str = "",
                   description: str = "", widget_type: str = "gauge"):
        """Add or update a metric."""
        self.metrics[name] = {
            "value": value,
            "unit": unit,
            "description": description,
            "widget_type": widget_type,
            "updated_at": datetime.now().isoformat()
        }
        self._check_thresholds(name, value)

    def update_metric(self, name: str, value: float):
        """Update an existing metric value."""
        if name in self.metrics:
            self.metrics[name]["value"] = value
            self.metrics[name]["updated_at"] = datetime.now().isoformat()
            self._check_thresholds(name, value)

    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a metric by name."""
        return self.metrics.get(name)

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics."""
        return self.metrics.copy()

    def remove_metric(self, name: str):
        """Remove a metric."""
        if name in self.metrics:
            del self.metrics[name]

    def add_threshold(self, threshold: MetricThreshold):
        """Add a threshold rule."""
        self.thresholds[threshold.metric_name] = threshold

    def remove_threshold(self, metric_name: str):
        """Remove a threshold rule."""
        if metric_name in self.thresholds:
            del self.thresholds[metric_name]

    def _check_thresholds(self, metric_name: str, value: float):
        """Check all thresholds for a metric."""
        if metric_name in self.thresholds:
            threshold = self.thresholds[metric_name]
            violation = threshold.check(value)
            if violation:
                self._add_alert(metric_name, violation, threshold.alert_level)

    def _add_alert(self, metric_name: str, message: str, level: AlertLevel):
        """Add an alert."""
        alert = {
            "metric": metric_name,
            "message": message,
            "level": level.value,
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False
        }
        self.alerts.append(alert)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

    def acknowledge_alert(self, index: int):
        """Acknowledge an alert by index."""
        if 0 <= index < len(self.alerts):
            self.alerts[index]["acknowledged"] = True

    def get_alerts(self, level: Optional[AlertLevel] = None,
                   acknowledged: bool = False) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered."""
        alerts = self.alerts
        if level:
            alerts = [a for a in alerts if a["level"] == level.value]
        if not acknowledged:
            alerts = [a for a in alerts if not a["acknowledged"]]
        return alerts

    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()

    def add_widget(self, widget_config: Dict[str, Any]):
        """Add a widget to the panel."""
        self.widgets.append(widget_config)

    def remove_widget(self, index: int):
        """Remove a widget by index."""
        if 0 <= index < len(self.widgets):
            del self.widgets[index]

    def get_widgets(self) -> List[Dict[str, Any]]:
        """Get all widgets."""
        return self.widgets.copy()

    def get_panel_data(self) -> Dict[str, Any]:
        """Get complete panel data for rendering."""
        return {
            "metrics": self.metrics.copy(),
            "alerts": [a for a in self.alerts if not a["acknowledged"]],
            "widgets": self.widgets.copy(),
            "threshold_count": len(self.thresholds),
            "alert_count": len([a for a in self.alerts if not a["acknowledged"]])
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert panel to dictionary."""
        return self.get_panel_data()
