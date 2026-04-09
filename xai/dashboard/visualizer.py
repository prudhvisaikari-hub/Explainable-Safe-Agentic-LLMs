from typing import Dict, List, Tuple, Optional
import json


class DashboardVisualizer:
    """Visualization component for dashboard data rendering."""

    def __init__(self):
        self.data_buffer: Dict[str, List] = {}
        self.config: Dict[str, any] = {
            "chart_type": "line",
            "update_interval": 1000,
            "max_data_points": 100
        }

    def set_config(self, key: str, value):
        """Set a visualization configuration option."""
        self.config[key] = value

    def get_config(self, key: str, default=None):
        """Get a visualization configuration option."""
        return self.config.get(key, default)

    def add_data(self, series: str, value: any, timestamp: Optional[float] = None):
        """Add a data point to a visualization series."""
        if series not in self.data_buffer:
            self.data_buffer[series] = []
        self.data_buffer[series].append((timestamp or len(self.data_buffer[series]), value))
        max_points = self.config.get("max_data_points", 100)
        if len(self.data_buffer[series]) > max_points:
            self.data_buffer[series] = self.data_buffer[series][-max_points:]

    def add_batch_data(self, series: str, data: List[Tuple[float, any]]):
        """Add multiple data points at once."""
        if series not in self.data_buffer:
            self.data_buffer[series] = []
        self.data_buffer[series].extend(data)
        max_points = self.config.get("max_data_points", 100)
        if len(self.data_buffer[series]) > max_points:
            self.data_buffer[series] = self.data_buffer[series][-max_points:]

    def get_series(self, series: str) -> List[Tuple[float, any]]:
        """Get all data points for a series."""
        return self.data_buffer.get(series, [])

    def get_all_series(self) -> Dict[str, List[Tuple[float, any]]]:
        """Get all visualization data."""
        return {k: v.copy() for k, v in self.data_buffer.items()}

    def clear_series(self, series: str):
        """Clear data for a specific series."""
        if series in self.data_buffer:
            self.data_buffer[series] = []

    def clear_all(self):
        """Clear all visualization data."""
        self.data_buffer.clear()

    def to_json(self) -> str:
        """Export visualization data as JSON."""
        return json.dumps({
            "config": self.config,
            "data": {k: v for k, v in self.data_buffer.items()}
        }, default=str)

    def from_json(self, json_str: str):
        """Import visualization data from JSON."""
        data = json.loads(json_str)
        if "config" in data:
            self.config.update(data["config"])
        if "data" in data:
            self.data_buffer.update(data["data"])

    def generate_chart_spec(self, series: str, chart_type: Optional[str] = None) -> Dict:
        """Generate a chart specification for rendering."""
        chart_type = chart_type or self.config.get("chart_type", "line")
        data = self.get_series(series)
        labels = [str(x[0]) for x in data]
        values = [x[1] for x in data]
        return {
            "type": chart_type,
            "labels": labels,
            "data": values,
            "title": series,
            "series_name": series
        }

    def generate_multi_series_spec(self, series_list: List[str]) -> Dict:
        """Generate a multi-series chart specification."""
        datasets = []
        labels = set()
        for series in series_list:
            data = self.get_series(series)
            datasets.append({
                "label": series,
                "data": [x[1] for x in data]
            })
            labels.update([str(x[0]) for x in data])
        return {
            "type": "line",
            "labels": sorted(labels),
            "datasets": datasets
        }

    def get_statistics(self, series: str) -> Dict[str, float]:
        """Calculate basic statistics for a series."""
        data = self.get_series(series)
        if not data:
            return {}
        values = [x[1] for x in data if isinstance(x[1], (int, float))]
        if not values:
            return {}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "latest": values[-1]
        }
