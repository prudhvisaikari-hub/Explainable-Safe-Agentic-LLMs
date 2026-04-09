from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class ReportGenerator:
    """Generator for experiment reports and summaries."""

    def __init__(self, title: str = "Experiment Report"):
        self.title = title
        self.sections: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            "author": "",
            "date": datetime.now().isoformat(),
            "version": "1.0"
        }

    def set_metadata(self, key: str, value: Any):
        """Set a metadata field."""
        self.metadata[key] = value

    def add_section(self, title: str, content: str, section_type: str = "text"):
        """Add a section to the report."""
        self.sections.append({
            "title": title,
            "content": content,
            "type": section_type,
            "order": len(self.sections)
        })

    def add_metrics_section(self, metrics: Dict[str, Any], title: str = "Performance Metrics"):
        """Add a metrics section."""
        content = self._format_metrics(metrics)
        self.add_section(title, content, "metrics")

    def add_table_section(self, headers: List[str], rows: List[List[Any]],
                          title: str = "Data Table"):
        """Add a table section."""
        table_data = {"headers": headers, "rows": rows}
        self.sections.append({
            "title": title,
            "content": table_data,
            "type": "table",
            "order": len(self.sections)
        })

    def add_chart_section(self, chart_data: Dict[str, Any], title: str = "Chart"):
        """Add a chart/visualization section."""
        self.sections.append({
            "title": title,
            "content": chart_data,
            "type": "chart",
            "order": len(self.sections)
        })

    def add_conclusion(self, text: str):
        """Add a conclusion section."""
        self.add_section("Conclusion", text, "conclusion")

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as text."""
        lines = []
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def generate_text_report(self) -> str:
        """Generate a text-based report."""
        lines = [
            "=" * 60,
            self.title.upper(),
            "=" * 60,
            "",
            f"Generated: {self.metadata.get('date', 'N/A')}",
            f"Author: {self.metadata.get('author', 'N/A')}",
            f"Version: {self.metadata.get('version', 'N/A')}",
            "",
            "-" * 60
        ]

        for section in sorted(self.sections, key=lambda x: x["order"]):
            lines.append(f"\n{section['title'].upper()}")
            lines.append("-" * len(section["title"]))
            if isinstance(section["content"], str):
                lines.append(section["content"])
            elif isinstance(section["content"], dict):
                if "headers" in section["content"]:
                    header = section["content"]["headers"]
                    lines.append(" | ".join(str(h) for h in header))
                    lines.append("-" * (len(header) * 15))
                    for row in section["content"]["rows"]:
                        lines.append(" | ".join(str(c) for c in row))
                else:
                    for k, v in section["content"].items():
                        lines.append(f"  {k}: {v}")
            lines.append("")

        return "\n".join(lines)

    def generate_json_report(self) -> str:
        """Generate a JSON-based report."""
        return json.dumps({
            "title": self.title,
            "metadata": self.metadata,
            "sections": self.sections
        }, indent=2, default=str)

    def export_text(self, filename: str):
        """Export report as text file."""
        with open(filename, "w") as f:
            f.write(self.generate_text_report())
        print(f"Text report exported to {filename}")

    def export_json(self, filename: str):
        """Export report as JSON file."""
        with open(filename, "w") as f:
            f.write(self.generate_json_report())
        print(f"JSON report exported to {filename}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the report."""
        return {
            "title": self.title,
            "num_sections": len(self.sections),
            "section_types": list(set(s["type"] for s in self.sections)),
            "metadata": self.metadata
        }

    def clear(self):
        """Clear all sections."""
        self.sections.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "title": self.title,
            "metadata": self.metadata,
            "sections": self.sections.copy()
        }
