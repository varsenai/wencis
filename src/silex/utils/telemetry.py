from __future__ import annotations

import json
from datetime import datetime, timezone
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.sdk.trace.export import ReadableSpan
from silex.utils.config import WORKSPACE_DIR

class JSONLFileSpanExporter(SpanProcessor):
    """Exports OpenTelemetry spans cleanly to a JSONL file to prevent console noise."""
    def __init__(self, filename: str = "telemetry_traces.jsonl"):
        self.filepath = WORKSPACE_DIR / filename

    def on_end(self, span: ReadableSpan) -> None:
        try:
            span_data = {
                "name": span.name,
                "context": {
                    "trace_id": hex(span.context.trace_id),
                    "span_id": hex(span.context.span_id),
                },
                "parent_id": hex(span.parent.span_id) if span.parent else None,
                "start_time": datetime.fromtimestamp(span.start_time / 1e9, timezone.utc).isoformat(),
                "end_time": datetime.fromtimestamp(span.end_time / 1e9, timezone.utc).isoformat(),
                "duration_ms": (span.end_time - span.start_time) / 1e6,
                "attributes": dict(span.attributes),
                "status": span.status.status_code.name,
            }
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(span_data) + "\n")
        except Exception:
            pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

# Initialize Tracer Provider
provider = TracerProvider()
provider.add_span_processor(JSONLFileSpanExporter())
trace.set_tracer_provider(provider)

# Expose tracer
tracer = trace.get_tracer("vyn.telemetry")
