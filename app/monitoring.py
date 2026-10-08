"""Request logging and Prometheus metrics for the prediction API.

Every request is logged as a single JSON line on stdout, so `docker logs` /
`kubectl logs` can be grepped or shipped to a log store as-is. Metrics are
exposed in Prometheus text format at /metrics.
"""

import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# Paths hit by probes / the scraper every few seconds — counted in metrics but
# not logged, otherwise they drown out real traffic.
QUIET_PATHS = {"/health", "/metrics"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "extra_fields", {}))
        return json.dumps(entry, default=str)


def get_logger(name: str = "heart_api") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


logger = get_logger()


def log_event(msg: str, **fields) -> None:
    logger.info(msg, extra={"extra_fields": fields})


# ------------------------------------------------------------------ metrics
REQUESTS = Counter(
    "heart_api_requests_total",
    "HTTP requests by method, route and status code",
    ["method", "path", "status"],
)
LATENCY = Histogram(
    "heart_api_request_duration_seconds",
    "HTTP request latency by route",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
IN_FLIGHT = Gauge("heart_api_requests_in_progress", "Requests currently being handled")
PREDICTIONS = Counter(
    "heart_api_predictions_total",
    "Predictions served, by outcome and risk level",
    ["label", "risk_level"],
)
PROBABILITY = Histogram(
    "heart_api_probability_disease",
    "Distribution of predicted probability of heart disease",
    buckets=(0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0),
)
MODEL_INFO = Gauge(
    "heart_api_model_info",
    "Currently loaded model (value is always 1)",
    ["model_name", "trained_at"],
)


def route_template(request: Request) -> str:
    """Route pattern (e.g. /predict) instead of the raw URL, to bound label cardinality."""
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


async def observe_request(request: Request, call_next) -> Response:
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()
    IN_FLIGHT.inc()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        IN_FLIGHT.dec()
        elapsed = time.perf_counter() - start
        path = route_template(request)
        if path != "/metrics":
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(request.method, path).observe(elapsed)
        if path not in QUIET_PATHS:
            log_event(
                "request",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=status,
                duration_ms=round(elapsed * 1000, 2),
                client=request.client.host if request.client else None,
            )


def record_prediction(request_id: str, features: dict, result: dict) -> None:
    PREDICTIONS.labels(result["label"], result["risk_level"]).inc()
    PROBABILITY.observe(result["probability_disease"])
    log_event(
        "prediction",
        request_id=request_id,
        prediction=result["prediction"],
        risk_level=result["risk_level"],
        probability_disease=result["probability_disease"],
        features=features,
    )


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
