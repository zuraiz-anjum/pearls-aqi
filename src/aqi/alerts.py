"""Hazardous-AQI alerting.

Fires when any of the three forecast days crosses the configured threshold
(default 200 = Unhealthy). Posts a Slack-compatible payload if ALERT_WEBHOOK_URL
is set, and always returns the event so a caller can log or display it.

Two things that stop this being annoying:

  - Deduplication. The hourly pipeline would otherwise fire the same alert 24
    times for one bad day. We keep the last sent fingerprint on disk and stay
    quiet unless the situation actually changes.
  - The prediction interval. Alerting on a point estimate of 205 when the model's
    own error band is +/-40 is a coin flip dressed up as a warning, so the default
    requires the *lower* bound to clear the threshold.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import requests

from .aqi_math import category, health_note
from .config import PROCESSED_DIR, settings

log = logging.getLogger(__name__)

STATE_PATH = PROCESSED_DIR / "alert_state.json"
RESEND_AFTER = timedelta(hours=12)


def _fingerprint(triggered: list[dict]) -> str:
    """Same days at the same severity = same alert, however the number wobbles."""
    return "|".join(f"{d['valid_for']}:{category(d['aqi'])}" for d in triggered)


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def evaluate(prediction: dict, threshold: float | None = None, use_lower_bound: bool = True) -> dict:
    """Decide whether this forecast is worth waking someone up for."""
    threshold = threshold if threshold is not None else settings.alert_threshold

    triggered = []
    for day in prediction["forecast"]:
        value = day["aqi_low"] if use_lower_bound else day["aqi"]
        if value >= threshold:
            triggered.append(day)

    if not triggered:
        return {"triggered": False, "threshold": threshold, "days": []}

    worst = max(triggered, key=lambda d: d["aqi"])
    return {
        "triggered": True,
        "threshold": threshold,
        "days": triggered,
        "worst": worst,
        "fingerprint": _fingerprint(triggered),
        "headline": (
            f"{category(worst['aqi'])} air expected in {settings.city.title()} "
            f"on {worst['valid_for']} (AQI ~{worst['aqi']:.0f})"
        ),
        "advice": health_note(worst["aqi"]),
    }


def should_send(event: dict) -> bool:
    if not event.get("triggered"):
        return False

    state = _load_state()
    if state.get("fingerprint") != event["fingerprint"]:
        return True

    last = state.get("sent_at")
    if not last:
        return True
    try:
        return datetime.fromisoformat(last) < datetime.now(timezone.utc) - RESEND_AFTER
    except ValueError:
        return True


def to_slack_payload(event: dict, prediction: dict, why: str = "") -> dict:
    lines = [f"*{event['headline']}*", "", event["advice"]]
    if why:
        lines += ["", f"_{why}_"]

    lines += ["", "*Forecast*"]
    for day in prediction["forecast"]:
        flag = " :warning:" if day in event["days"] else ""
        lines.append(
            f"- {day['valid_for']}: *{day['aqi']:.0f}* ({day['category']}), "
            f"range {day['aqi_low']:.0f}-{day['aqi_high']:.0f}{flag}"
        )

    current = (
        f"Current reading: {prediction['current_aqi']:.0f} "
        f"({prediction['current_category']}), "
        f"as of {prediction['as_of_local'][:16].replace('T', ' ')}"
    )
    lines += ["", current]
    return {"text": "\n".join(lines)}


def send(event: dict, prediction: dict, why: str = "") -> bool:
    """Post to the webhook. Returns whether anything actually went out."""
    if not event.get("triggered"):
        return False
    if not should_send(event):
        log.info("alert suppressed - already sent for %s", event["fingerprint"])
        return False

    if not settings.alert_webhook:
        log.warning("ALERT WOULD FIRE: %s (no ALERT_WEBHOOK_URL configured)", event["headline"])
        return False

    payload = to_slack_payload(event, prediction, why)
    try:
        resp = requests.post(settings.alert_webhook, json=payload, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        log.error("failed to post alert: %s", exc)
        return False

    _save_state(
        {
            "fingerprint": event["fingerprint"],
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "headline": event["headline"],
        }
    )
    log.info("alert sent: %s", event["headline"])
    return True


def check_and_notify(prediction: dict | None = None, explain: bool = True) -> dict:
    """Entry point for the hourly job. Safe to call every run."""
    if prediction is None:
        from .inference import predict

        prediction = predict()

    event = evaluate(prediction)
    if not event["triggered"]:
        log.info("no alert: peak forecast %s", max(d["aqi"] for d in prediction["forecast"]))
        return event

    why = ""
    if explain:
        from .explain import narrate

        why = narrate(event["worst"]["horizon_days"])

    event["sent"] = send(event, prediction, why)
    event["why"] = why
    return event


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    print(json.dumps(check_and_notify(), indent=2, default=str))
