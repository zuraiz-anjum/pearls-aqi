"""Server-rendered SVG for the Flask pages.

No charting library, no JavaScript, no CDN. The whole forecast chart is a few
hundred bytes of markup built here and dropped into the template, which means
the page renders complete on first byte, works with scripts disabled, prints
properly, and never has a loading spinner where the chart should be.

It is deliberately less capable than Plotly - no hover, no zoom. The Streamlit
dashboard has all of that. This page is the one you send to someone who just
wants to know whether to go for a run tomorrow.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

from aqi.aqi_math import CATEGORY_COLOURS, category

# CSS class per category. Kept short because they end up in class attributes.
CATEGORY_SLUG = {
    "Good": "good",
    "Moderate": "moderate",
    "Unhealthy for Sensitive Groups": "usg",
    "Unhealthy": "unhealthy",
    "Very Unhealthy": "very",
    "Hazardous": "hazardous",
    "Unknown": "unknown",
}

# Text-safe versions of the EPA colours. The official swatches are designed for
# filled blocks - #ffff00 on white is unreadable as text and fails contrast by a
# mile - so these are darkened until each clears 4.5:1 on the light surface.
# The raw EPA colours still get used for the faint background bands, where
# contrast is not the point.
CATEGORY_INK = {
    "Good": "#15803d",
    "Moderate": "#a16207",
    "Unhealthy for Sensitive Groups": "#c2410c",
    "Unhealthy": "#b91c1c",
    "Very Unhealthy": "#6b21a8",
    "Hazardous": "#7e0023",
    "Unknown": "#64748b",
}

_BANDS = [
    (0, 50, "Good"),
    (50, 100, "Moderate"),
    (100, 150, "Unhealthy for Sensitive Groups"),
    (150, 200, "Unhealthy"),
    (200, 300, "Very Unhealthy"),
    (300, 500, "Hazardous"),
]

_PAD_L, _PAD_R, _PAD_T, _PAD_B = 44, 16, 14, 30


def _utc(ts: str | datetime) -> datetime:
    if isinstance(ts, datetime):
        return ts
    return datetime.fromisoformat(str(ts).replace("Z", ""))


def _fmt(v: float) -> str:
    return f"{v:.1f}"


def forecast_chart(
    history: list[dict],
    forecast: list[dict],
    current_aqi: float,
    as_of: str | None = None,
    tz: str = "Asia/Karachi",
    width: int = 920,
    height: int = 320,
) -> str:
    """Observed AQI line, forecast points, and the empirical band between them."""
    points = [(_utc(r["ts"]), r.get("aqi")) for r in history if r.get("ts")]
    points = sorted(points, key=lambda p: p[0])

    if not points:
        return (
            f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="No observations yet">'
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" class="chart-empty">'
            f"No observations yet</text></svg>"
        )

    observed = [(t, v) for t, v in points if v is not None]
    anchor = _utc(as_of) if as_of else (observed[-1][0] if observed else points[-1][0])
    t0 = points[0][0]
    t_end = anchor + timedelta(days=max(f["horizon_days"] for f in forecast) if forecast else 0, hours=6)
    span = max((t_end - t0).total_seconds(), 3600)

    highs = [v for _, v in observed] + [f["aqi_high"] for f in forecast] + [current_aqi]
    y_max = max(320.0, 1.15 * max(highs))

    inner_w = width - _PAD_L - _PAD_R
    inner_h = height - _PAD_T - _PAD_B

    def x_of(t: datetime) -> float:
        return _PAD_L + (t - t0).total_seconds() / span * inner_w

    def y_of(v: float) -> float:
        return _PAD_T + (1 - v / y_max) * inner_h

    opening = (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="chart-title chart-desc" preserveAspectRatio="xMidYMid meet">'
    )
    out = [
        opening,
        '<title id="chart-title">Observed and forecast AQI</title>',
        f'<desc id="chart-desc">{escape(_describe(observed, forecast, current_aqi))}</desc>',
    ]

    # Category bands. Faint, so the line still reads as the subject.
    for lo, hi, name in _BANDS:
        if lo >= y_max:
            break
        top = y_of(min(hi, y_max))
        out.append(
            f'<rect x="{_PAD_L}" y="{_fmt(top)}" width="{inner_w}" height="{_fmt(y_of(lo) - top)}" '
            f'fill="{CATEGORY_COLOURS[name]}" fill-opacity="0.07"/>'
        )

    # Gridlines on the category edges, labelled at the left.
    for edge in (50, 100, 150, 200, 300):
        if edge >= y_max:
            continue
        y = y_of(edge)
        out.append(
            f'<line x1="{_PAD_L}" x2="{width - _PAD_R}" y1="{_fmt(y)}" y2="{_fmt(y)}" class="grid"/>'
            f'<text x="{_PAD_L - 8}" y="{_fmt(y + 4)}" text-anchor="end" class="tick">{edge}</text>'
        )

    # Day ticks at local midnight. Every day if the window is short, else every other.
    local_tz = ZoneInfo(tz)
    first_local = t0.replace(tzinfo=timezone.utc).astimezone(local_tz)
    day = first_local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    span_days = span / 86400
    step = 1 if span_days <= 8 else 2
    i = 0
    while day.astimezone(timezone.utc).replace(tzinfo=None) < t_end:
        t_utc = day.astimezone(timezone.utc).replace(tzinfo=None)
        if i % step == 0:
            x = x_of(t_utc)
            out.append(
                f'<line x1="{_fmt(x)}" x2="{_fmt(x)}" y1="{_PAD_T}" y2="{_PAD_T + inner_h}" class="grid grid-v"/>'
                f'<text x="{_fmt(x)}" y="{height - 10}" text-anchor="middle" class="tick">'
                f"{day:%a} {day.day}</text>"
            )
        day += timedelta(days=1)
        i += 1

    # Observed line. A None breaks the path so a station outage shows as a gap,
    # not as a straight line drawn confidently across it.
    segments, current = [], []
    for t, v in points:
        if v is None:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(f"{_fmt(x_of(t))},{_fmt(y_of(v))}")
    if current:
        segments.append(current)
    for seg in segments:
        if len(seg) == 1:
            x, y = seg[0].split(",")
            out.append(f'<circle cx="{x}" cy="{y}" r="2" class="observed-dot"/>')
        else:
            out.append(f'<polyline points="{" ".join(seg)}" class="observed"/>')

    # Forecast: band, dashed line from the anchor, one dot per day.
    if forecast:
        ax, ay = x_of(anchor), y_of(current_aqi)
        f_pts = [(x_of(anchor + timedelta(days=f["horizon_days"])), f) for f in forecast]

        hi_path = [f"{_fmt(ax)},{_fmt(ay)}"] + [f"{_fmt(x)},{_fmt(y_of(f['aqi_high']))}" for x, f in f_pts]
        lo_path = [f"{_fmt(x)},{_fmt(y_of(f['aqi_low']))}" for x, f in reversed(f_pts)] + [f"{_fmt(ax)},{_fmt(ay)}"]
        out.append(f'<polygon points="{" ".join(hi_path + lo_path)}" class="band"/>')

        line = [f"{_fmt(ax)},{_fmt(ay)}"] + [f"{_fmt(x)},{_fmt(y_of(f['aqi']))}" for x, f in f_pts]
        out.append(f'<polyline points="{" ".join(line)}" class="forecast"/>')

        for x, f in f_pts:
            cat = category(f["aqi"])
            out.append(
                f'<circle cx="{_fmt(x)}" cy="{_fmt(y_of(f["aqi"]))}" r="4.5" '
                f'fill="{CATEGORY_INK[cat]}" class="forecast-dot">'
                f"<title>{escape(f['valid_for'])}: AQI {f['aqi']:.0f} ({escape(cat)})</title></circle>"
            )

        # "now" marker
        out.append(
            f'<line x1="{_fmt(ax)}" x2="{_fmt(ax)}" y1="{_PAD_T}" y2="{_PAD_T + inner_h}" class="now"/>'
        )

    out.append("</svg>")
    return "".join(out)


def _describe(observed, forecast, current_aqi) -> str:
    """Plain-language summary for screen readers. The chart's key insight, not its geometry."""
    if not forecast:
        return f"Current AQI {current_aqi:.0f}."
    days = ", ".join(f"{f['valid_for']} about {f['aqi']:.0f} ({category(f['aqi'])})" for f in forecast)
    span = f"{len(observed)} hourly observations" if observed else "no observations"
    return f"Current AQI {current_aqi:.0f} ({category(current_aqi)}). {span}. Forecast: {days}."
