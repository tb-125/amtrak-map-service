import os
import io
import zipfile
import math
from datetime import datetime, timedelta, date

# --- Render-safe matplotlib setup (BEFORE importing pyplot) ---
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")  # non-GUI backend for servers

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd
import requests
import pytz
from astral import LocationInfo
from astral.sun import sun

from flask import Flask, Response, request, abort

app = Flask(__name__)

GTFS_URL = "https://content.amtrak.com/content/gtfs/GTFS.zip"

LONG_DISTANCE_NAMES = [
    "Auto Train",
    "California Zephyr",
    "Cardinal",
    "City of New Orleans",
    "Coast Starlight",
    "Crescent",
    "Empire Builder",
    "Lake Shore Limited",
    "Southwest Chief",
    "Sunset Limited",
    "Texas Eagle",
    "Silver Meteor",
    "Palmetto",
    "Floridian",
]

DIRECTION_LABELS = {
    0: "Westbound / Southbound",
    1: "Eastbound / Northbound",
}

KEY_CITIES = [
    ("Seattle", -122.3321, 47.6062),
    ("Portland", -122.6765, 45.5231),
    ("San Francisco", -122.4194, 37.7749),
    ("Los Angeles", -118.2437, 34.0522),
    ("San Diego", -117.1611, 32.7157),
    ("Denver", -104.9903, 39.7392),
    ("Chicago", -87.6298, 41.8781),
    ("St Louis", -90.1994, 38.6270),
    ("New Orleans", -90.0715, 29.9511),
    ("Atlanta", -84.3880, 33.7490),
    ("Washington, DC", -77.0369, 38.9072),
    ("New York", -74.0060, 40.7128),
    ("Boston", -71.0589, 42.3601),
]

# ---- Styling ----
DAY_LINEWIDTH = 2.6
NIGHT_LINEWIDTH = 1.1
NIGHT_ALPHA = 0.55

ARROW_EVERY_N_SEGMENTS = 14
ARROW_HEAD_LENGTH = 3.5
ARROW_HEAD_WIDTH = 2.4
ARROW_LW_DAY = 0.45
ARROW_LW_NIGHT = 0.35
ARROW_ALPHA_DAY = 0.5
ARROW_ALPHA_NIGHT = 0.3

ROUTE_LABEL_FONTSIZE = 6.5
LABEL_OFFSET_DEGREES = 0.35
LABEL_HALO_WIDTH = 2.2

# Fixed, print-safe colours
HEX_PALETTE = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
    "#66a61e", "#e6ab02", "#a6761d", "#666666",
    "#1f78b4", "#b2df8a", "#fb9a99", "#fdbf6f",
    "#cab2d6", "#6a3d9a",
]
ROUTE_COLOURS = {name: HEX_PALETTE[i % len(HEX_PALETTE)] for i, name in enumerate(LONG_DISTANCE_NAMES)}

# --- In-memory cache for GTFS ---
_GTFS_CACHE = {"fetched_at": None, "zip_bytes": None}


def _download_gtfs_zip_bytes(max_age_minutes: int = 60) -> bytes:
    now = datetime.utcnow()
    if _GTFS_CACHE["zip_bytes"] is not None and _GTFS_CACHE["fetched_at"] is not None:
        age = (now - _GTFS_CACHE["fetched_at"]).total_seconds() / 60.0
        if age <= max_age_minutes:
            return _GTFS_CACHE["zip_bytes"]

    r = requests.get(GTFS_URL, timeout=60)
    r.raise_for_status()
    _GTFS_CACHE["zip_bytes"] = r.content
    _GTFS_CACHE["fetched_at"] = now
    return r.content


def _read_txt(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with z.open(name) as f:
        return pd.read_csv(f)


def _parse_gtfs_time(t):
    if pd.isna(t) or not isinstance(t, str) or not t.strip():
        return None
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _service_active_on(row, yyyymmdd: str) -> bool:
    d = datetime.strptime(yyyymmdd, "%Y%m%d").date()
    start = datetime.strptime(str(row["start_date"]), "%Y%m%d").date()
    end = datetime.strptime(str(row["end_date"]), "%Y%m%d").date()
    if not (start <= d <= end):
        return False
    weekday = d.weekday()
    cols = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    return int(row[cols[weekday]]) == 1


def _is_daylight(lat: float, lon: float, tz_name: str, dt_local) -> bool:
    try:
        tz = pytz.timezone(tz_name)
    except Exception:
        tz = pytz.UTC
    loc = LocationInfo(latitude=lat, longitude=lon, timezone=getattr(tz, "zone", "UTC"))
    if dt_local.tzinfo is None:
        dt_local = tz.localize(dt_local)
    s = sun(loc.observer, date=dt_local.date(), tzinfo=dt_local.tzinfo)
    return s["sunrise"] <= dt_local <= s["sunset"]


def _draw_background(ax):
    for nm, lon, lat in KEY_CITIES:
        ax.scatter(lon, lat, s=10, color="0.70", zorder=2)
        ax.text(lon + 0.25, lat + 0.15, nm, fontsize=7, color="0.50", zorder=2)


def _draw_direction_arrow(ax, x0, y0, x1, y1, colour, is_day: bool):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=f"-|>,head_length={ARROW_HEAD_LENGTH},head_width={ARROW_HEAD_WIDTH}",
            color=colour,
            lw=ARROW_LW_DAY if is_day else ARROW_LW_NIGHT,
            linestyle="-" if is_day else "--",
            alpha=ARROW_ALPHA_DAY if is_day else ARROW_ALPHA_NIGHT,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=6,
    )


def _perpendicular_offset(x0, y0, x1, y1, distance):
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0, 0.0
    ux = -dy / length
    uy = dx / length
    return ux * distance, uy * distance


def _load_representative_trips(run_date: date):
    """Load GTFS, filter services active on run_date, and pick a representative trip per (route, direction)."""
    zip_bytes = _download_gtfs_zip_bytes()
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))

    routes = _read_txt(z, "routes.txt")
    trips = _read_txt(z, "trips.txt")
    stop_times = _read_txt(z, "stop_times.txt")
    stops = _read_txt(z, "stops.txt")
    calendar = _read_txt(z, "calendar.txt")

    yyyymmdd = run_date.strftime("%Y%m%d")
    active_services = {
        row["service_id"]
        for _, row in calendar.iterrows()
        if _service_active_on(row, yyyymmdd)
    }

    trips = trips[trips["service_id"].isin(active_services)]
    routes = routes[routes["route_long_name"].isin(LONG_DISTANCE_NAMES)]
    trips = trips.merge(routes[["route_id", "route_long_name"]], on="route_id")

    stop_times = stop_times[stop_times["trip_id"].isin(trips["trip_id"])].copy()
    stop_times["arr_sec"] = stop_times["arrival_time"].apply(_parse_gtfs_time)
    stop_times["dep_sec"] = stop_times["departure_time"].apply(_parse_gtfs_time)
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

    stops = stops[["stop_id", "stop_lat", "stop_lon", "stop_timezone"]].copy()
    stops["stop_timezone"] = stops["stop_timezone"].fillna("UTC")

    reps = []
    for (name, direction), grp in trips.groupby(["route_long_name", "direction_id"]):
        counts = stop_times[stop_times["trip_id"].isin(grp["trip_id"])].groupby("trip_id").size()
        if len(counts):
            reps.append((name, int(direction), counts.idxmax()))

    if not reps:
        raise RuntimeError("No matching long-distance trips found for that date in the GTFS feed.")

    return reps, stop_times, stops


def _make_figure_for_direction(run_date: date, direction: int):
    """Build a Matplotlib figure for a given direction and return the Figure."""
    reps, stop_times, stops = _load_representative_trips(run_date)

    legend_handles = [mlines.Line2D([], [], color=ROUTE_COLOURS[nm], lw=3, label=nm) for nm in LONG_DISTANCE_NAMES]
    day_leg = mlines.Line2D([], [], color="black", lw=3, label="Daylight (solid)")
    night_leg = mlines.Line2D([], [], color="black", lw=1.2, linestyle="--", label="Darkness (dashed)")

    fig, ax = plt.subplots(figsize=(16, 9))

    ax.set_title(
        f"Amtrak Long-Distance Routes — {DIRECTION_LABELS.get(direction, direction)}\n"
        f"Daylight vs Darkness — {run_date}",
        fontsize=14,
    )

    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    _draw_background(ax)

    ax.legend(handles=[day_leg, night_leg], loc="lower left", fontsize=8, frameon=False)

    key = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        fontsize=8,
        frameon=False,
        title="Train services",
        title_fontsize=9,
        ncol=1,
    )
    ax.add_artist(key)

    for name, dir_id, trip_id in reps:
        if dir_id != direction:
            continue

        colour = ROUTE_COLOURS.get(name, "#000000")

        st = stop_times[stop_times["trip_id"] == trip_id].merge(stops, on="stop_id", how="left")
        st = st.dropna(subset=["stop_lat", "stop_lon", "dep_sec", "arr_sec", "stop_timezone"])
        if len(st) < 2:
            continue

        for i in range(len(st) - 1):
            a = st.iloc[i]
            b = st.iloc[i + 1]

            t0 = a["dep_sec"]
            t1 = b["arr_sec"]
            if t0 is None or t1 is None:
                continue

            mid_sec = int((t0 + t1) / 2)
            dt_naive = datetime(run_date.year, run_date.month, run_date.day) + timedelta(seconds=mid_sec)

            tz_name = a["stop_timezone"]
            try:
                dt_local = pytz.timezone(tz_name).localize(dt_naive)
            except Exception:
                dt_local = pytz.UTC.localize(dt_naive)

            lat_mid = (float(a["stop_lat"]) + float(b["stop_lat"])) / 2.0
            lon_mid = (float(a["stop_lon"]) + float(b["stop_lon"])) / 2.0
            daylight = _is_daylight(lat_mid, lon_mid, tz_name, dt_local)

            x0, y0 = float(a["stop_lon"]), float(a["stop_lat"])
            x1, y1 = float(b["stop_lon"]), float(b["stop_lat"])

            ax.plot(
                [x0, x1],
                [y0, y1],
                color=colour,
                linewidth=DAY_LINEWIDTH if daylight else NIGHT_LINEWIDTH,
                linestyle="-" if daylight else "--",
                alpha=1.0 if daylight else NIGHT_ALPHA,
                zorder=5,
            )

            if i % ARROW_EVERY_N_SEGMENTS == 0:
                xm0 = x0 + (x1 - x0) * 0.47
                ym0 = y0 + (y1 - y0) * 0.47
                xm1 = x0 + (x1 - x0) * 0.53
                ym1 = y0 + (y1 - y0) * 0.53
                _draw_direction_arrow(ax, xm0, ym0, xm1, ym1, colour, daylight)

        # label near midpoint, offset perpendicular
        mid = len(st) // 2
        a = st.iloc[mid - 1]
        b = st.iloc[mid]

        x_mid = (float(a["stop_lon"]) + float(b["stop_lon"])) / 2.0
        y_mid = (float(a["stop_lat"]) + float(b["stop_lat"])) / 2.0
        dx, dy = _perpendicular_offset(
            float(a["stop_lon"]), float(a["stop_lat"]),
            float(b["stop_lon"]), float(b["stop_lat"]),
            LABEL_OFFSET_DEGREES
        )

        ax.text(
            x_mid + dx,
            y_mid + dy,
            name,
            fontsize=ROUTE_LABEL_FONTSIZE,
            weight="bold",
            ha="center",
            va="center",
            color=colour,
            zorder=7,
            path_effects=[pe.withStroke(linewidth=LABEL_HALO_WIDTH, foreground="white")],
        )

    ax.grid(True, linewidth=0.3, alpha=0.4)

    # Make room on right for external legend
    fig.tight_layout(rect=[0, 0, 0.80, 1])

    return fig


def build_pdf_bytes(run_date: date) -> bytes:
    out = io.BytesIO()
    with PdfPages(out) as pdf:
        for direction in [0, 1]:
            fig = _make_figure_for_direction(run_date, direction)
            pdf.savefig(fig)
            plt.close(fig)
    out.seek(0)
    return out.getvalue()


def build_png_bytes(run_date: date, direction: int) -> bytes:
    fig = _make_figure_for_direction(run_date, direction)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170)  # dpi tuned for web
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


@app.get("/")
def index():
    # Default date (your target) unless query overrides
    d = request.args.get("date", "2026-02-06").strip()

    # Simple HTML that shows images immediately + lets you change date
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Amtrak Daylight Map</title>
      <style>
        body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 18px; }}
        .row {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
        .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px; }}
        img {{ width: 100%; height: auto; display: block; border-radius: 6px; }}
        .top {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom: 12px; }}
        input {{ padding: 8px 10px; border-radius: 8px; border: 1px solid #ccc; }}
        a {{ color: #0b57d0; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
      </style>
    </head>
    <body>
      <div class="top">
        <h2 style="margin:0;">Amtrak Daylight Map</h2>
        <form method="get" action="/" style="margin:0;">
          <label>Date (YYYY-MM-DD): </label>
          <input name="date" value="{d}" />
          <button type="submit">Update</button>
        </form>
        <span>Optional PDF: <a href="/map.pdf?date={d}">download/print</a></span>
      </div>

      <div class="row">
        <div class="card">
          <h3 style="margin-top:0;">{DIRECTION_LABELS[0]}</h3>
          <img src="/map.png?date={d}&dir=0" alt="Map direction 0">
        </div>
        <div class="card">
          <h3 style="margin-top:0;">{DIRECTION_LABELS[1]}</h3>
          <img src="/map.png?date={d}&dir=1" alt="Map direction 1">
        </div>
      </div>
    </body>
    </html>
    """
    return Response(html, mimetype="text/html")


@app.get("/map.png")
def map_png():
    d = request.args.get("date", "").strip()
    dir_str = request.args.get("dir", "").strip()

    if not d:
        abort(400, "Missing required query parameter: date=YYYY-MM-DD")
    try:
        run_date = datetime.strptime(d, "%Y-%m-%d").date()
    except ValueError:
        abort(400, "Invalid date format. Use YYYY-MM-DD (e.g. 2026-02-06).")

    try:
        direction = int(dir_str)
    except Exception:
        abort(400, "Missing or invalid dir. Use dir=0 or dir=1")

    if direction not in (0, 1):
        abort(400, "dir must be 0 or 1")

    try:
        png_bytes = build_png_bytes(run_date, direction)
    except Exception as e:
        abort(500, f"Failed to generate PNG: {e}")

    return Response(png_bytes, mimetype="image/png", headers={"Cache-Control": "public, max-age=300"})


@app.get("/map.pdf")
def map_pdf():
    d = request.args.get("date", "").strip()
    if not d:
        abort(400, "Missing required query parameter: date=YYYY-MM-DD")
    try:
        run_date = datetime.strptime(d, "%Y-%m-%d").date()
    except ValueError:
        abort(400, "Invalid date format. Use YYYY-MM-DD (e.g. 2026-02-06).")

    try:
        pdf_bytes = build_pdf_bytes(run_date)
    except Exception as e:
        abort(500, f"Failed to generate PDF: {e}")

    filename = f"amtrak_long_distance_daylight_{run_date}.pdf"
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )
