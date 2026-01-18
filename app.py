import os
import io
import zipfile
import math
from datetime import datetime, timedelta, date

# --- Render-safe matplotlib setup (BEFORE importing pyplot) ---
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")  # server-safe backend

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

# Basemap sources (try CDN first, then GitHub raw)
US_STATES_GEOJSON_URLS = [
    "https://cdn.jsdelivr.net/gh/PublicaMundi/MappingAPI@master/data/us-states.json",
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/us-states.json",
]

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

# ---- Line styling (day/night) ----
DAY_LINEWIDTH = 2.6
NIGHT_LINEWIDTH = 1.1
NIGHT_ALPHA = 0.55

# ---- Labels ----
ROUTE_LABEL_FONTSIZE = 6.3
LABEL_HALO_WIDTH = 2.2
LABEL_OFFSET_DEGREES = 0.28  # how far label sits off the track

# ---- Direction separation ("small gap" between directions) ----
# This is the perpendicular offset applied to each segment.
# Increase slightly if you want the two directions more clearly separated.
DIRECTION_GAP_DEG = 0.20  # ~small; try 0.08–0.16

# ---- Chevrons (unfilled) ----
CHEVRON_EVERY_N_SEGMENTS = 18
CHEVRON_SKIP_END_SEGMENTS = 6
CHEVRON_SIZE_DEG = 0.085
CHEVRON_ANGLE_DEG = 24
CHEVRON_LW_DAY = 0.35
CHEVRON_LW_NIGHT = 0.28
CHEVRON_ALPHA_DAY = 0.55
CHEVRON_ALPHA_NIGHT = 0.35

# ---- Fixed, print-safe colours ----
HEX_PALETTE = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
    "#66a61e", "#e6ab02", "#a6761d", "#666666",
    "#1f78b4", "#b2df8a", "#fb9a99", "#fdbf6f",
    "#cab2d6", "#6a3d9a",
]
ROUTE_COLOURS = {name: HEX_PALETTE[i % len(HEX_PALETTE)] for i, name in enumerate(LONG_DISTANCE_NAMES)}

# --- caches ---
_GTFS_CACHE = {"fetched_at": None, "zip_bytes": None}
_STATES_CACHE = {"fetched_at": None, "geojson": None}


def _download_bytes_cached(url: str, cache: dict, key_bytes: str, key_time: str, max_age_minutes: int = 1440) -> bytes:
    now = datetime.utcnow()
    if cache.get(key_bytes) is not None and cache.get(key_time) is not None:
        age = (now - cache[key_time]).total_seconds() / 60.0
        if age <= max_age_minutes:
            return cache[key_bytes]
    r = requests.get(url, timeout=60, headers={"User-Agent": "amtrak-map-service/1.0"})
    r.raise_for_status()
    cache[key_bytes] = r.content
    cache[key_time] = now
    return r.content


def _download_gtfs_zip_bytes(max_age_minutes: int = 60) -> bytes:
    return _download_bytes_cached(GTFS_URL, _GTFS_CACHE, "zip_bytes", "fetched_at", max_age_minutes)


def _download_states_geojson(max_age_minutes: int = 1440):
    # cache hit
    now = datetime.utcnow()
    if _STATES_CACHE["geojson"] is not None and _STATES_CACHE["fetched_at"] is not None:
        age = (now - _STATES_CACHE["fetched_at"]).total_seconds() / 60.0
        if age <= max_age_minutes:
            return _STATES_CACHE["geojson"]

    # try multiple URLs
    last_err = None
    for url in US_STATES_GEOJSON_URLS:
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "amtrak-map-service/1.0"})
            r.raise_for_status()
            gj = r.json()
            _STATES_CACHE["geojson"] = gj
            _STATES_CACHE["fetched_at"] = now
            return gj
        except Exception as e:
            last_err = e

    # fail gracefully
    print(f"[WARN] Failed to download states basemap from all sources: {last_err}")
    return {"type": "FeatureCollection", "features": []}


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


def _perp_unit(dx, dy):
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0, 0.0
    # perpendicular unit vector
    return (-dy / length, dx / length)


def _draw_states_basemap(ax):
    gj = _download_states_geojson()
    feats = gj.get("features", [])

    # Make it visible but subtle
    line_color = "#000000"
    lw = 0.7
    alpha = 0.38

    for feat in feats:
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])

        def draw_ring(ring):
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            ax.plot(xs, ys, linewidth=lw, alpha=alpha, zorder=1, color=line_color)

        if gtype == "Polygon":
            for ring in coords:
                draw_ring(ring)
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    draw_ring(ring)


def _draw_chevron(ax, x, y, heading_rad, colour, is_day: bool):
    lw = CHEVRON_LW_DAY if is_day else CHEVRON_LW_NIGHT
    alpha = CHEVRON_ALPHA_DAY if is_day else CHEVRON_ALPHA_NIGHT

    a = math.radians(CHEVRON_ANGLE_DEG)
    s = CHEVRON_SIZE_DEG

    tip_x = x + math.cos(heading_rad) * (s * 0.35)
    tip_y = y + math.sin(heading_rad) * (s * 0.35)

    left_dir = heading_rad + math.pi - a
    right_dir = heading_rad + math.pi + a

    left_x = tip_x + math.cos(left_dir) * (s * 0.85)
    left_y = tip_y + math.sin(left_dir) * (s * 0.85)
    right_x = tip_x + math.cos(right_dir) * (s * 0.85)
    right_y = tip_y + math.sin(right_dir) * (s * 0.85)

    ax.plot([left_x, tip_x], [left_y, tip_y], color=colour, lw=lw, alpha=alpha, zorder=6)
    ax.plot([right_x, tip_x], [right_y, tip_y], color=colour, lw=lw, alpha=alpha, zorder=6)


def _load_representative_trips(run_date: date):
    """
    Pick one representative trip per (route, direction) = the trip with most stops.
    """
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


def _make_single_map_two_directions(run_date: date):
    """
    ONE map, but each route is drawn as a paired track:
    direction 0 is offset to one side, direction 1 to the other, leaving a small gap.
    """
    reps, stop_times, stops = _load_representative_trips(run_date)

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_title(f"Amtrak Long-Distance Routes — Both Directions (separated)\nDaylight vs Darkness — {run_date}", fontsize=14)

    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Basemap
    _draw_states_basemap(ax)

    # Legends
    legend_routes = [mlines.Line2D([], [], color=ROUTE_COLOURS[nm], lw=3, label=nm) for nm in LONG_DISTANCE_NAMES]
    day_leg = mlines.Line2D([], [], color="black", lw=3, label="Daylight (solid)")
    night_leg = mlines.Line2D([], [], color="black", lw=1.2, linestyle="--", label="Darkness (dashed)")

    ax.legend(handles=[day_leg, night_leg], loc="lower left", fontsize=8, frameon=False)
    key = ax.legend(
        handles=legend_routes,
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

    # Draw both directions, offset per segment
    for name, dir_id, trip_id in reps:
        colour = ROUTE_COLOURS.get(name, "#000000")

        st = stop_times[stop_times["trip_id"] == trip_id].merge(stops, on="stop_id", how="left")
        st = st.dropna(subset=["stop_lat", "stop_lon", "dep_sec", "arr_sec", "stop_timezone"])
        if len(st) < 2:
            continue

        # dir sign controls which side of the "centerline" this direction sits
        # (0 -> one side, 1 -> other side)
        sign = -1.0 if dir_id == 0 else 1.0

        nseg = len(st) - 1

        for i in range(nseg):
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

            dx = x1 - x0
            dy = y1 - y0
            ux, uy = _perp_unit(dx, dy)

            # apply perpendicular offset so the two directions sit side-by-side
            offx = ux * DIRECTION_GAP_DEG * sign
            offy = uy * DIRECTION_GAP_DEG * sign

            xs = [x0 + offx, x1 + offx]
            ys = [y0 + offy, y1 + offy]

            ax.plot(
                xs,
                ys,
                color=colour,
                linewidth=DAY_LINEWIDTH if daylight else NIGHT_LINEWIDTH,
                linestyle="-" if daylight else "--",
                alpha=1.0 if daylight else NIGHT_ALPHA,
                zorder=5,
            )

            # chevrons (skip near ends)
            near_start = i < CHEVRON_SKIP_END_SEGMENTS
            near_end = i > (nseg - 1 - CHEVRON_SKIP_END_SEGMENTS)
            if (i % CHEVRON_EVERY_N_SEGMENTS == 0) and (not near_start) and (not near_end):
                xc = (x0 + x1) / 2.0 + offx
                yc = (y0 + y1) / 2.0 + offy
                heading = math.atan2(dy, dx)
                _draw_chevron(ax, xc, yc, heading, colour, daylight)

        # label near midpoint, offset away from the *directional* track
        mid = len(st) // 2
        a = st.iloc[mid - 1]
        b = st.iloc[mid]
        x0, y0 = float(a["stop_lon"]), float(a["stop_lat"])
        x1, y1 = float(b["stop_lon"]), float(b["stop_lat"])

        dx = x1 - x0
        dy = y1 - y0
        ux, uy = _perp_unit(dx, dy)

        # place label slightly further out than the track offset so it doesn't overlap
        track_offx = ux * DIRECTION_GAP_DEG * sign
        track_offy = uy * DIRECTION_GAP_DEG * sign
        label_offx = ux * (DIRECTION_GAP_DEG * sign + LABEL_OFFSET_DEGREES * sign)
        label_offy = uy * (DIRECTION_GAP_DEG * sign + LABEL_OFFSET_DEGREES * sign)

        xm = (x0 + x1) / 2.0 + label_offx
        ym = (y0 + y1) / 2.0 + label_offy

        ax.text(
            xm,
            ym,
            name,
            fontsize=ROUTE_LABEL_FONTSIZE,
            weight="bold",
            ha="center",
            va="center",
            color=colour,
            zorder=7,
            path_effects=[pe.withStroke(linewidth=LABEL_HALO_WIDTH, foreground="white")],
        )

    ax.grid(True, linewidth=0.3, alpha=0.20)

    # room on right for legend
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    return fig


def build_png_bytes(run_date: date) -> bytes:
    fig = _make_single_map_two_directions(run_date)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def build_pdf_bytes(run_date: date) -> bytes:
    out = io.BytesIO()
    with PdfPages(out) as pdf:
        fig = _make_single_map_two_directions(run_date)
        pdf.savefig(fig)
        plt.close(fig)
    out.seek(0)
    return out.getvalue()


@app.get("/")
def index():
    d = request.args.get("date", "2026-02-06").strip()
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Amtrak Daylight Map</title>
      <style>
        body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 18px; }}
        .top {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom: 12px; }}
        .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px; }}
        img {{ width: 100%; height: auto; display: block; border-radius: 6px; }}
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
        <span>PDF: <a href="/map.pdf?date={d}">print/download</a></span>
      </div>

      <div class="card">
        <img src="/map.png?date={d}" alt="Amtrak map">
      </div>
    </body>
    </html>
    """
    return Response(html, mimetype="text/html")


@app.get("/map.png")
def map_png():
    d = request.args.get("date", "").strip()
    if not d:
        abort(400, "Missing required query parameter: date=YYYY-MM-DD")
    try:
        run_date = datetime.strptime(d, "%Y-%m-%d").date()
    except ValueError:
        abort(400, "Invalid date format. Use YYYY-MM-DD (e.g. 2026-02-06).")

    try:
        png_bytes = build_png_bytes(run_date)
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
