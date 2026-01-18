import os
import io
import zipfile
import math
from datetime import datetime, timedelta, date

# --- Render-safe matplotlib setup (BEFORE importing pyplot) ---
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")

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

# ---- Day/Night styling ----
DAY_LINEWIDTH = 2.6
NIGHT_LINEWIDTH = 1.1
NIGHT_ALPHA = 0.55

# ---- Parallel separation ----
PARALLEL_GAP_DEG = 0.17

# ---- Station label styling ----
STATION_FONTSIZE = 7.0
STATION_HALO_WIDTH = 2.6

# A small, curated set of trigraphs to avoid clutter.
# These are used only if they exist in the GTFS stops list.
KEY_STATION_CODES = [
    "SEA", "PDX", "SAC", "EMY", "SJC", "LAX", "SAN",
    "ABQ", "DEN", "SLC", "CHI", "STL", "MSP",
    "NOL", "ATL", "WAS", "NYP", "BOS", "PHL", "BAL", "CLT", "ORL", "MIA",
]

# ---- Chevrons (bigger) ----
CHEVRON_EVERY_N_SEGMENTS = 10
CHEVRON_SKIP_END_SEGMENTS = 3
CHEVRON_SIZE_DEG = 0.18      # bigger than before
CHEVRON_ANGLE_DEG = 24
CHEVRON_LW_DAY = 1.05        # slightly thicker
CHEVRON_LW_NIGHT = 0.90
CHEVRON_ALPHA_DAY = 0.95
CHEVRON_ALPHA_NIGHT = 0.75
CHEVRON_ZORDER = 10

# ---- Fixed colours ----
HEX_PALETTE = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
    "#66a61e", "#e6ab02", "#a6761d", "#666666",
    "#1f78b4", "#b2df8a", "#fb9a99", "#fdbf6f",
    "#cab2d6", "#6a3d9a",
]
ROUTE_COLOURS = {name: HEX_PALETTE[i % len(HEX_PALETTE)] for i, name in enumerate(LONG_DISTANCE_NAMES)}

_GTFS_CACHE = {"fetched_at": None, "zip_bytes": None}


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


def _sun_times(lat: float, lon: float, tz_name: str, on_date: date):
    try:
        tz = pytz.timezone(tz_name)
    except Exception:
        tz = pytz.UTC
    loc = LocationInfo(latitude=lat, longitude=lon, timezone=getattr(tz, "zone", "UTC"))
    s = sun(loc.observer, date=on_date, tzinfo=tz)
    return s["sunrise"], s["sunset"]


def _is_daylight_at(lat: float, lon: float, tz_name: str, dt_local) -> bool:
    sunrise, sunset = _sun_times(lat, lon, tz_name, dt_local.date())
    return sunrise <= dt_local <= sunset


def _perp_unit(dx, dy):
    L = math.hypot(dx, dy)
    if L == 0:
        return 0.0, 0.0
    return (-dy / L, dx / L)


def _offset_polyline_constant_parallel(lons, lats, gap_deg, side_sign):
    n = len(lons)
    if n < 2:
        return lons, lats

    outx = [0.0] * n
    outy = [0.0] * n

    for i in range(n):
        if i == 0:
            dx = lons[1] - lons[0]
            dy = lats[1] - lats[0]
        elif i == n - 1:
            dx = lons[n - 1] - lons[n - 2]
            dy = lats[n - 1] - lats[n - 2]
        else:
            dx = lons[i + 1] - lons[i - 1]
            dy = lats[i + 1] - lats[i - 1]

        ux, uy = _perp_unit(dx, dy)
        outx[i] = lons[i] + ux * gap_deg * side_sign
        outy[i] = lats[i] + uy * gap_deg * side_sign

    return outx, outy


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

    ax.plot([left_x, tip_x], [left_y, tip_y], color=colour, lw=lw, alpha=alpha, zorder=CHEVRON_ZORDER)
    ax.plot([right_x, tip_x], [right_y, tip_y], color=colour, lw=lw, alpha=alpha, zorder=CHEVRON_ZORDER)


def _load_trips_and_stops(run_date: date):
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

    # Stops often contain stop_code with trigraphs (not always). We'll use it if present.
    stops_cols = ["stop_id", "stop_lat", "stop_lon", "stop_timezone"]
    if "stop_code" in stops.columns:
        stops_cols.append("stop_code")
    stops = stops[stops_cols].copy()
    stops["stop_timezone"] = stops["stop_timezone"].fillna("UTC")
    if "stop_code" not in stops.columns:
        stops["stop_code"] = None

    reps = {}
    for (name, direction), grp in trips.groupby(["route_long_name", "direction_id"]):
        counts = stop_times[stop_times["trip_id"].isin(grp["trip_id"])].groupby("trip_id").size()
        if len(counts):
            reps[(name, int(direction))] = counts.idxmax()

    if not reps:
        raise RuntimeError("No matching long-distance trips found for that date in the GTFS feed.")

    return reps, stop_times, stops


def _map_seg_index(i, nseg_geom, nseg_time):
    if nseg_geom <= 1 or nseg_time <= 1:
        return min(i, max(0, nseg_time - 1))
    return int(round(i * (nseg_time - 1) / (nseg_geom - 1)))


def _trip_anchor_datetime(run_date: date, st_time: pd.DataFrame):
    first = st_time.iloc[0]
    origin_tz = first["stop_timezone"]
    origin_dep_sec = first["dep_sec"]
    if origin_dep_sec is None or pd.isna(origin_dep_sec):
        origin_dep_sec = first["arr_sec"]

    try:
        tz = pytz.timezone(origin_tz)
    except Exception:
        tz = pytz.UTC

    origin_dt_local = tz.localize(datetime(run_date.year, run_date.month, run_date.day) + timedelta(seconds=int(origin_dep_sec)))
    return origin_dt_local, int(origin_dep_sec)


def _draw_station_labels(ax, stops: pd.DataFrame):
    """
    Draw a curated set of station trigraphs (stop_code) if present.
    We only label those in KEY_STATION_CODES to keep it uncluttered.
    """
    if "stop_code" not in stops.columns:
        return

    stops2 = stops.dropna(subset=["stop_lat", "stop_lon"]).copy()
    stops2["stop_code"] = stops2["stop_code"].fillna("").astype(str).str.strip()
    stops2 = stops2[stops2["stop_code"].isin(KEY_STATION_CODES)]

    # If duplicates exist (multiple stop_ids share same code), take the first
    stops2 = stops2.drop_duplicates(subset=["stop_code"])

    for _, r in stops2.iterrows():
        code = r["stop_code"]
        x = float(r["stop_lon"])
        y = float(r["stop_lat"])

        ax.text(
            x, y, code,
            fontsize=STATION_FONTSIZE,
            ha="center", va="center",
            color="black",
            zorder=20,
            path_effects=[pe.withStroke(linewidth=STATION_HALO_WIDTH, foreground="white")],
        )


def _make_map(run_date: date):
    reps, stop_times, stops = _load_trips_and_stops(run_date)

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_title(
        f"Amtrak Long-Distance Routes — Both Directions (parallel tracks)\nDaylight vs Darkness — {run_date}",
        fontsize=14,
    )
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linewidth=0.3, alpha=0.20)

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

    # Station labels (trigraphs)
    _draw_station_labels(ax, stops)

    for name in LONG_DISTANCE_NAMES:
        trip0 = reps.get((name, 0))
        trip1 = reps.get((name, 1))
        center_trip = trip0 or trip1
        if not center_trip:
            continue

        colour = ROUTE_COLOURS.get(name, "#000000")

        st_geom = stop_times[stop_times["trip_id"] == center_trip].merge(stops, on="stop_id", how="left")
        st_geom = st_geom.dropna(subset=["stop_lat", "stop_lon", "stop_timezone"])
        st_geom = st_geom.sort_values("stop_sequence")
        if len(st_geom) < 2:
            continue

        lons = [float(v) for v in st_geom["stop_lon"].tolist()]
        lats = [float(v) for v in st_geom["stop_lat"].tolist()]

        left_lons, left_lats = _offset_polyline_constant_parallel(lons, lats, PARALLEL_GAP_DEG, side_sign=-1.0)
        right_lons, right_lats = _offset_polyline_constant_parallel(lons, lats, PARALLEL_GAP_DEG, side_sign=+1.0)

        dir_tracks = [
            (0, left_lons, left_lats, trip0),
            (1, right_lons, right_lats, trip1),
        ]

        for direction_id, xs, ys, dir_trip in dir_tracks:
            if not dir_trip:
                continue

            st_time = stop_times[stop_times["trip_id"] == dir_trip].merge(stops, on="stop_id", how="left")
            st_time = st_time.dropna(subset=["dep_sec", "arr_sec", "stop_timezone", "stop_lat", "stop_lon"])
            st_time = st_time.sort_values("stop_sequence")
            if len(st_time) < 2:
                continue

            origin_dt_local, origin_dep_sec = _trip_anchor_datetime(run_date, st_time)

            if direction_id == 1:
                xs = list(reversed(xs))
                ys = list(reversed(ys))

            nseg = len(xs) - 1
            nseg_time = len(st_time) - 1
            if nseg <= 0 or nseg_time <= 0:
                continue

            for i in range(nseg):
                j = _map_seg_index(i, nseg, nseg_time)
                a = st_time.iloc[j]
                b = st_time.iloc[min(j + 1, len(st_time) - 1)]

                t0 = a["dep_sec"]
                t1 = b["arr_sec"]
                if t0 is None or t1 is None:
                    continue

                mid_sec = int((int(t0) + int(t1)) / 2)
                delta_from_origin = mid_sec - origin_dep_sec
                dt_at_origin = origin_dt_local + timedelta(seconds=delta_from_origin)

                tz_name_here = a["stop_timezone"]
                try:
                    tz_here = pytz.timezone(tz_name_here)
                except Exception:
                    tz_here = pytz.UTC

                dt_here = dt_at_origin.astimezone(tz_here)

                lat_mid = (float(a["stop_lat"]) + float(b["stop_lat"])) / 2.0
                lon_mid = (float(a["stop_lon"]) + float(b["stop_lon"])) / 2.0
                sunrise, sunset = _sun_times(lat_mid, lon_mid, tz_name_here, dt_here.date())
                daylight = sunrise <= dt_here <= sunset

                x0, y0 = xs[i], ys[i]
                x1, y1 = xs[i + 1], ys[i + 1]

                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color=colour,
                    linewidth=DAY_LINEWIDTH if daylight else NIGHT_LINEWIDTH,
                    linestyle="-" if daylight else "--",
                    alpha=1.0 if daylight else NIGHT_ALPHA,
                    zorder=5,
                )

                near_start = i < CHEVRON_SKIP_END_SEGMENTS
                near_end = i > (nseg - 1 - CHEVRON_SKIP_END_SEGMENTS)
                if (i % CHEVRON_EVERY_N_SEGMENTS == 0) and (not near_start) and (not near_end):
                    heading = math.atan2((y1 - y0), (x1 - x0))
                    _draw_chevron(ax, (x0 + x1) / 2.0, (y0 + y1) / 2.0, heading, colour, daylight)

    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    return fig


def build_png_bytes(run_date: date) -> bytes:
    fig = _make_map(run_date)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def build_pdf_bytes(run_date: date) -> bytes:
    out = io.BytesIO()
    with PdfPages(out) as pdf:
        fig = _make_map(run_date)
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
