import os
import io
import zipfile
import math
from datetime import datetime, timedelta, date
from collections import defaultdict

# --- Render-safe matplotlib setup (BEFORE importing pyplot) ---
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages

from PIL import Image

import pandas as pd
import requests
import pytz
from astral import LocationInfo
from astral.sun import sun

from flask import Flask, Response, request, abort

app = Flask(__name__)

GTFS_URL = "https://content.amtrak.com/content/gtfs/GTFS.zip"

# ---------------- ROUTES ----------------
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

# ---- Label styling (trigraphs + POIs) ----
LABEL_FONTSIZE = 7.0
LABEL_HALO_WIDTH = 2.6

# ---- Chevrons (night-only) ----
CHEVRON_EVERY_N_SEGMENTS = 10
CHEVRON_SKIP_END_SEGMENTS = 3
CHEVRON_SIZE_DEG = 0.36
CHEVRON_ANGLE_DEG = 24
CHEVRON_LW_NIGHT = 1.05
CHEVRON_ALPHA_NIGHT = 0.90
CHEVRON_ZORDER = 10

# ---- Light accuracy ----
# Civil twilight: treat "light" as dawn..dusk
# Sampling: vote across multiple points along the segment
LIGHT_SAMPLE_POINTS_PER_SEGMENT = 5
LIGHT_MAJORITY_THRESHOLD = 0.5  # >0.5 => mostly light

# ---- Auto-fit map extent to routes ----
AUTO_FIT_MAP = True
FIT_MARGIN_DEG_X = 2.0
FIT_MARGIN_DEG_Y = 1.5
# Safety clamp (CONUS-ish window)
FIT_MIN_LON, FIT_MAX_LON = -125, -66
FIT_MIN_LAT, FIT_MAX_LAT = 24, 50

# ---- Fixed colours ----
HEX_PALETTE = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
    "#66a61e", "#e6ab02", "#a6761d", "#666666",
    "#1f78b4", "#b2df8a", "#fb9a99", "#fdbf6f",
    "#cab2d6", "#6a3d9a",
]
ROUTE_COLOURS = {name: HEX_PALETTE[i % len(HEX_PALETTE)] for i, name in enumerate(LONG_DISTANCE_NAMES)}

# ---- Background states PNG ----
BACKGROUND_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "assets", "us_states.png")
BACKGROUND_ALPHA = 0.20

def _draw_background_states(ax):
    # Never crash if missing/bad image
    if not os.path.exists(BACKGROUND_IMAGE_PATH):
        print("Background image not found:", BACKGROUND_IMAGE_PATH)
        return
    try:
        img = Image.open(BACKGROUND_IMAGE_PATH)
        w, h = img.size
        # Safety: skip if someone swaps in a huge image again
        if w * h > 180_000_000:
            print("Background image too large, skipping:", img.size)
            return
        img = img.convert("RGBA")
        ax.imshow(
            img,
            extent=(FIT_MIN_LON, FIT_MAX_LON, FIT_MIN_LAT, FIT_MAX_LAT),
            origin="upper",
            alpha=BACKGROUND_ALPHA,
            zorder=0,
            aspect="auto",
        )
    except Exception as e:
        print("Failed to draw background:", e)

# ---- Trigraph markers (FTW removed) ----
STATION_MARKERS = [
    ("SEA", -122.3301, 47.6038),
    ("PDX", -122.6765, 45.5231),
    ("SPK", -117.4260, 47.6588),
    ("MSP", -93.2650, 44.9778),
    ("FAR", -96.7898, 46.8772),
    ("GPK", -113.9980, 48.4210),
    ("SAC", -121.4944, 38.5816),
    ("RNO", -119.8138, 39.5296),
    ("DEN", -104.9903, 39.7392),
    ("GJT", -108.5506, 39.0639),
    ("GLN", -107.3248, 39.5505),
    ("CHI", -87.6300, 41.8819),
    ("KCY", -94.5786, 39.0997),
    ("ABQ", -106.6504, 35.0844),
    ("FLG", -111.6513, 35.1983),
    ("STL", -90.1994, 38.6270),
    ("LRK", -92.2896, 34.7465),
    ("DAL", -96.7970, 32.7767),
    ("AUS", -97.7431, 30.2672),
    ("SAS", -98.4936, 29.4241),
    ("NOL", -90.0715, 29.9511),
    ("ATL", -84.3880, 33.7490),
    ("WAS", -77.0067, 38.8977),
    ("NYP", -73.9940, 40.7527),
    ("BOS", -71.0589, 42.3601),
    ("MIA", -80.1918, 25.7617),
]

# ---- Scenic POIs (nudged off tracks) ----
# (label, lon, lat, dx, dy)
SCENIC_POIS = [
    ("GLACIER NP", -113.80, 48.70,  0.40,  0.20),
    ("MARIAS PASS", -113.30, 48.30,  0.35, -0.20),
    ("COLUMBIA R.", -120.00, 46.10, -0.35,  0.20),
    ("ROCKY MTNS", -106.50, 39.40,  0.45,  0.25),
    ("GLENWOOD\nCANYON", -107.20, 39.60, -0.45,  0.10),
    ("RATON PASS", -105.20, 36.90,  0.45, -0.25),
    ("MISSISSIPPI", -90.20, 35.10,  0.55, -0.10),
]

# --- Sunset Limited only NOL <-> SAS (to avoid clashing with Texas Eagle elsewhere) ---
SUBSEGMENT_LIMITS = {
    "Sunset Limited": (("NOL", -90.0715, 29.9511), ("SAS", -98.4936, 29.4241)),
}

# ---- GTFS cache (avoid re-downloading every request) ----
_GTFS_CACHE = {"fetched_at": None, "zip_bytes": None}
GTFS_MAX_AGE_MINUTES = 60

def _download_gtfs_zip_bytes() -> bytes:
    now = datetime.utcnow()
    if _GTFS_CACHE["zip_bytes"] is not None and _GTFS_CACHE["fetched_at"] is not None:
        age = (now - _GTFS_CACHE["fetched_at"]).total_seconds() / 60.0
        if age <= GTFS_MAX_AGE_MINUTES:
            return _GTFS_CACHE["zip_bytes"]

    r = requests.get(GTFS_URL, timeout=60, headers={"User-Agent": "amtrak-map-service/1.0"})
    r.raise_for_status()
    _GTFS_CACHE["zip_bytes"] = r.content
    _GTFS_CACHE["fetched_at"] = now
    return r.content

def _draw_text_label(ax, text, lon, lat):
    ax.text(
        lon, lat, text,
        fontsize=LABEL_FONTSIZE,
        ha="center", va="center",
        color="black",
        zorder=20,
        path_effects=[pe.withStroke(linewidth=LABEL_HALO_WIDTH, foreground="white")],
    )

def _draw_station_labels(ax):
    seen = set()
    for code, lon, lat in STATION_MARKERS:
        if code in seen:
            continue
        seen.add(code)
        _draw_text_label(ax, code, lon, lat)

def _draw_scenic_pois(ax):
    for label, lon, lat, dx, dy in SCENIC_POIS:
        _draw_text_label(ax, label, lon + dx, lat + dy)

def _sun_civil_times(lat: float, lon: float, tz_name: str, on_date: date):
    try:
        tz = pytz.timezone(tz_name)
    except Exception:
        tz = pytz.UTC
    loc = LocationInfo(latitude=lat, longitude=lon, timezone=getattr(tz, "zone", "UTC"))
    s = sun(loc.observer, date=on_date, tzinfo=tz)
    return s["dawn"], s["dusk"]

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

def _branch_key_for_trip(trip_id: str, stop_times: pd.DataFrame, stops: pd.DataFrame):
    st = stop_times[stop_times["trip_id"] == trip_id].merge(stops, on="stop_id", how="left")
    st = st.dropna(subset=["stop_lat", "stop_lon"]).sort_values("stop_sequence")
    if len(st) < 2:
        return None
    a = st.iloc[0]
    b = st.iloc[-1]
    p0 = (round(float(a["stop_lat"]), 1), round(float(a["stop_lon"]), 1))
    p1 = (round(float(b["stop_lat"]), 1), round(float(b["stop_lon"]), 1))
    return frozenset((p0, p1))

def _load_trips_and_stops(run_date: date):
    zip_bytes = _download_gtfs_zip_bytes()
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))

    routes = pd.read_csv(z.open("routes.txt"))
    trips = pd.read_csv(z.open("trips.txt"))
    stop_times = pd.read_csv(z.open("stop_times.txt"))
    stops = pd.read_csv(z.open("stops.txt"))
    calendar = pd.read_csv(z.open("calendar.txt"))

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

    trip_stop_counts = stop_times.groupby("trip_id").size().to_dict()

    candidates = defaultdict(list)
    for _, r in trips.iterrows():
        name = r["route_long_name"]
        direction = int(r.get("direction_id", 0))
        tid = r["trip_id"]
        bk = _branch_key_for_trip(tid, stop_times, stops)
        if bk is None:
            continue
        candidates[(name, direction, bk)].append(tid)

    routes_branches = defaultdict(lambda: defaultdict(dict))
    for (name, direction, bk), tids in candidates.items():
        best = max(tids, key=lambda t: trip_stop_counts.get(t, 0))
        routes_branches[name][bk][direction] = best

    if not routes_branches:
        raise RuntimeError("No matching long-distance trips found for that date in the GTFS feed.")

    return routes_branches, stop_times, stops

def _nearest_index_by_coord(st_df: pd.DataFrame, lon: float, lat: float) -> int:
    lons = st_df["stop_lon"].astype(float).to_numpy()
    lats = st_df["stop_lat"].astype(float).to_numpy()
    d2 = (lons - lon) ** 2 + (lats - lat) ** 2
    return int(d2.argmin())

def _apply_subsegment_if_needed(route_name: str, st_df: pd.DataFrame) -> pd.DataFrame:
    if route_name not in SUBSEGMENT_LIMITS:
        return st_df

    (_, lon1, lat1), (_, lon2, lat2) = SUBSEGMENT_LIMITS[route_name]
    if len(st_df) < 2:
        return st_df

    i1 = _nearest_index_by_coord(st_df, lon1, lat1)
    i2 = _nearest_index_by_coord(st_df, lon2, lat2)
    lo, hi = (i1, i2) if i1 <= i2 else (i2, i1)

    sliced = st_df.iloc[lo:hi + 1].copy()
    return sliced if len(sliced) >= 2 else st_df

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

def _map_seg_index(i, nseg_geom, nseg_time):
    if nseg_geom <= 1 or nseg_time <= 1:
        return min(i, max(0, nseg_time - 1))
    return int(round(i * (nseg_time - 1) / (nseg_geom - 1)))

def _draw_chevron_night_only(ax, x, y, heading_rad, colour):
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

    ax.plot([left_x, tip_x], [left_y, tip_y], color=colour, lw=CHEVRON_LW_NIGHT, alpha=CHEVRON_ALPHA_NIGHT, zorder=CHEVRON_ZORDER)
    ax.plot([right_x, tip_x], [right_y, tip_y], color=colour, lw=CHEVRON_LW_NIGHT, alpha=CHEVRON_ALPHA_NIGHT, zorder=CHEVRON_ZORDER)

def _classify_light_civil_with_sampling(origin_dt_local, origin_dep_sec: int, a_row, b_row, tz_name_here: str) -> bool:
    try:
        tz_here = pytz.timezone(tz_name_here)
    except Exception:
        tz_here = pytz.UTC

    t0 = int(a_row["dep_sec"])
    t1 = int(b_row["arr_sec"])
    if t1 < t0:
        t1 = t1 + 24 * 3600

    lat0 = float(a_row["stop_lat"])
    lon0 = float(a_row["stop_lon"])
    lat1 = float(b_row["stop_lat"])
    lon1 = float(b_row["stop_lon"])

    n = max(3, int(LIGHT_SAMPLE_POINTS_PER_SEGMENT))
    votes = 0
    for k in range(n):
        f = k / (n - 1) if n > 1 else 0.5
        t = int(round(t0 + (t1 - t0) * f))

        dt_at_origin = origin_dt_local + timedelta(seconds=(t - origin_dep_sec))
        dt_here = dt_at_origin.astimezone(tz_here)

        lat = lat0 + (lat1 - lat0) * f
        lon = lon0 + (lon1 - lon0) * f

        dawn, dusk = _sun_civil_times(lat, lon, tz_name_here, dt_here.date())
        votes += 1 if (dawn <= dt_here <= dusk) else 0

    return (votes / n) > LIGHT_MAJORITY_THRESHOLD

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

        L = math.hypot(dx, dy)
        if L == 0:
            ux, uy = 0.0, 0.0
        else:
            ux, uy = (-dy / L, dx / L)

        outx[i] = lons[i] + ux * gap_deg * side_sign
        outy[i] = lats[i] + uy * gap_deg * side_sign

    return outx, outy

def _compute_route_bounds(routes_branches, stop_times, stops):
    min_lon = 999.0
    max_lon = -999.0
    min_lat = 999.0
    max_lat = -999.0

    for name in LONG_DISTANCE_NAMES:
        if name not in routes_branches:
            continue
        for _, dir_map in routes_branches[name].items():
            center_trip = dir_map.get(0) or dir_map.get(1)
            if not center_trip:
                continue

            st_geom = stop_times[stop_times["trip_id"] == center_trip].merge(stops, on="stop_id", how="left")
            st_geom = st_geom.dropna(subset=["stop_lat", "stop_lon", "stop_timezone"]).sort_values("stop_sequence")
            st_geom = _apply_subsegment_if_needed(name, st_geom)
            if len(st_geom) < 2:
                continue

            lons = st_geom["stop_lon"].astype(float).to_numpy()
            lats = st_geom["stop_lat"].astype(float).to_numpy()

            min_lon = min(min_lon, float(lons.min()))
            max_lon = max(max_lon, float(lons.max()))
            min_lat = min(min_lat, float(lats.min()))
            max_lat = max(max_lat, float(lats.max()))

    if min_lon > 900:
        return FIT_MIN_LON, FIT_MAX_LON, FIT_MIN_LAT, FIT_MAX_LAT

    return min_lon, max_lon, min_lat, max_lat

def _make_map(run_date: date):
    routes_branches, stop_times, stops = _load_trips_and_stops(run_date)

    fig, ax = plt.subplots(figsize=(18, 10))

    if AUTO_FIT_MAP:
        min_lon, max_lon, min_lat, max_lat = _compute_route_bounds(routes_branches, stop_times, stops)

        # Pad to make room for offsets/chevrons/labels
        min_lon -= FIT_MARGIN_DEG_X
        max_lon += FIT_MARGIN_DEG_X
        min_lat -= FIT_MARGIN_DEG_Y
        max_lat += FIT_MARGIN_DEG_Y

        # Clamp to CONUS-ish window (keeps view sensible)
        min_lon = max(min_lon, FIT_MIN_LON)
        max_lon = min(max_lon, FIT_MAX_LON)
        min_lat = max(min_lat, FIT_MIN_LAT)
        max_lat = min(max_lat, FIT_MAX_LAT)

        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
    else:
        ax.set_xlim(FIT_MIN_LON, FIT_MAX_LON)
        ax.set_ylim(FIT_MIN_LAT, FIT_MAX_LAT)

    # Background goes first (always drawn in the CONUS extent; your axes may be zoomed within it)
    _draw_background_states(ax)

    ax.set_title(f"Amtrak Long-Distance Routes\nCivil Twilight (dawn to dusk) — {run_date}", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linewidth=0.3, alpha=0.20)

    # Legends
    style_handles = [
        mlines.Line2D([], [], color="black", lw=DAY_LINEWIDTH, linestyle="-", label="Light (civil twilight)"),
        mlines.Line2D([], [], color="black", lw=NIGHT_LINEWIDTH, linestyle="--", alpha=NIGHT_ALPHA, label="Dark"),
    ]
    style_leg = ax.legend(handles=style_handles, loc="lower left", fontsize=9, frameon=True)
    ax.add_artist(style_leg)

    legend_routes = [mlines.Line2D([], [], color=ROUTE_COLOURS[nm], lw=3, label=nm) for nm in LONG_DISTANCE_NAMES]
    route_leg = ax.legend(
        handles=legend_routes,
        loc="lower right",
        fontsize=8,
        frameon=True,
        title="Train services",
        title_fontsize=9,
    )
    ax.add_artist(route_leg)

    _draw_station_labels(ax)
    _draw_scenic_pois(ax)

    for name in LONG_DISTANCE_NAMES:
        if name not in routes_branches:
            continue

        colour = ROUTE_COLOURS.get(name, "#000000")

        for _, dir_map in routes_branches[name].items():
            trip0 = dir_map.get(0)
            trip1 = dir_map.get(1)
            if not trip0 and not trip1:
                continue

            center_trip = trip0 or trip1

            st_geom = stop_times[stop_times["trip_id"] == center_trip].merge(stops, on="stop_id", how="left")
            st_geom = st_geom.dropna(subset=["stop_lat", "stop_lon", "stop_timezone"]).sort_values("stop_sequence")
            st_geom = _apply_subsegment_if_needed(name, st_geom)
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
                st_time = st_time.dropna(subset=["dep_sec", "arr_sec", "stop_timezone", "stop_lat", "stop_lon"]).sort_values("stop_sequence")
                st_time = _apply_subsegment_if_needed(name, st_time)
                if len(st_time) < 2:
                    continue

                origin_dt_local, origin_dep_sec = _trip_anchor_datetime(run_date, st_time)

                # Reverse drawn geometry for direction 1 so chevrons point opposite
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

                    if a["dep_sec"] is None or b["arr_sec"] is None:
                        continue

                    tz_name_here = a["stop_timezone"]
                    is_light = _classify_light_civil_with_sampling(origin_dt_local, origin_dep_sec, a, b, tz_name_here)

                    x0, y0 = xs[i], ys[i]
                    x1, y1 = xs[i + 1], ys[i + 1]

                    ax.plot(
                        [x0, x1],
                        [y0, y1],
                        color=colour,
                        linewidth=DAY_LINEWIDTH if is_light else NIGHT_LINEWIDTH,
                        linestyle="-" if is_light else "--",
                        alpha=1.0 if is_light else NIGHT_ALPHA,
                        zorder=5,
                    )

                    # Chevrons NIGHT ONLY + keep away from ends
                    if not is_light:
                        near_start = i < CHEVRON_SKIP_END_SEGMENTS
                        near_end = i > (nseg - 1 - CHEVRON_SKIP_END_SEGMENTS)
                        if (i % CHEVRON_EVERY_N_SEGMENTS == 0) and (not near_start) and (not near_end):
                            heading = math.atan2((y1 - y0), (x1 - x0))
                            _draw_chevron_night_only(ax, (x0 + x1) / 2.0, (y0 + y1) / 2.0, heading, colour)

    fig.tight_layout()
    return fig

def build_png_bytes(run_date: date) -> bytes:
    fig = _make_map(run_date)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

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
