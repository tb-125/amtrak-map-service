import os
import io
import math
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import pytz
from astral import LocationInfo
from astral.sun import sun

from flask import Flask, request, send_file, abort, make_response

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from PIL import Image  # <-- requires Pillow in requirements.txt


# ----------------------------
# Config
# ----------------------------

GTFS_URL = "https://content.amtrak.com/content/gtfs/GTFS.zip"

CACHE_DIR = Path(os.environ.get("CACHE_DIR", "./cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GTFS_CACHE_PATH = CACHE_DIR / "GTFS.zip"
GTFS_CACHE_MAX_AGE_HOURS = int(os.environ.get("GTFS_CACHE_MAX_AGE_HOURS", "24"))

# Basemap image (add this file to your repo): assets/us_basemap.png
BASEMAP_PATH = os.environ.get("BASEMAP_PATH", "assets/us_basemap.png")

# Plot extent (Lower 48-ish)
X_MIN, X_MAX = -125, -66
Y_MIN, Y_MAX = 24, 50

# Render output
FIGSIZE = (16, 9)
DPI = 170

# Day/Night styling (like your reference)
DAY_COLOR = "#d21f1f"    # red
NIGHT_COLOR = "#1f3aa6"  # blue
DAY_LW = 3.2
NIGHT_LW = 3.2

# Station dots
STATION_DOT_SIZE = 18
STATION_DOT_EDGE_LW = 0.9

# Chevrons (optional — keep subtle)
DRAW_CHEVRONS = True
CHEVRON_SCALE = 1.0
CHEVRON_LW = 0.9
CHEVRON_ALPHA = 0.9

# Inset (East/Florida)
DRAW_INSET = True
INSET_BOUNDS = (-92, -74, 24, 42)  # x0,x1,y0,y1

# Which routes to show (GTFS route_long_name must match)
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

# Major city labels (subtle, map-like)
MAJOR_CITIES = {
    "Seattle": (-122.3321, 47.6062),
    "Portland": (-122.6765, 45.5231),
    "San Francisco": (-122.4194, 37.7749),
    "Los Angeles": (-118.2437, 34.0522),
    "San Diego": (-117.1611, 32.7157),
    "Denver": (-104.9903, 39.7392),
    "Salt Lake City": (-111.8910, 40.7608),
    "Chicago": (-87.6298, 41.8781),
    "St. Louis": (-90.1994, 38.6270),
    "Dallas": (-96.7970, 32.7767),
    "Houston": (-95.3698, 29.7604),
    "New Orleans": (-90.0715, 29.9511),
    "Atlanta": (-84.3880, 33.7490),
    "Washington, DC": (-77.0369, 38.9072),
    "New York": (-74.0060, 40.7128),
    "Boston": (-71.0589, 42.3601),
    "Miami": (-80.1918, 25.7617),
}

CITY_OFFSETS = {
    "Washington, DC": (0.7, -0.2),
    "New York": (0.6, 0.35),
    "Boston": (0.55, 0.25),
    "Chicago": (-0.75, 0.25),
}

# Scenic labels (optional; italic like you wanted previously)
SCENIC_LABELS = [
    ("Puget Sound", -122.6, 47.2),
    ("Cascade Range", -121.3, 47.0),
    ("Columbia River Gorge", -121.7, 45.7),
    ("Sierra Nevada", -120.4, 39.3),
    ("Rocky Mountains", -106.2, 39.5),
    ("Marias Pass (Glacier)", -113.5, 48.2),
    ("Hudson River Valley", -73.9, 41.6),
    ("Cumberland / Appalachians", -84.7, 36.6),
    ("Gulf Coast", -89.5, 30.3),
    ("Florida Coast", -80.6, 26.2),
]


# ----------------------------
# Helpers
# ----------------------------

def parse_gtfs_time(t):
    """Return seconds since midnight. GTFS can be 24+ hours (e.g., 25:10:00)."""
    if pd.isna(t) or not isinstance(t, str):
        return None
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def service_active_on(row, yyyymmdd):
    d = datetime.strptime(yyyymmdd, "%Y%m%d").date()
    start = datetime.strptime(str(row["start_date"]), "%Y%m%d").date()
    end = datetime.strptime(str(row["end_date"]), "%Y%m%d").date()
    if not (start <= d <= end):
        return False
    weekday = d.weekday()
    cols = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    return int(row[cols[weekday]]) == 1


def is_daylight(lat, lon, tz_name, dt_local):
    try:
        tz = pytz.timezone(tz_name)
    except Exception:
        tz = pytz.UTC
    loc = LocationInfo(latitude=float(lat), longitude=float(lon), timezone=getattr(tz, "zone", "UTC"))
    if dt_local.tzinfo is None:
        dt_local = tz.localize(dt_local)
    s = sun(loc.observer, date=dt_local.date(), tzinfo=dt_local.tzinfo)
    return s["sunrise"] <= dt_local <= s["sunset"]


def gtfs_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.utcnow() - datetime.utcfromtimestamp(path.stat().st_mtime)
    return age.total_seconds() < (GTFS_CACHE_MAX_AGE_HOURS * 3600)


def get_gtfs_zip() -> zipfile.ZipFile:
    if not gtfs_is_fresh(GTFS_CACHE_PATH):
        r = requests.get(GTFS_URL, timeout=60)
        r.raise_for_status()
        GTFS_CACHE_PATH.write_bytes(r.content)
    return zipfile.ZipFile(GTFS_CACHE_PATH)


def read_txt(z, name):
    with z.open(name) as f:
        return pd.read_csv(f)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def min_dist_to_points(p, pts):
    if not pts:
        return 999.0
    px, py = p
    best = 999.0
    for x, y in pts:
        d = math.hypot(px - x, py - y)
        if d < best:
            best = d
    return best


def best_label_position(anchor, route_pts, placed_label_pts):
    """Pick a label position away from route lines + other labels."""
    ax0, ay0 = anchor
    radii = [0.7, 1.0, 1.3, 1.6]
    directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    best = (ax0 + 1.0, ay0 + 0.6)
    best_score = -1e9

    for r in radii:
        for dx, dy in directions:
            mag = math.hypot(dx, dy)
            x = clamp(ax0 + (dx / mag) * r, X_MIN + 0.7, X_MAX - 0.7)
            y = clamp(ay0 + (dy / mag) * r, Y_MIN + 0.7, Y_MAX - 0.7)

            d_route = min_dist_to_points((x, y), route_pts)
            d_label = min_dist_to_points((x, y), placed_label_pts)
            score = 3 * d_route + 2 * d_label - r

            # hard constraints
            if d_route < 0.35 or d_label < 0.55:
                continue

            if score > best_score:
                best_score = score
                best = (x, y)

    return best


def draw_chevron(ax, tip_x, tip_y, angle_rad, color, size, width, lw=CHEVRON_LW, alpha=CHEVRON_ALPHA, zorder=5):
    ux = math.cos(angle_rad)
    uy = math.sin(angle_rad)

    bx = tip_x - ux * size
    by = tip_y - uy * size

    px = -uy
    py = ux

    x1 = bx + px * width
    y1 = by + py * width
    x2 = bx - px * width
    y2 = by - py * width

    ax.plot([x1, tip_x], [y1, tip_y], color=color, linewidth=lw, alpha=alpha, zorder=zorder)
    ax.plot([x2, tip_x], [y2, tip_y], color=color, linewidth=lw, alpha=alpha, zorder=zorder)


def add_route_chevrons(ax, midpoints, color):
    if not DRAW_CHEVRONS:
        return
    if len(midpoints) < 8:
        return

    start_skip = max(3, len(midpoints) // 12)
    mids = midpoints[start_skip : len(midpoints) - start_skip]
    if len(mids) < 3:
        return

    step = 10 if len(mids) > 80 else 7 if len(mids) > 45 else 5

    for i in range(0, len(mids), step):
        mx, my, ang, seg_len = mids[i]
        base_size = max(0.10, min(0.22, seg_len * 0.55))
        size = base_size * CHEVRON_SCALE
        width = (size * 0.45) * CHEVRON_SCALE
        draw_chevron(ax, mx, my, ang, color=color, size=size, width=width)


def direction_label(dir_param: str) -> str:
    return "Westbound / Northbound" if dir_param == "west" else "Eastbound / Southbound"


def direction_id_from_param(dir_param: str) -> int:
    return 0 if dir_param == "west" else 1


def pick_representative_trips(trips_df, stop_times_df):
    """Pick the trip with the most stops per (route, direction) as a representative geometry."""
    counts = stop_times_df.groupby("trip_id").size()
    reps = {}
    for (rname, direction), grp in trips_df.groupby(["route_long_name", "direction_id"]):
        sub = counts[counts.index.isin(grp["trip_id"])]
        if len(sub):
            reps[(rname, int(direction))] = sub.sort_values(ascending=False).index[0]
    return reps


def in_bounds(lon, lat, x0, x1, y0, y1):
    return (x0 <= lon <= x1) and (y0 <= lat <= y1)


# ----------------------------
# Renderer
# ----------------------------

def render_map_png(date_str: str, dir_param: str) -> bytes:
    run_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    if dir_param not in ("west", "east"):
        raise ValueError("dir must be west or east")

    z = get_gtfs_zip()

    routes = read_txt(z, "routes.txt")
    trips = read_txt(z, "trips.txt")
    stop_times = read_txt(z, "stop_times.txt")
    stops = read_txt(z, "stops.txt")
    calendar = read_txt(z, "calendar.txt")

    yyyymmdd = run_date.strftime("%Y%m%d")
    active_services = {r["service_id"] for _, r in calendar.iterrows() if service_active_on(r, yyyymmdd)}

    routes = routes[routes["route_long_name"].isin(LONG_DISTANCE_NAMES)]
    trips = trips[trips["service_id"].isin(active_services)]
    trips = trips.merge(routes[["route_id", "route_long_name"]], on="route_id", how="inner")

    stop_times = stop_times[stop_times["trip_id"].isin(trips["trip_id"])]
    stop_times["arr_sec"] = stop_times["arrival_time"].apply(parse_gtfs_time)
    stop_times["dep_sec"] = stop_times["departure_time"].apply(parse_gtfs_time)
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

    stops = stops[["stop_id", "stop_lat", "stop_lon", "stop_timezone"]]
    stops["stop_timezone"] = stops["stop_timezone"].fillna("UTC")

    reps = pick_representative_trips(trips, stop_times)
    direction_id = direction_id_from_param(dir_param)

    # --- Figure setup (map-like) ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_axis_off()

    img = None
    if os.path.exists(BASEMAP_PATH):
        img = Image.open(BASEMAP_PATH)
        ax.imshow(img, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], aspect="auto", zorder=0)
    else:
        ax.set_facecolor("#f4fbff")

    fig.suptitle(
        "Amtrak long-distance trains by night and day",
        fontsize=20,
        fontweight="bold",
        color=DAY_COLOR,
        y=0.97,
    )
    ax.set_title(f"{run_date} — {direction_label(dir_param)}", fontsize=12, pad=12, color="0.25")

    # Inset
    inset = None
    x0, x1, y0, y1 = INSET_BOUNDS
    if DRAW_INSET:
        inset = ax.inset_axes([0.70, 0.08, 0.28, 0.40])
        inset.set_xlim(x0, x1)
        inset.set_ylim(y0, y1)
        inset.set_axis_off()
        if img is not None:
            inset.imshow(img, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], aspect="auto", zorder=0)
        else:
            inset.set_facecolor("#f4fbff")

    # Labels placement helpers
    placed_label_pts = []

    # City labels (subtle)
    for city, (lon, lat) in MAJOR_CITIES.items():
        dx, dy = CITY_OFFSETS.get(city, (0.25, 0.15))
        x, y = lon + dx, lat + dy
        ax.text(x, y, city, fontsize=8, color="0.20", alpha=0.75, ha="left", va="center", zorder=2)
        placed_label_pts.append((x, y))

    route_pts = []  # for keeping scenic labels away from lines

    # --- Plot routes ---
    for rname in LONG_DISTANCE_NAMES:
        key = (rname, direction_id)
        if key not in reps:
            continue

        trip_id = reps[key]

        st = stop_times[stop_times["trip_id"] == trip_id].merge(stops, on="stop_id").dropna()
        if len(st) < 2:
            continue

        chevrons = []
        chevrons_inset = []

        # draw segments
        for i in range(len(st) - 1):
            a, b = st.iloc[i], st.iloc[i + 1]
            if a["dep_sec"] is None or b["arr_sec"] is None:
                continue

            ax0p, ay0p = float(a["stop_lon"]), float(a["stop_lat"])
            bx0p, by0p = float(b["stop_lon"]), float(b["stop_lat"])

            mx, my = (ax0p + bx0p) / 2.0, (ay0p + by0p) / 2.0
            ang = math.atan2(by0p - ay0p, bx0p - ax0p)
            seg_len = math.hypot(bx0p - ax0p, by0p - ay0p)

            route_pts.extend([(ax0p, ay0p), (mx, my), (bx0p, by0p)])
            chevrons.append((mx, my, ang, seg_len))

            if inset is not None and (in_bounds(ax0p, ay0p, x0, x1, y0, y1) or in_bounds(bx0p, by0p, x0, x1, y0, y1)):
                chevrons_inset.append((mx, my, ang, seg_len))

            # midpoint time (local to stop timezone)
            mid = int((a["dep_sec"] + b["arr_sec"]) / 2)
            dt = datetime(run_date.year, run_date.month, run_date.day) + timedelta(seconds=mid)
            try:
                dt = pytz.timezone(a["stop_timezone"]).localize(dt)
            except Exception:
                dt = pytz.UTC.localize(dt)

            daylight = is_daylight(
                (float(a["stop_lat"]) + float(b["stop_lat"])) / 2.0,
                (float(a["stop_lon"]) + float(b["stop_lon"])) / 2.0,
                a["stop_timezone"],
                dt,
            )

            color = DAY_COLOR if daylight else NIGHT_COLOR
            lw = DAY_LW if daylight else NIGHT_LW

            ax.plot([ax0p, bx0p], [ay0p, by0p], color=color, linewidth=lw, alpha=1.0, zorder=3)

            if inset is not None and (in_bounds(ax0p, ay0p, x0, x1, y0, y1) or in_bounds(bx0p, by0p, x0, x1, y0, y1)):
                inset.plot([ax0p, bx0p], [ay0p, by0p], color=color, linewidth=lw, alpha=1.0, zorder=3)

        # station dots (white)
        ax.scatter(
            st["stop_lon"].astype(float),
            st["stop_lat"].astype(float),
            s=STATION_DOT_SIZE,
            facecolors="white",
            edgecolors="0.15",
            linewidths=STATION_DOT_EDGE_LW,
            zorder=4,
        )

        if inset is not None:
            # Only dots in inset window
            st_in = st[
                (st["stop_lon"].astype(float) >= x0) &
                (st["stop_lon"].astype(float) <= x1) &
                (st["stop_lat"].astype(float) >= y0) &
                (st["stop_lat"].astype(float) <= y1)
            ]
            if len(st_in):
                inset.scatter(
                    st_in["stop_lon"].astype(float),
                    st_in["stop_lat"].astype(float),
                    s=STATION_DOT_SIZE,
                    facecolors="white",
                    edgecolors="0.15",
                    linewidths=STATION_DOT_EDGE_LW,
                    zorder=4,
                )

        # chevrons: use a neutral dark (so they don't compete with day/night)
        add_route_chevrons(ax, chevrons, color="0.10")
        if inset is not None:
            add_route_chevrons(inset, chevrons_inset, color="0.10")

    # Scenic labels (italic, subtle)
    for label, lon, lat in SCENIC_LABELS:
        x, y = best_label_position((lon, lat), route_pts, placed_label_pts)
        ax.text(x, y, label, fontsize=8, fontstyle="italic", color="0.25", alpha=0.75, ha="left", va="center", zorder=2)
        placed_label_pts.append((x, y))

    # Day/Night legend (like the reference)
    day_handle = mlines.Line2D([], [], color=DAY_COLOR, linewidth=4, label="Daylight")
    night_handle = mlines.Line2D([], [], color=NIGHT_COLOR, linewidth=4, label="Darkness")
    ax.legend(handles=[day_handle, night_handle], loc="lower left", frameon=True, fontsize=10)

    # Render to PNG bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ----------------------------
# Web app
# ----------------------------

app = Flask(__name__)


@app.get("/")
def home():
    return {"ok": True, "map": "/amtrak-map.png?date=YYYY-MM-DD&dir=west|east"}


@app.get("/amtrak-map.png")
def amtrak_map_png():
    date_str = request.args.get("date", "").strip()
    dir_param = request.args.get("dir", "west").lower().strip()

    if not date_str:
        abort(400, description="Missing date=YYYY-MM-DD")

    cache_key = f"amtrak_{date_str}_{dir_param}.png"
    out_path = CACHE_DIR / cache_key

    # cache within the container lifetime
    if out_path.exists():
        return send_file(out_path, mimetype="image/png")

    try:
        png_bytes = render_map_png(date_str, dir_param)
    except ValueError as e:
        abort(400, description=str(e))
    except Exception as e:
        abort(500, description=f"Failed to render map: {e}")

    out_path.write_bytes(png_bytes)

    resp = make_response(png_bytes)
    resp.headers["Content-Type"] = "image/png"
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
