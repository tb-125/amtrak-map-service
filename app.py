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

GTFS_URL = "https://content.amtrak.com/content/gtfs/GTFS.zip"

CACHE_DIR = Path(os.environ.get("CACHE_DIR", "./cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GTFS_CACHE_PATH = CACHE_DIR / "GTFS.zip"
GTFS_CACHE_MAX_AGE_HOURS = int(os.environ.get("GTFS_CACHE_MAX_AGE_HOURS", "24"))

FIGSIZE = (16, 9)
DPI = 160

X_MIN, X_MAX = -125, -66
Y_MIN, Y_MAX = 24, 50

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

COLOR_MAP = {name: plt.get_cmap("tab20")(i) for i, name in enumerate(LONG_DISTANCE_NAMES)}

MAJOR_CITIES = {
    "Seattle": (-122.3321, 47.6062),
    "Portland": (-122.6765, 45.5231),
    "San Francisco": (-122.4194, 37.7749),
    "Los Angeles": (-118.2437, 34.0522),
    "San Diego": (-117.1611, 32.7157),
    "Phoenix": (-112.0740, 33.4484),
    "Denver": (-104.9903, 39.7392),
    "Salt Lake City": (-111.8910, 40.7608),
    "Chicago": (-87.6298, 41.8781),
    "St. Louis": (-90.1994, 38.6270),
    "Dallas": (-96.7970, 32.7767),
    "Houston": (-95.3698, 29.7604),
    "New Orleans": (-90.0715, 29.9511),
    "Atlanta": (-84.3880, 33.7490),
    "Washington, DC": (-77.0369, 38.9072),
    "Philadelphia": (-75.1652, 39.9526),
    "New York": (-74.0060, 40.7128),
    "Boston": (-71.0589, 42.3601),
    "Miami": (-80.1918, 25.7617),
}

CITY_OFFSETS = {
    "Washington, DC": (0.7, -0.2),
    "Philadelphia": (0.6, 0.2),
    "New York": (0.7, 0.35),
    "Boston": (0.55, 0.25),
    "Chicago": (-0.75, 0.25),
}

SCENIC_LABELS = [
    ("Puget Sound", -122.6, 47.2),
    ("Cascade Range", -121.3, 47.0),
    ("Columbia River Gorge", -121.7, 45.7),
    ("Sierra Nevada", -120.4, 39.3),
    ("Rocky Mountains", -106.2, 39.5),
    ("Moffat Tunnel / Front Range", -105.8, 39.9),
    ("Marias Pass (Glacier area)", -113.5, 48.2),
    ("Mississippi River (Midwest)", -91.2, 41.2),
    ("Mississippi River (South)", -90.4, 35.2),
    ("Cumberland Plateau / Appalachians", -84.7, 36.6),
    ("New River Gorge region", -81.1, 38.0),
    ("Hudson River Valley", -73.9, 41.6),
    ("Potomac River", -77.3, 39.0),
    ("Arizona / New Mexico Desert", -110.8, 35.2),
    ("Santa Barbara Coast", -120.2, 34.4),
    ("Gulf Coast", -89.5, 30.3),
    ("Florida Coast / Everglades region", -80.6, 26.2),
]

CHEVRON_SCALE = 1.5
CHEVRON_LW = 0.8
CHEVRON_ALPHA = 0.9


def parse_gtfs_time(t):
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
    loc = LocationInfo(latitude=lat, longitude=lon, timezone=tz.zone if hasattr(tz, "zone") else "UTC")
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

            if d_route < 0.4 or d_label < 0.5:
                continue

            if score > best_score:
                best_score = score
                best = (x, y)

    return best


def draw_chevron(ax, tip_x, tip_y, angle_rad, color, size, width, lw=CHEVRON_LW, alpha=CHEVRON_ALPHA, zorder=4):
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


def add_route_chevrons(ax, route_midpoints, color):
    if len(route_midpoints) < 8:
        return

    start_skip = max(3, len(route_midpoints) // 12)
    mids = route_midpoints[start_skip : len(route_midpoints) - start_skip]
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
    counts = stop_times_df.groupby("trip_id").size()
    reps = {}
    for (rname, direction), grp in trips_df.groupby(["route_long_name", "direction_id"]):
        sub = counts[counts.index.isin(grp["trip_id"])]
        if len(sub):
            reps[(rname, int(direction))] = sub.sort_values(ascending=False).index[0]
    return reps


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
    active = {r["service_id"] for _, r in calendar.iterrows() if service_active_on(r, yyyymmdd)}

    routes = routes[routes["route_long_name"].isin(LONG_DISTANCE_NAMES)]
    trips = trips[trips["service_id"].isin(active)]
    trips = trips.merge(routes[["route_id","route_long_name"]], on="route_id", how="inner")

    stop_times = stop_times[stop_times["trip_id"].isin(trips["trip_id"])]
    stop_times["arr_sec"] = stop_times["arrival_time"].apply(parse_gtfs_time)
    stop_times["dep_sec"] = stop_times["departure_time"].apply(parse_gtfs_time)
    stop_times = stop_times.sort_values(["trip_id","stop_sequence"])

    stops = stops[["stop_id","stop_lat","stop_lon","stop_timezone"]]
    stops["stop_timezone"] = stops["stop_timezone"].fillna("UTC")

    reps = pick_representative_trips(trips, stop_times)
    direction_id = direction_id_from_param(dir_param)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.set_title(f"Amtrak Long-Distance Routes — Daylight vs Darkness — {run_date} — {direction_label(dir_param)}")

    placed_pts = []
    for city, (lon, lat) in MAJOR_CITIES.items():
        dx, dy = CITY_OFFSETS.get(city, (0.25, 0.15))
        x, y = lon + dx, lat + dy
        ax.text(x, y, city, fontsize=8, color="0.55", ha="left", va="center", zorder=1)
        placed_pts.append((x, y))

    route_pts = []

    for rname in LONG_DISTANCE_NAMES:
        key = (rname, direction_id)
        if key not in reps:
            continue

        trip_id = reps[key]
        color = COLOR_MAP[rname]

        st = stop_times[stop_times["trip_id"] == trip_id].merge(stops, on="stop_id").dropna()
        if len(st) < 2:
            continue

        chevrons = []

        for i in range(len(st) - 1):
            a, b = st.iloc[i], st.iloc[i + 1]
            if a["dep_sec"] is None or b["arr_sec"] is None:
                continue

            ax0, ay0 = float(a["stop_lon"]), float(a["stop_lat"])
            bx0, by0 = float(b["stop_lon"]), float(b["stop_lat"])
            mx, my = (ax0 + bx0) / 2.0, (ay0 + by0) / 2.0

            ang = math.atan2(by0 - ay0, bx0 - ax0)
            seg_len = math.hypot(bx0 - ax0, by0 - ay0)

            route_pts.extend([(ax0, ay0), (mx, my), (bx0, by0)])
            chevrons.append((mx, my, ang, seg_len))

            mid = int((a["dep_sec"] + b["arr_sec"]) / 2)
            dt = datetime(run_date.year, run_date.month, run_date.day) + timedelta(seconds=mid)
            try:
                dt = pytz.timezone(a["stop_timezone"]).localize(dt)
            except Exception:
                dt = pytz.UTC.localize(dt)

            daylight = is_daylight(
                (a["stop_lat"] + b["stop_lat"]) / 2,
                (a["stop_lon"] + b["stop_lon"]) / 2,
                a["stop_timezone"],
                dt,
            )

            ax.plot(
                [ax0, bx0],
                [ay0, by0],
                color=color,
                linewidth=2.6 if daylight else 1.1,
                linestyle="-" if daylight else "--",
                alpha=1.0 if daylight else 0.6,
                zorder=3 if daylight else 2,
            )

        add_route_chevrons(ax, chevrons, color=color)

    for label, lon, lat in SCENIC_LABELS:
        x, y = best_label_position((lon, lat), route_pts, placed_pts)
        ax.text(x, y, label, fontsize=8, fontstyle="italic", color="0.45", ha="left", va="center", zorder=1)
        placed_pts.append((x, y))

    handles = [mlines.Line2D([], [], color=COLOR_MAP[n], linewidth=3, label=n) for n in LONG_DISTANCE_NAMES]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False, title="Routes (colour key)")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


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
