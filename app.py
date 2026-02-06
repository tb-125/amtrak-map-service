import os
import io
import json
import math
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import pytz
from astral import LocationInfo
from astral.sun import sun

from flask import Flask, request, send_file, abort, make_response, Response

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from PIL import Image


# ============================================================
# Configuration
# ============================================================

GTFS_URL = "https://content.amtrak.com/content/gtfs/GTFS.zip"

CACHE_DIR = Path(os.environ.get("CACHE_DIR", "./cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GTFS_CACHE_PATH = CACHE_DIR / "GTFS.zip"
GTFS_CACHE_MAX_AGE_HOURS = int(os.environ.get("GTFS_CACHE_MAX_AGE_HOURS", "24"))

BASEMAP_PATH = os.environ.get("BASEMAP_PATH", "assets/us_basemap.png")

# Map bounds (Lower 48)
X_MIN, X_MAX = -125, -66
Y_MIN, Y_MAX = 24, 50

FIGSIZE = (16, 9)
DPI = 170

DAY_COLOR = "#d21f1f"     # red
NIGHT_COLOR = "#1f3aa6"   # blue
LINE_WIDTH = 3.2

STATION_DOT_SIZE = 18
STATION_DOT_EDGE = 0.9

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

MAJOR_CITIES = {
    "Seattle": (-122.33, 47.61),
    "Portland": (-122.68, 45.52),
    "San Francisco": (-122.42, 37.77),
    "Los Angeles": (-118.24, 34.05),
    "Denver": (-104.99, 39.74),
    "Chicago": (-87.63, 41.88),
    "New York": (-74.00, 40.71),
    "Washington, DC": (-77.04, 38.91),
    "Miami": (-80.19, 25.76),
}

CITY_OFFSETS = {
    "Chicago": (-0.8, 0.3),
    "New York": (0.6, 0.3),
    "Washington, DC": (0.6, -0.2),
}


# ============================================================
# Helper functions
# ============================================================

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

    loc = LocationInfo(latitude=lat, longitude=lon, timezone=tz.zone)
    if dt_local.tzinfo is None:
        dt_local = tz.localize(dt_local)

    s = sun(loc.observer, date=dt_local.date(), tzinfo=dt_local.tzinfo)
    return s["sunrise"] <= dt_local <= s["sunset"]


def gtfs_is_fresh(path):
    if not path.exists():
        return False
    age = datetime.utcnow() - datetime.utcfromtimestamp(path.stat().st_mtime)
    return age.total_seconds() < GTFS_CACHE_MAX_AGE_HOURS * 3600


def get_gtfs_zip():
    if not gtfs_is_fresh(GTFS_CACHE_PATH):
        r = requests.get(GTFS_URL, timeout=60)
        r.raise_for_status()
        GTFS_CACHE_PATH.write_bytes(r.content)
    return zipfile.ZipFile(GTFS_CACHE_PATH)


def read_txt(z, name):
    with z.open(name) as f:
        return pd.read_csv(f)


def direction_id_from_param(dir_param):
    return 0 if dir_param == "west" else 1


def direction_label(dir_param):
    return "Westbound / Northbound" if dir_param == "west" else "Eastbound / Southbound"


def pick_representative_trips(trips_df, stop_times_df):
    counts = stop_times_df.groupby("trip_id").size()
    reps = {}
    for (name, direction), grp in trips_df.groupby(["route_long_name", "direction_id"]):
        sub = counts[counts.index.isin(grp["trip_id"])]
        if len(sub):
            reps[(name, int(direction))] = sub.sort_values(ascending=False).index[0]
    return reps


# ============================================================
# PNG Renderer
# ============================================================

def render_map_png(date_str, dir_param):
    run_date = datetime.strptime(date_str, "%Y-%m-%d").date()

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
    trips = trips.merge(routes[["route_id","route_long_name"]], on="route_id")

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
    ax.set_axis_off()

    if os.path.exists(BASEMAP_PATH):
        img = Image.open(BASEMAP_PATH)
        ax.imshow(img, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], aspect="auto", zorder=0)
    else:
        ax.set_facecolor("#eef6fb")

    fig.suptitle(
        "Amtrak long-distance trains by night and day",
        fontsize=20, fontweight="bold", color=DAY_COLOR, y=0.96
    )
    ax.set_title(f"{run_date} — {direction_label(dir_param)}", fontsize=12, color="0.3")

    for city, (lon, lat) in MAJOR_CITIES.items():
        dx, dy = CITY_OFFSETS.get(city, (0.25, 0.15))
        ax.text(lon + dx, lat + dy, city, fontsize=8, color="0.25", zorder=2)

    for rname in LONG_DISTANCE_NAMES:
        key = (rname, direction_id)
        if key not in reps:
            continue

        trip_id = reps[key]
        st = stop_times[stop_times["trip_id"] == trip_id].merge(stops, on="stop_id")
        if len(st) < 2:
            continue

        for i in range(len(st) - 1):
            a, b = st.iloc[i], st.iloc[i+1]
            if a["dep_sec"] is None or b["arr_sec"] is None:
                continue

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
                dt
            )

            ax.plot(
                [a["stop_lon"], b["stop_lon"]],
                [a["stop_lat"], b["stop_lat"]],
                color=DAY_COLOR if daylight else NIGHT_COLOR,
                linewidth=LINE_WIDTH,
                zorder=3
            )

        ax.scatter(
            st["stop_lon"], st["stop_lat"],
            s=STATION_DOT_SIZE,
            facecolors="white",
            edgecolors="0.15",
            linewidths=STATION_DOT_EDGE,
            zorder=4
        )

    legend = [
        mlines.Line2D([], [], color=DAY_COLOR, linewidth=4, label="Daylight"),
        mlines.Line2D([], [], color=NIGHT_COLOR, linewidth=4, label="Darkness"),
    ]
    ax.legend(handles=legend, loc="lower left", frameon=True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# GeoJSON Renderer
# ============================================================

def render_map_geojson(date_str, dir_param):
    run_date = datetime.strptime(date_str, "%Y-%m-%d").date()

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
    trips = trips.merge(routes[["route_id","route_long_name"]], on="route_id")

    stop_times = stop_times[stop_times["trip_id"].isin(trips["trip_id"])]
    stop_times["arr_sec"] = stop_times["arrival_time"].apply(parse_gtfs_time)
    stop_times["dep_sec"] = stop_times["departure_time"].apply(parse_gtfs_time)
    stop_times = stop_times.sort_values(["trip_id","stop_sequence"])

    stops = stops[["stop_id","stop_lat","stop_lon","stop_timezone"]]
    stops["stop_timezone"] = stops["stop_timezone"].fillna("UTC")

    reps = pick_representative_trips(trips, stop_times)
    direction_id = direction_id_from_param(dir_param)

    features = []

    for rname in LONG_DISTANCE_NAMES:
        key = (rname, direction_id)
        if key not in reps:
            continue

        trip_id = reps[key]
        st = stop_times[stop_times["trip_id"] == trip_id].merge(stops, on="stop_id")
        if len(st) < 2:
            continue

        for i in range(len(st) - 1):
            a, b = st.iloc[i], st.iloc[i+1]
            if a["dep_sec"] is None or b["arr_sec"] is None:
                continue

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
                dt
            )

            features.append({
                "type": "Feature",
                "properties": {
                    "route": rname,
                    "direction": direction_label(dir_param),
                    "date": date_str,
                    "day": bool(daylight)
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [a["stop_lon"], a["stop_lat"]],
                        [b["stop_lon"], b["stop_lat"]]
                    ]
                }
            })

    return {
        "type": "FeatureCollection",
        "features": features
    }


# ============================================================
# Flask app
# ============================================================

app = Flask(__name__)


@app.get("/")
def home():
    return {"ok": True}


@app.get("/amtrak-map.png")
def map_png():
    date_str = request.args.get("date")
    dir_param = request.args.get("dir", "west")
    if not date_str:
        abort(400)

    out = CACHE_DIR / f"amtrak_{date_str}_{dir_param}.png"
    if out.exists():
        return send_file(out, mimetype="image/png")

    png = render_map_png(date_str, dir_param)
    out.write_bytes(png)
    return make_response(png, 200, {"Content-Type": "image/png"})


@app.get("/amtrak-map.geojson")
def map_geojson():
    date_str = request.args.get("date")
    dir_param = request.args.get("dir", "west")
    if not date_str:
        abort(400)

    out = CACHE_DIR / f"amtrak_{date_str}_{dir_param}.geojson"
    if out.exists():
        return send_file(out, mimetype="application/geo+json")

    geo = render_map_geojson(date_str, dir_param)
    data = json.dumps(geo).encode("utf-8")
    out.write_bytes(data)
    return Response(data, mimetype="application/geo+json")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
