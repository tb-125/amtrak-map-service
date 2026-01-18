import os
import io
import json
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

# Prefer local basemap (add this file to your repo!)
LOCAL_STATES_GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "data", "us-states.json")

# Only used as fallback if local file is missing
US_STATES_GEOJSON_URLS = [
    "https://cdn.jsdelivr.net/gh/PublicaMundi/MappingAPI@master/data/us-states.json",
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/us-states.json",
    "https://eric.clst.org/assets/wiki/uploads/Stuff/us-states.json",
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

# ---- Day/Night styling ----
DAY_LINEWIDTH = 2.6
NIGHT_LINEWIDTH = 1.1
NIGHT_ALPHA = 0.55

# ---- Direction separation (YOUR GAP) ----
# Bigger number = bigger parallel-track separation.
# This is now a smooth polyline offset, so it won't "break apart" like before.
DIRECTION_GAP_DEG = 0.28  # try 0.24–0.34

# ---- Route labels ----
ROUTE_LABEL_FONTSIZE = 6.3
LABEL_HALO_WIDTH = 2.2
LABEL_EXTRA_OFFSET_DEG = 0.30  # push labels outward with the wider tracks

# ---- Chevrons (unfilled) ----
CHEVRON_EVERY_N_SEGMENTS = 18
CHEVRON_SKIP_END_SEGMENTS = 6
CHEVRON_SIZE_DEG = 0.085
CHEVRON_ANGLE_DEG = 24
CHEVRON_LW_DAY = 0.35
CHEVRON_LW_NIGHT = 0.28
CHEVRON_ALPHA_DAY = 0.55
CHEVRON_ALPHA_NIGHT = 0.35

# ---- Fixed colours ----
HEX_PALETTE = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
    "#66a61e", "#e6ab02", "#a6761d", "#666666",
    "#1f78b4", "#b2df8a", "#fb9a99", "#fdbf6f",
    "#cab2d6", "#6a3d9a",
]
ROUTE_COLOURS = {name: HEX_PALETTE[i % len(HEX_PALETTE)] for i, name in enumerate(LONG_DISTANCE_NAMES)}

# --- caches ---
_GTFS_CACHE = {"fetched_at": None, "zip_bytes": None}
_STATES_CACHE = {"loaded": False, "geojson": None, "source": None}


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


def _load_states_geojson():
    """
    Load basemap from local file first (reliable on Render).
    Fallback to remote URLs only if local file isn't present.
    Cached in-memory after first load.
    """
    if _STATES_CACHE["loaded"] and _STATES_CACHE["geojson"] is not None:
        return _STATES_CACHE["geojson"], _STATES_CACHE["source"]

    # 1) local file
    try:
        if os.path.exists(LOCAL_STATES_GEOJSON_PATH):
            with open(LOCAL_STATES_GEOJSON_PATH, "r", encoding="utf-8") as f:
                gj = json.load(f)
            _STATES_CACHE.update({"loaded": True, "geojson": gj, "source": "local"})
            return gj, "local"
    except Exception as e:
        print(f"[WARN] Failed reading local basemap {LOCAL_STATES_GEOJSON_PATH}: {e}")

    # 2) remote fallback
    last_err = None
    for url in US_STATES_GEOJSON_URLS:
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "amtrak-map-service/1.0"})
            r.raise_for_status()
            gj = r.json()
            _STATES_CACHE.update({"loaded": True, "geojson": gj, "source": url})
            return gj, url
        except Exception as e:
            last_err = e

    print(f"[WARN] Failed to load basemap from all sources: {last_err}")
    _STATES_CACHE.update({"loaded": True, "geojson": {"type": "FeatureCollection", "features": []}, "source": None})
    return _STATES_CACHE["geojson"], None


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
    L = math.hypot(dx, dy)
    if L == 0:
        return 0.0, 0.0
    return (-dy / L, dx / L)


def _offset_polyline(lons, lats, gap_deg, sign):
    """
    Smooth vertex-based offset:
    tangent at each vertex is computed using neighbours; perpendicular gives offset direction.
    """
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
        outx[i] = lons[i] + ux * gap_deg * sign
        outy[i] = lats[i] + uy * gap_deg * sign

    return outx, outy


def _draw_states_basemap(ax):
    gj, source = _load_states_geojson()
    feats = gj.get("features", [])

    # More visible (you c
