import argparse
import io
import zipfile
from datetime import datetime, timedelta
import math

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import requests
import pytz
from astral import LocationInfo
from astral.sun import sun

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
    ("Denver", -104.9903, 39.7392),
    ("Chicago", -87.6298, 41.8781),
    ("New York", -74.0060, 40.7128),
]

# ---- Route styling ----
DAY_LINEWIDTH = 2.6
NIGHT_LINEWIDTH = 1.1
NIGHT_ALPHA = 0.55

# ---- Direction arrows (subtle) ----
ARROW_EVERY_N_SEGMENTS = 14
ARROW_HEAD_LENGTH = 3.5
ARROW_HEAD_WIDTH = 2.4
ARROW_LW_DAY = 0.45
ARROW_LW_NIGHT = 0.35
ARROW_ALPHA_DAY = 0.5
ARROW_ALPHA_NIGHT = 0.3

# ---- Labels ----
ROUTE_LABEL_FONTSIZE = 6.5
LABEL_OFFSET_DEGREES = 0.35
LABEL_HALO_WIDTH = 2.2

# ---- Fixed, print-safe colours (no colormap dependency) ----
HEX_PALETTE = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
    "#66a61e", "#e6ab02", "#a6761d", "#666666",
    "#1f78b4", "#b2df8a", "#fb9a99", "#fdbf6f",
    "#cab2d6", "#6a3d9a",
]
ROUTE_COLOURS = {name: HEX_PALETTE[i % len(HEX_PALETTE)] for i, name in enumerate(LONG_DISTANCE_NAMES)}

def parse_gtfs_time(t):
    if pd.isna(t) or not isinstance(t, str) or not t.strip():
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
    loc = LocationInfo(latitude=lat, longitude=lon, timezone=getattr(tz, "zone", "UTC"))
    if dt_local.tzinfo is None:
        dt_local = tz.localize(dt_local)
    s = sun(loc.observer, date=dt_local.date(), tzinfo=dt_local.tzinfo)
    return s["sunrise"] <= dt_local <= s["sunset"]

def download_gtfs():
    r = requests.get(GTFS_URL, timeout=60)
    r.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(r.content))

def read_txt(z, name):
    with z.open(name) as f:
        return pd.read_csv(f)

def draw_background(ax):
    for name, lon, lat in KEY_CITIES:
        ax.scatter(lon, lat, s=10, color="0.65", zorder=2)
        ax.text(lon + 0.25, lat + 0.15, name, fontsize=7, color="0.45", zorder=2)

def draw_direction_arrow(ax, x0, y0, x1, y1, colour, is_day):
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

def perpendicular_offset(x0, y0, x1, y1, distance):
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0, 0.0
    ux = -dy / length
    uy = dx / length
    return ux * distance, uy * distance

def main(run_date, out_pdf):
    z = download_gtfs()

    routes = read_txt(z, "routes.txt")
    trips = read_txt(z, "trips.txt")
    stop_times = read_txt(z, "stop_times.txt")
    stops = read_txt(z, "stops.txt")
    calendar = read_txt(z, "calendar.txt")

    yyyymmdd = run_date.strftime("%Y%m%d")

    active_services = {
        row["service_id"]
        for _, row in calendar.iterrows()
        if service_active_on(row, yyyymmdd)
    }

    trips = trips[trips["service_id"].isin(active_services)]
    routes = routes[routes["route_long_name"].isin(LONG_DISTANCE_NAMES)]
    trips = trips.merge(routes[["route_id","route_long_name"]], on="route_id")

    stop_times = stop_times[stop_times["trip_id"].isin(trips["trip_id"])].copy()
    stop_times["arr_sec"] = stop_times["arrival_time"].apply(parse_gtfs_time)
    stop_times["dep_sec"] = stop_times["departure_time"].apply(parse_gtfs_time)
    stop_times = stop_times.sort_values(["trip_id","stop_sequence"])

    stops = stops[["stop_id","stop_lat","stop_lon","stop_timezone"]].copy()
    stops["stop_timezone"] = stops["stop_timezone"].fillna("UTC")

    # Representative trip per (route, direction): choose trip with most stops
    reps = []
    for (name, direction), grp in trips.groupby(["route_long_name","direction_id"]):
        counts = stop_times[stop_times["trip_id"].isin(grp["trip_id"])].groupby("trip_id").size()
        if len(counts):
            reps.append((name, int(direction), counts.idxmax()))

    if not reps:
        raise RuntimeError("No matching long-distance trips found for that date in the GTFS feed.")

    plt.rcParams["font.family"] = "DejaVu Sans"

    # Legend handles (colour → service)
    legend_handles = [
        mlines.Line2D([], [], color=ROUTE_COLOURS[name], lw=3, label=name)
        for name in LONG_DISTANCE_NAMES
    ]

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out_pdf) as pdf:
        for direction in [0, 1]:
            fig, ax = plt.subplots(figsize=(16, 9))

            ax.set_title(
                f"Amtrak Long-Distance Routes — {DIRECTION_LABELS[direction]}\n"
                f"Daylight vs Darkness — {run_date}",
                fontsize=14
            )

            ax.set_xlim(-125, -66)
            ax.set_ylim(24, 50)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

            draw_background(ax)

            # Day/night legend (small)
            day_leg = mlines.Line2D([], [], color="black", lw=3, label="Daylight (solid)")
            night_leg = mlines.Line2D([], [], color="black", lw=1.2, linestyle="--", label="Darkness (dashed)")

            ax.legend(handles=[day_leg, night_leg], loc="lower left", fontsize=8, frameon=False)

            # Big route key OUTSIDE plot so it doesn't break layout
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

                st = stop_times[stop_times["trip_id"] == trip_id].merge(stops, on="stop_id").dropna()
                if len(st) < 2:
                    continue

                for i in range(len(st) - 1):
                    a = st.iloc[i]
                    b = st.iloc[i + 1]

                    mid_sec = int((a["dep_sec"] + b["arr_sec"]) / 2)
                    dt_naive = datetime(run_date.year, run_date.month, run_date.day) + timedelta(seconds=mid_sec)

                    tz_name = a["stop_timezone"]
                    try:
                        dt_local = pytz.timezone(tz_name).localize(dt_naive)
                    except Exception:
                        dt_local = pytz.UTC.localize(dt_naive)

                    daylight = is_daylight(
                        (a["stop_lat"] + b["stop_lat"]) / 2,
                        (a["stop_lon"] + b["stop_lon"]) / 2,
                        tz_name,
                        dt_local,
                    )

                    x0, y0 = a["stop_lon"], a["stop_lat"]
                    x1, y1 = b["stop_lon"], b["stop_lat"]

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
                        draw_direction_arrow(ax, xm0, ym0, xm1, ym1, colour, daylight)

                mid = len(st) // 2
                a, b = st.iloc[mid - 1], st.iloc[mid]
                dx, dy = perpendicular_offset(
                    a["stop_lon"], a["stop_lat"],
                    b["stop_lon"], b["stop_lat"],
                    LABEL_OFFSET_DEGREES
                )

                ax.text(
                    (a["stop_lon"] + b["stop_lon"]) / 2 + dx,
                    (a["stop_lat"] + b["stop_lat"]) / 2 + dy,
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

            # IMPORTANT: make room on the right for the external legend
            fig.tight_layout(rect=[0, 0, 0.80, 1])

            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved: {out_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--out", default="amtrak_long_distance_daylight.pdf")
    args = parser.parse_args()

    run_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    main(run_date, args.out)
