import os
import io
import zipfile
import math
from datetime import datetime, timedelta, date
from collections import defaultdict

# --- Render-safe matplotlib setup ---
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

# ---------------- CONFIG ----------------

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

DAY_LINEWIDTH = 2.6
NIGHT_LINEWIDTH = 1.1
NIGHT_ALPHA = 0.55
PARALLEL_GAP_DEG = 0.17

LABEL_FONTSIZE = 7.0
LABEL_HALO_WIDTH = 2.6

CHEVRON_EVERY_N_SEGMENTS = 10
CHEVRON_SKIP_END_SEGMENTS = 3
CHEVRON_SIZE_DEG = 0.36
CHEVRON_ANGLE_DEG = 24
CHEVRON_LW_NIGHT = 1.05
CHEVRON_ALPHA_NIGHT = 0.90

LIGHT_SAMPLE_POINTS_PER_SEGMENT = 5
LIGHT_MAJORITY_THRESHOLD = 0.5

HEX_PALETTE = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
    "#66a61e", "#e6ab02", "#a6761d", "#666666",
    "#1f78b4", "#b2df8a", "#fb9a99", "#fdbf6f",
    "#cab2d6", "#6a3d9a",
]
ROUTE_COLOURS = {name: HEX_PALETTE[i % len(HEX_PALETTE)] for i, name in enumerate(LONG_DISTANCE_NAMES)}

_GTFS_CACHE = {"fetched_at": None, "zip_bytes": None}

# --- Background states PNG ---
BACKGROUND_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "assets", "us_states.png")
BACKGROUND_ALPHA = 0.20

def _draw_background_states(ax):
    if not os.path.exists(BACKGROUND_IMAGE_PATH):
        print("Background image not found:", BACKGROUND_IMAGE_PATH)
        return
    img = Image.open(BACKGROUND_IMAGE_PATH).convert("RGBA")
    ax.imshow(
        img,
        extent=(-125, -66, 24, 50),
        origin="upper",
        alpha=BACKGROUND_ALPHA,
        zorder=0,
        aspect="auto",
    )

# --- Station trigraphs ---
STATION_MARKERS = [
    ("SEA",-122.33,47.60),("PDX",-122.68,45.52),("SPK",-117.43,47.66),
    ("MSP",-93.27,44.98),("FAR",-96.79,46.88),("GPK",-113.99,48.42),
    ("SAC",-121.49,38.58),("RNO",-119.81,39.53),("DEN",-104.99,39.74),
    ("GJT",-108.55,39.06),("GLN",-107.32,39.55),("CHI",-87.63,41.88),
    ("KCY",-94.58,39.10),("ABQ",-106.65,35.08),("FLG",-111.65,35.20),
    ("STL",-90.20,38.63),("LRK",-92.29,34.75),("DAL",-96.80,32.78),
    ("AUS",-97.74,30.27),("SAS",-98.49,29.42),("NOL",-90.07,29.95),
    ("ATL",-84.39,33.75),("WAS",-77.01,38.90),("NYP",-73.99,40.75),
    ("BOS",-71.06,42.36),("MIA",-80.19,25.76),
]

# --- Scenic POIs (nudged off tracks) ---
SCENIC_POIS = [
    ("GLACIER NP",-113.8,48.7,0.4,0.2),
    ("MARIAS PASS",-113.3,48.3,0.35,-0.2),
    ("COLUMBIA R.",-120.0,46.1,-0.35,0.2),
    ("ROCKY MTNS",-106.5,39.4,0.45,0.25),
    ("GLENWOOD\nCANYON",-107.2,39.6,-0.45,0.1),
    ("RATON PASS",-105.2,36.9,0.45,-0.25),
    ("MISSISSIPPI",-90.2,35.1,0.55,-0.1),
]

# --- Sunset Limited only NOL↔SAS ---
SUBSEGMENT_LIMITS = {
    "Sunset Limited": (("NOL",-90.07,29.95),("SAS",-98.49,29.42))
}

# ---------------- Helpers ----------------

def _draw_text(ax,text,lon,lat):
    ax.text(lon,lat,text,fontsize=LABEL_FONTSIZE,ha="center",va="center",
            color="black",path_effects=[pe.withStroke(linewidth=LABEL_HALO_WIDTH,foreground="white")])

def _draw_station_labels(ax):
    for code,lon,lat in STATION_MARKERS:
        _draw_text(ax,code,lon,lat)

def _draw_scenic(ax):
    for label,lon,lat,dx,dy in SCENIC_POIS:
        _draw_text(ax,label,lon+dx,lat+dy)

def _sun_civil_times(lat,lon,tz_name,on_date):
    try: tz=pytz.timezone(tz_name)
    except: tz=pytz.UTC
    loc=LocationInfo(latitude=lat,longitude=lon,timezone=tz.zone)
    s=sun(loc.observer,date=on_date,tzinfo=tz)
    return s["dawn"],s["dusk"]

def _classify_light(run_date,origin_dt,origin_sec,a,b,tz_name):
    try: tz=pytz.timezone(tz_name)
    except: tz=pytz.UTC
    t0=int(a["dep_sec"]); t1=int(b["arr_sec"])
    if t1<t0: t1+=24*3600
    lat0=float(a["stop_lat"]); lon0=float(a["stop_lon"])
    lat1=float(b["stop_lat"]); lon1=float(b["stop_lon"])
    votes=0
    n=LIGHT_SAMPLE_POINTS_PER_SEGMENT
    for k in range(n):
        f=k/(n-1)
        t=int(t0+(t1-t0)*f)
        dt=origin_dt+timedelta(seconds=(t-origin_sec))
        dt=dt.astimezone(tz)
        lat=lat0+(lat1-lat0)*f; lon=lon0+(lon1-lon0)*f
        dawn,dusk=_sun_civil_times(lat,lon,tz_name,dt.date())
        if dawn<=dt<=dusk: votes+=1
    return (votes/n)>LIGHT_MAJORITY_THRESHOLD

# ---------------- GTFS Load ----------------

def _download_gtfs():
    if _GTFS_CACHE["zip_bytes"] is None:
        r=requests.get(GTFS_URL,timeout=60)
        r.raise_for_status()
        _GTFS_CACHE["zip_bytes"]=r.content
    return _GTFS_CACHE["zip_bytes"]

def _parse_time(t):
    if not isinstance(t,str): return None
    h,m,s=t.split(":")
    return int(h)*3600+int(m)*60+int(s)

def _load(run_date):
    z=zipfile.ZipFile(io.BytesIO(_download_gtfs()))
    routes=pd.read_csv(z.open("routes.txt"))
    trips=pd.read_csv(z.open("trips.txt"))
    st=pd.read_csv(z.open("stop_times.txt"))
    stops=pd.read_csv(z.open("stops.txt"))
    cal=pd.read_csv(z.open("calendar.txt"))

    ymd=run_date.strftime("%Y%m%d")
    active=set()
    for _,r in cal.iterrows():
        d=datetime.strptime(ymd,"%Y%m%d").date()
        if not(datetime.strptime(str(r.start_date),"%Y%m%d").date()<=d<=datetime.strptime(str(r.end_date),"%Y%m%d").date()): continue
        if r[["monday","tuesday","wednesday","thursday","friday","saturday","sunday"][d.weekday()]]==1:
            active.add(r.service_id)

    trips=trips[trips.service_id.isin(active)]
    routes=routes[routes.route_long_name.isin(LONG_DISTANCE_NAMES)]
    trips=trips.merge(routes[["route_id","route_long_name"]],on="route_id")

    st=st[st.trip_id.isin(trips.trip_id)]
    st["dep_sec"]=st.departure_time.apply(_parse_time)
    st["arr_sec"]=st.arrival_time.apply(_parse_time)

    stops=stops[["stop_id","stop_lat","stop_lon","stop_timezone"]].fillna("UTC")
    return trips,st,stops

# ---------------- Map Build ----------------

def _make_map(run_date):
    trips,st,stops=_load(run_date)

    fig,ax=plt.subplots(figsize=(18,10))
    ax.set_xlim(-125,-66)
    ax.set_ylim(24,50)
    _draw_background_states(ax)

    ax.set_title(f"Amtrak Long-Distance Routes\nCivil Twilight (dawn → dusk) — {run_date}",fontsize=14)
    ax.grid(True,linewidth=0.3,alpha=0.2)

    style_handles=[
        mlines.Line2D([],[],color="black",lw=DAY_LINEWIDTH,label="Light (civil twilight)"),
        mlines.Line2D([],[],color="black",lw=NIGHT_LINEWIDTH,linestyle="--",alpha=NIGHT_ALPHA,label="Dark")
    ]
    ax.legend(handles=style_handles,loc="lower left",fontsize=9)

    ax.legend(handles=[mlines.Line2D([],[],color=ROUTE_COLOURS[n],lw=3,label=n) for n in LONG_DISTANCE_NAMES],
              loc="lower right",fontsize=8,title="Train services")

    _draw_station_labels(ax)
    _draw_scenic(ax)

    for name in LONG_DISTANCE_NAMES:
        rt=trips[trips.route_long_name==name]
        if rt.empty: continue
        trip_id=rt.iloc[0].trip_id

        geom=st[st.trip_id==trip_id].merge(stops,on="stop_id").sort_values("stop_sequence")
        if name in SUBSEGMENT_LIMITS:
            (c1,lon1,lat1),(c2,lon2,lat2)=SUBSEGMENT_LIMITS[name]
            d1=((geom.stop_lon-lon1)**2+(geom.stop_lat-lat1)**2).idxmin()
            d2=((geom.stop_lon-lon2)**2+(geom.stop_lat-lat2)**2).idxmin()
            geom=geom.loc[min(d1,d2):max(d1,d2)]

        if len(geom)<2: continue

        origin=geom.iloc[0]
        tz=pytz.timezone(origin.stop_timezone)
        origin_dt=tz.localize(datetime(run_date.year,run_date.month,run_date.day)+timedelta(seconds=int(origin.dep_sec or origin.arr_sec)))
        origin_sec=int(origin.dep_sec or origin.arr_sec)

        for i in range(len(geom)-1):
            a=geom.iloc[i]; b=geom.iloc[i+1]
            light=_classify_light(run_date,origin_dt,origin_sec,a,b,a.stop_timezone)
            x0,y0=float(a.stop_lon),float(a.stop_lat)
            x1,y1=float(b.stop_lon),float(b.stop_lat)
            ax.plot([x0,x1],[y0,y1],color=ROUTE_COLOURS[name],
                    lw=DAY_LINEWIDTH if light else NIGHT_LINEWIDTH,
                    linestyle="-" if light else "--",
                    alpha=1.0 if light else NIGHT_ALPHA)

    fig.tight_layout()
    return fig

# ---------------- Web Endpoints ----------------

def build_png_bytes(run_date):
    fig=_make_map(run_date)
    buf=io.BytesIO()
    fig.savefig(buf,format="png",dpi=170)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

@app.get("/")
def index():
    d=request.args.get("date","2026-02-06")
    return f'<img src="/map.png?date={d}" style="width:100%">'

@app.get("/map.png")
def map_png():
    d=request.args.get("date","")
    run_date=datetime.strptime(d,"%Y-%m-%d").date()
    return Response(build_png_bytes(run_date),mimetype="image/png")
