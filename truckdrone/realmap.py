"""
Interactive Folium maps of a delivery run on the real city (Review-1
commitment: "Folium maps").

The dashboard draws in an abstract 1000-pixel frame so it can animate fast.
These maps do the opposite: a static, zoomable, real-world view on
OpenStreetMap tiles, where you can click a customer and see who delivered it
and when, follow the truck down the actual streets it drove, and compare
against the truck-only route on the same map.

Everything is drawn from what the simulator really did: truck legs follow the
stored street polyline between consecutive stops (the same geometry the
simulator's distances come from), drone legs are straight lines because
drones fly straight.

    python -m truckdrone.realmap --policy greedy --instance 0
"""

import argparse
import os

from . import config
from .rollout import make_rollout
from .scenario import CITIES, StreetPaths, load_scenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_DIR = os.path.join(ROOT, "results", "maps")

TRUCK, DRONE, BASE, DEPOT = "#2f6fed", "#db6d28", "#8b949e", "#8957e5"


def _hhmm(start_hour, t_s):
    h = start_hour + t_s / 3600.0
    return "{:02d}:{:02d}".format(int(h) % 24, int((h % 1) * 60))


def build_map(scenario, trace, out_path):
    import folium

    city = scenario["city"]
    paths = StreetPaths(city)
    ll = scenario["lonlat"]                     # (lon, lat) per site
    lat = lambda n: [float(ll[n][1]), float(ll[n][0])]     # noqa: E731
    start = trace["world"]["start_hour"]
    hyb, base = trace["hybrid"], trace["baseline"]

    # Esri basemaps: they need no API key (CARTO's now do) and, unlike the
    # openstreetmap.org tile server, they serve a map opened straight from
    # disk, which sends no Referer.
    m = folium.Map(location=[CITIES[city]["lat"], CITIES[city]["lon"]],
                   zoom_start=13, tiles=None, control_scale=True)
    esri = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
    folium.TileLayer(esri.format("Canvas/World_Light_Gray_Base"), name="Light",
                     attr="Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; "
                          "OpenStreetMap contributors", max_zoom=16).add_to(m)
    folium.TileLayer(esri.format("World_Street_Map"), name="Streets",
                     attr="Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; "
                          "OpenStreetMap contributors", show=False).add_to(m)

    def street(a, b):
        pts = paths.lonlat(a, b)
        return [[float(p[1]), float(p[0])] for p in pts] if pts is not None else []

    base_layer = folium.FeatureGroup(name="Truck-only baseline route", show=False)
    for leg in base["legs"]:
        folium.PolyLine(street(leg["from"], leg["to"]), color=BASE, weight=4,
                        opacity=0.7).add_to(base_layer)
    base_layer.add_to(m)

    truck_layer = folium.FeatureGroup(name="Truck route ({})".format(trace["policy"]))
    for k, leg in enumerate(hyb["legs"], 1):
        folium.PolyLine(street(leg["from"], leg["to"]), color=TRUCK, weight=5,
                        opacity=0.85, tooltip="Leg {}: {:.1f} km, departs {}".format(
                            k, leg["km"], _hhmm(start, leg["t_start"]))).add_to(truck_layer)
    truck_layer.add_to(m)

    drone_layer = folium.FeatureGroup(name="Drone sorties")
    drone_served = {}
    for s in hyb["sorties"]:
        drone_served[s["customer"]] = s
        folium.PolyLine([lat(s["launch"]), lat(s["customer"])], color=DRONE, weight=3,
                        dash_array="8 6", tooltip="Drone {} out with 1 parcel".format(
                            s["drone"] + 1)).add_to(drone_layer)
        folium.PolyLine([lat(s["customer"]), lat(s["recovery"])], color=TRUCK, weight=2,
                        dash_array="2 6", opacity=0.8,
                        tooltip="Drone {} flies on to meet the truck".format(
                            s["drone"] + 1)).add_to(drone_layer)
    drone_layer.add_to(m)

    truck_served = {leg["truck_delivery"]: leg for leg in hyb["legs"]
                    if leg["truck_delivery"] is not None}
    for c in trace["customers"]:
        if c in drone_served:
            s = drone_served[c]
            colour, text = DRONE, ("<b>Customer {}</b><br>Delivered by drone {}<br>"
                                   "launched {}, {} km sortie<br>saved {} km of truck "
                                   "detour".format(c, s["drone"] + 1,
                                                   _hhmm(start, s["t_launch"]),
                                                   s["km"], s["detour_saved_km"]))
        else:
            leg = truck_served.get(c)
            colour = TRUCK
            text = "<b>Customer {}</b><br>Delivered by truck{}".format(
                c, " at " + _hhmm(start, leg["t_arrive"]) if leg else "")
        folium.CircleMarker(lat(c), radius=8, color="white", weight=2, fill=True,
                            fill_color=colour, fill_opacity=0.95,
                            popup=folium.Popup(text, max_width=260)).add_to(m)
    folium.Marker(lat(trace["depot"]), tooltip="Depot",
                  icon=folium.Icon(color="purple", icon="home")).add_to(m)

    h, b, c = hyb["stats"], trace["baseline_stats"], trace["comparison"]
    legend = """
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;
         padding:12px 14px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.2);
         font:13px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;max-width:330px">
      <b style="font-size:14px">{name} &middot; instance #{inst}</b><br>
      <span style="color:#555">{policy} &middot; start {start} &middot; wind {wind:.1f} m/s</span>
      <hr style="margin:7px 0">
      <span style="color:{truck}">&#9644;</span> truck on real streets &nbsp;
      <span style="color:{drone}">&#9476;</span> drone out &nbsp;
      <span style="color:{truck}">&#8943;</span> drone to rendezvous<br>
      <span style="color:{drone}">&#9679;</span> drone-delivered &nbsp;
      <span style="color:{truck}">&#9679;</span> truck-delivered
      <hr style="margin:7px 0">
      <b>{time} min</b> vs {btime} min truck-only ({dt:+.1f}%)<br>
      {km} km driven vs {bkm} km ({dk:+.1f}%)<br>
      &#8377;{cost:.0f} vs &#8377;{bcost:.0f} ({dc:+.1f}%) &middot; {drones} of {n} by drone
    </div>""".format(
        name=CITIES[city]["name"], inst=trace["instance"] + 1, policy=trace["policy"],
        start=_hhmm(start, 0), wind=trace["world"]["wind_ms"], truck=TRUCK, drone=DRONE,
        time=h["time_min"], btime=b["time_min"], dt=c["time_change_pct"],
        km=h["truck_km"], bkm=b["truck_km"], dk=c["truck_km_change_pct"],
        cost=h["cost_inr"], bcost=b["cost_inr"], dc=c["cost_change_pct"],
        drones=h["drone_deliveries"], n=h["customers"])
    m.get_root().html.add_child(folium.Element(legend))
    stops = [lat(n) for n in [trace["depot"]] + list(trace["customers"])]
    m.fit_bounds([[min(p[0] for p in stops), min(p[1] for p in stops)],
                  [max(p[0] for p in stops), max(p[1] for p in stops)]], padding=(30, 30))
    folium.LayerControl(collapsed=False).add_to(m)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    m.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="greedy")
    ap.add_argument("--city", default=config.CITY)
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--world", type=int, default=config.EVAL_WORLD_SEEDS[0])
    args = ap.parse_args()
    sc = load_scenario(args.city)
    trace = make_rollout(sc, args.policy, args.instance, args.world)
    out = build_map(sc, trace, os.path.join(MAP_DIR, "{}-{}.html".format(args.city, args.instance)))
    print(out, "({:.0f} KB)".format(os.path.getsize(out) / 1024))


if __name__ == "__main__":
    main()
