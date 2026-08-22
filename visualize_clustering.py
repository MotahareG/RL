#!/usr/bin/env python3
# visualize_clustering.py
# ==============================================================================
# California AV Collision Risk Map — Presentation-Ready Visualisation
#
# Produces TWO outputs:
#
#   1. california_risk_map.html  (interactive, OpenStreetMap background)
#      Open in any browser for a zoomable, pannable presentation map.
#      Layers (toggle via ⊞ control):
#        • OpenStreetMap basemap tiles (+ CartoDB light option)
#        • KDE collision-density heatmap
#        • Individual collision markers (clustered, click for info)
#        • K-means risk-zone boundary circles
#        • Candidate station markers with popup info
#
#   2. california_risk_map_static.png  (clean static for slides/LaTeX)
#
# Usage:
#   python visualize_clustering.py                     # default n_fac=5
#   python visualize_clustering.py --n_fac 8
# ==============================================================================

import argparse
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity

# ── Constants ─────────────────────────────────────────────────────────────────
CA_BBOX   = ((32.5, -124.5), (42.0, -114.0))
LAT_MIN, LON_MIN = CA_BBOX[0]
LAT_MAX, LON_MAX = CA_BBOX[1]
CA_CENTER = (37.5, -119.5)

CITIES = {
    'San Francisco': (37.77,  -122.42, 'High AV activity'),
    'Mountain View': (37.39,  -122.08, 'Waymo HQ'),
    'San Jose':      (37.34,  -121.89, 'AV testing hub'),
    'Los Angeles':   (34.05,  -118.24, 'AV deployment'),
    'San Diego':     (32.72,  -117.16, 'AV testing'),
    'Sacramento':    (38.58,  -121.49, 'State capital'),
}

CLUSTER_COLORS = [
    '#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
    '#a65628','#f781bf','#999999','#66c2a5','#fc8d62',
]


def normalize_array(pts_geo):
    lat = (pts_geo[:, 0] - LAT_MIN) / (LAT_MAX - LAT_MIN)
    lon = (pts_geo[:, 1] - LON_MIN) / (LON_MAX - LON_MIN)
    return np.stack([lat, lon], axis=1)

def denormalize_array(pts_norm):
    lat = pts_norm[:, 0] * (LAT_MAX - LAT_MIN) + LAT_MIN
    lon = pts_norm[:, 1] * (LON_MAX - LON_MIN) + LON_MIN
    return np.stack([lat, lon], axis=1)


def load_collision_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[(df['Latitude']  >= LAT_MIN) & (df['Latitude']  <= LAT_MAX)]
    df = df[(df['Longitude'] >= LON_MIN) & (df['Longitude'] <= LON_MAX)]
    df = df.dropna(subset=['Latitude', 'Longitude'])
    pts_geo  = df[['Latitude', 'Longitude']].values
    pts_norm = normalize_array(pts_geo)
    return pts_geo, pts_norm, df


def compute_candidates(pts_norm, n_fac):
    km = KMeans(n_clusters=n_fac, n_init=10, random_state=42)
    km.fit(pts_norm)
    centres_norm = np.clip(km.cluster_centers_, 0.0, 1.0)
    centres_geo  = denormalize_array(centres_norm)
    return centres_geo, centres_norm, km.labels_


# ==============================================================================
# 1. INTERACTIVE FOLIUM / OSM MAP
# ==============================================================================

def make_folium_map(pts_geo, pts_norm, df, centres_geo, centres_norm,
                    labels, n_fac, save_path):

    cluster_sizes = np.bincount(labels, minlength=n_fac)

    # ── Base map ──────────────────────────────────────────────────────────────
    m = folium.Map(location=CA_CENTER, zoom_start=6,
                   tiles='OpenStreetMap', control_scale=True)
    folium.TileLayer('CartoDB positron', name='CartoDB (light)').add_to(m)

    # ── Heatmap layer ─────────────────────────────────────────────────────────
    HeatMap(
        [[r[0], r[1]] for r in pts_geo],
        name='Collision density heatmap',
        min_opacity=0.35,
        radius=20, blur=28,
        gradient={0.0:'blue', 0.35:'cyan', 0.65:'yellow',
                  0.85:'orange', 1.0:'red'},
    ).add_to(m)

    # ── Individual markers (clustered) ────────────────────────────────────────
    mc = MarkerCluster(name='Individual collision reports', show=False)
    for i, (lat, lon) in enumerate(pts_geo):
        row = df.iloc[i]
        html = (f"<b>AV Collision Report</b><br>"
                f"Date: {row.get('Date Of Accident','N/A')}<br>"
                f"City: {row.get('City','N/A')}, {row.get('County','N/A')}<br>"
                f"Risk zone: {labels[i]+1}")
        folium.CircleMarker(
            location=[lat, lon], radius=5,
            color=CLUSTER_COLORS[labels[i] % len(CLUSTER_COLORS)],
            fill=True, fill_opacity=0.75,
            popup=folium.Popup(html, max_width=240),
            tooltip=f"Collision in zone {labels[i]+1}"
        ).add_to(mc)
    mc.add_to(m)

    # ── Risk zone circles ─────────────────────────────────────────────────────
    zone_group = folium.FeatureGroup(name='High-risk zone boundaries', show=True)
    for i in range(n_fac):
        clat, clon = centres_geo[i]
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        members = pts_geo[labels == i]
        if len(members) > 1:
            dists = np.sqrt(((members[:, 0]-clat)*111000)**2 +
                            ((members[:, 1]-clon)*85000)**2)
            radius_m = max(float(np.percentile(dists, 75)), 5000)
        else:
            radius_m = 8000
        folium.Circle(
            location=[clat, clon], radius=radius_m,
            color=color, fill=True, fill_opacity=0.12, weight=2.5,
            popup=folium.Popup(
                f"<b>Risk Zone {i+1}</b><br>"
                f"Collisions: {cluster_sizes[i]}<br>"
                f"Centre: {clat:.4f}°N, {abs(clon):.4f}°W",
                max_width=220),
            tooltip=f"Risk Zone {i+1}  ({cluster_sizes[i]} collisions)"
        ).add_to(zone_group)
    zone_group.add_to(m)

    # ── Candidate station markers ─────────────────────────────────────────────
    stn_group = folium.FeatureGroup(
        name='Candidate stations (K-means centres)', show=True)
    for i in range(n_fac):
        clat, clon = centres_geo[i]
        lat_n, lon_n = centres_norm[i]
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        popup_html = (
            f"<b>Candidate Station S{i+1}</b><hr style='margin:4px 0'>"
            f"Location: {clat:.4f}°N, {abs(clon):.4f}°W<br>"
            f"Model coords: ({lat_n:.4f}, {lon_n:.4f})<br>"
            f"Accidents in zone: {cluster_sizes[i]}<br><br>"
            f"<i>Agent decides: BUILD or SKIP this station?<br>"
            f"Opening cost = C_j; routing safety benefit = avoid λ=0 edges.</i>"
        )
        folium.Marker(
            location=[clat, clon],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"★ Candidate Station S{i+1}  |  {cluster_sizes[i]} collisions nearby",
            icon=folium.DivIcon(
                html=(f'<div style="background:{color};border:2px solid #000;'
                      f'border-radius:50%;width:30px;height:30px;'
                      f'display:flex;align-items:center;justify-content:center;'
                      f'font-weight:bold;font-size:12px;color:#fff;'
                      f'box-shadow:2px 2px 5px rgba(0,0,0,0.5);">'
                      f'S{i+1}</div>'),
                icon_size=(30, 30), icon_anchor=(15, 15))
        ).add_to(stn_group)
    stn_group.add_to(m)

    # ── City markers ──────────────────────────────────────────────────────────
    city_group = folium.FeatureGroup(name='Major cities', show=True)
    for city, (lat, lon, note) in CITIES.items():
        folium.Marker(
            location=[lat, lon],
            tooltip=f"{city} — {note}",
            popup=folium.Popup(f"<b>{city}</b><br>{note}", max_width=180),
            icon=folium.Icon(color='gray', icon='building', prefix='fa')
        ).add_to(city_group)
    city_group.add_to(m)

    # ── Title ─────────────────────────────────────────────────────────────────
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
         z-index:1000;background:rgba(255,255,255,0.95);padding:10px 22px;
         border-radius:8px;font-family:Arial,sans-serif;text-align:center;
         border:1px solid #aaa;box-shadow:2px 2px 6px rgba(0,0,0,0.25);">
      <b style="font-size:15px">California AV Collision Risk Map (2019–2024)</b><br>
      <span style="font-size:11px;color:#444">
        {len(pts_geo)} collision reports &nbsp;|&nbsp;
        {n_fac} candidate station locations (K-means)
        &nbsp;|&nbsp; Agent decides how many to build
      </span>
    </div>"""))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = "".join([
        f'<div style="margin-top:4px">'
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'background:{CLUSTER_COLORS[i%len(CLUSTER_COLORS)]};border-radius:50%;'
        f'border:1px solid #000;vertical-align:middle"></span> '
        f'S{i+1}: {cluster_sizes[i]} accidents</div>'
        for i in range(n_fac)])

    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
         background:#fff;padding:12px 16px;border:2px solid #555;
         border-radius:8px;font-family:Arial,sans-serif;font-size:12px;
         box-shadow:3px 3px 8px rgba(0,0,0,0.3);max-width:260px;">
      <b style="font-size:13px">Legend</b><hr style="margin:5px 0">
      <div>🔴 Red heatmap = high collision density</div>
      <div>🔵 Blue heatmap = low collision density</div>
      <div style="margin-top:6px"><b>Candidate Stations (K-means centres):</b></div>
      {legend_items}
      <hr style="margin:6px 0">
      <span style="font-size:10px;color:#666">
        Use ⊞ (top-right) to toggle layers.<br>
        Click any marker for details.
      </span>
    </div>"""))

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(save_path)
    print(f"✅  Interactive OSM map → {save_path}")
    return m


# ==============================================================================
# 2. STATIC FIGURE
# ==============================================================================

def make_static_figure(pts_geo, centres_geo, centres_norm,
                       labels, n_fac, save_path):
    cluster_sizes = np.bincount(labels, minlength=n_fac)
    colours = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(n_fac)]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 8),
                                  gridspec_kw={'width_ratios': [2.2, 1]})

    for i in range(n_fac):
        mask = labels == i
        ax.scatter(pts_geo[mask, 1], pts_geo[mask, 0],
                   c=colours[i], s=22, alpha=0.55,
                   label=f'Zone {i+1} ({cluster_sizes[i]} accidents)',
                   edgecolors='none', zorder=3)

    for i in range(n_fac):
        lat, lon = centres_geo[i]
        ax.scatter(lon, lat, s=440, c=colours[i], marker='*',
                   edgecolors='black', linewidths=1.2, zorder=8)
        ax.annotate(f'S{i+1}  ({cluster_sizes[i]})',
                    (lon, lat), xytext=(8, 6), textcoords='offset points',
                    fontsize=8.5, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.25', fc=colours[i],
                              alpha=0.85, ec='black', lw=0.8))

    for city, (lat, lon, _) in CITIES.items():
        ax.plot(lon, lat, '^', color='#2c3e50', ms=7, zorder=6)
        ax.annotate(city, (lon, lat), xytext=(4, -13),
                    textcoords='offset points', fontsize=7.5, color='#2c3e50')

    ax.plot([LON_MIN,LON_MAX,LON_MAX,LON_MIN,LON_MIN],
            [LAT_MIN,LAT_MIN,LAT_MAX,LAT_MAX,LAT_MIN],
            'k--', lw=1, alpha=0.4)
    ax.set_xlim(LON_MIN-0.5, LON_MAX+0.5)
    ax.set_ylim(LAT_MIN-0.3, LAT_MAX+0.3)
    ax.set_xlabel('Longitude (°W)', fontsize=11)
    ax.set_ylabel('Latitude (°N)',  fontsize=11)
    ax.set_title(
        f'California AV Collision Clusters + Candidate Station Locations\n'
        f'★ = K-means centre (candidate station)   '
        f'dots = historical collision reports\n'
        f'Agent decides which stations to BUILD (0 … {n_fac}) '
        f'based on routing cost vs safety',
        fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=8, title='Risk Zones', framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle=':')
    ax.text(0.01, 0.01,
            '⚠ Note: Open california_risk_map.html for the full\n'
            'interactive version with OpenStreetMap background.',
            transform=ax.transAxes, fontsize=8, color='#666',
            va='bottom',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    bars = ax2.barh(np.arange(n_fac), cluster_sizes, color=colours,
                    edgecolor='black', linewidth=0.7, height=0.6)
    for bar, cnt in zip(bars, cluster_sizes):
        ax2.text(bar.get_width() + cluster_sizes.max()*0.01,
                 bar.get_y()+bar.get_height()/2,
                 str(cnt), va='center', fontsize=9)
    ax2.set_yticks(np.arange(n_fac))
    ax2.set_yticklabels([f'S{i+1}' for i in range(n_fac)], fontsize=10)
    ax2.set_xlabel('Accidents in zone', fontsize=10)
    ax2.set_title('Accidents per\nCandidate Station\n(justifies station need)',
                  fontsize=10, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3, linestyle=':')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.text(0.5, 0.005,
             f'Total: {len(pts_geo)} collisions  |  {n_fac} candidate stations  |  '
             f'CA DMV AV Collision Reports 2019–2024',
             ha='center', fontsize=8.5,
             bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.7))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅  Static figure → {save_path}")


# ==============================================================================
# Entry point
# ==============================================================================

def visualize_risk_and_candidates(
        collision_csv_path='CA_AV_Collision_2019-2024.csv',
        n_fac=None,
        html_save_path='california_risk_map.html',
        static_save_path='california_risk_map_static.png'):
    """
    Generate the interactive OSM map and the static clustering figure.

    Parameters
    ----------
    n_fac : int
        Number of K-means clusters = number of candidate station locations.
        **Must be passed explicitly** so it always matches the task's n_fac
        (e.g. 5 for lrp10, 8 for lrp20, 12 for lrp50, 20 for lrp100).
        When called from the CLI the --n_fac argument is used (default 5).
        When called programmatically from visualise_after_training() the
        task-specific value from args['n_fac'] is forwarded automatically.
        (Improvement 2: was hardcoded to 5, now always uses the correct value.)
    """
    if n_fac is None:
        raise ValueError(
            "n_fac must be specified.  Pass args['n_fac'] from the task config "
            "(e.g. n_fac=8 for lrp20) so the K-means cluster count matches the "
            "number of candidate stations in the current problem.")

    print("="*70)
    print("CALIFORNIA AV COLLISION RISK MAP  (OpenStreetMap + K-means)")
    print("="*70)

    print(f"\n[1/3] Loading collision data ...")
    pts_geo, pts_norm, df = load_collision_data(collision_csv_path)
    print(f"  {len(pts_geo)} records.")

    print(f"[2/3] K-means ({n_fac} clusters) ...")
    centres_geo, centres_norm, labels = compute_candidates(pts_norm, n_fac)
    sizes = np.bincount(labels, minlength=n_fac)

    print(f"\n{'='*70}")
    print(f"CANDIDATE STATION LOCATIONS  (n_fac = {n_fac})")
    print(f"{'='*70}")
    print(f"{'ID':<5}{'Latitude':>10}{'Longitude':>12}{'NLat':>8}{'NLon':>8}{'Accidents':>11}")
    print(f"{'-'*70}")
    for i in range(n_fac):
        print(f"S{i+1:<4}{centres_geo[i,0]:>10.4f}{centres_geo[i,1]:>12.4f}"
              f"{centres_norm[i,0]:>8.4f}{centres_norm[i,1]:>8.4f}{sizes[i]:>11d}")
    print(f"{'='*70}\n")

    print(f"[3/3] Generating outputs ...")
    make_folium_map(pts_geo, pts_norm, df, centres_geo, centres_norm,
                    labels, n_fac, html_save_path)
    make_static_figure(pts_geo, centres_geo, centres_norm,
                       labels, n_fac, static_save_path)
    return centres_geo, centres_norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="California AV Collision Risk Map with K-means candidate stations")
    parser.add_argument('--csv',    default='CA_AV_Collision_2019-2024.csv')
    parser.add_argument('--n_fac',  type=int, default=5,
                        help='Number of K-means clusters = candidate stations. '
                             'Match this to the task: lrp10→5, lrp20→8, '
                             'lrp50→12, lrp100→20  (Improvement 2)')
    parser.add_argument('--html',   default='california_risk_map.html')
    parser.add_argument('--static', default='california_risk_map_static.png')
    args = parser.parse_args()
    visualize_risk_and_candidates(
        args.csv, args.n_fac, args.html, args.static)
