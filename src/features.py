"""
features.py — Shared utility functions for Vehicle Telematics project.

Used across all notebooks:
    from src.features import load_and_clean, estimate_gears_kmeans, get_excluded_devices
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


# ─── Constants ────────────────────────────────────────────────────────────────

NUMERIC_COLS = [
    'tripID', 'deviceID', 'gps_speed', 'battery', 'cTemp', 'dtc',
    'eLoad', 'iat', 'imap', 'kpl', 'maf', 'rpm', 'speed', 'tAdv', 'tPos'
]

SPEED_BINS   = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]
SPEED_LABELS = [
    '0-10', '10-20', '20-30', '30-40', '40-50', '50-60',
    '60-70', '70-80', '80-90', '90-100', '100-120', '120+'
]


# ─── Data Loading & Cleaning ──────────────────────────────────────────────────

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load raw allcars.csv and apply standard cleaning pipeline.

    Steps:
      1. Force numeric columns (handles repeated header rows)
      2. Parse timestamps
      3. Drop rows with missing core sensors
      4. Apply realistic sensor range filters
      5. Sort chronologically per vehicle-trip

    Returns cleaned DataFrame.
    """
    df = pd.read_csv(filepath, low_memory=False)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['timeStamp'] = pd.to_datetime(df['timeStamp'], errors='coerce')
    df = df.dropna(subset=['gps_speed', 'rpm', 'eLoad', 'kpl']).reset_index(drop=True)

    # Sensor range filters — remove glitches and unrealistic values
    df = df[df['speed'].between(0, 150)]
    df = df[df['eLoad'].between(0, 100)]
    df = df[df['kpl'].between(0, 25)]
    df = df[df['rpm'].between(0, 7000)]

    if 'cTemp' in df.columns:
        df['cTemp'] = df['cTemp'].where(df['cTemp'].between(-10, 130), np.nan)

    df = df.sort_values(['deviceID', 'tripID', 'timeStamp']).reset_index(drop=True)

    return df


def get_excluded_devices(df: pd.DataFrame) -> list:
    """
    Returns list of deviceIDs to exclude from fuel efficiency analyses:
      - Vehicles where KPL sensor sum == 0 (sensor not available)
      - Vehicle 7 (intermittent/unreliable KPL sensor)
    """
    kpl_by_device  = df.groupby('deviceID')['kpl'].sum()
    no_kpl         = kpl_by_device[kpl_by_device == 0].index.tolist()
    unreliable     = [7.0]
    return no_kpl + unreliable


# ─── Gear Estimation ──────────────────────────────────────────────────────────

def estimate_gears_kmeans(vehicle_df: pd.DataFrame, n_gears: int = 6) -> pd.DataFrame:
    """
    Estimates gear position using KMeans on RPM/speed ratio.

    OBD data does not include a gear sensor. Gear is inferred from:
        gear_ratio = RPM / speed
    Higher ratio = lower gear (high RPM, low speed = 1st gear).

    Clustering is done PER VEHICLE so each vehicle is calibrated
    to its own engine/transmission characteristics.

    Args:
        vehicle_df : DataFrame for a single vehicle (must contain 'gear_ratio')
        n_gears    : Number of gear clusters (default 6)

    Returns:
        vehicle_df with new column 'est_gear' (int 1=lowest, 6=highest)
    """
    ratios = vehicle_df['gear_ratio'].dropna().values.reshape(-1, 1)

    if len(ratios) < n_gears * 10:
        vehicle_df = vehicle_df.copy()
        vehicle_df['est_gear'] = 3  # fallback to mid gear
        return vehicle_df

    km = KMeans(n_clusters=n_gears, random_state=42, n_init=10)
    km.fit(ratios)

    # Highest centroid = gear 1 (high RPM/speed ratio = low gear)
    center_rank = {
        old: i + 1
        for i, old in enumerate(
            sorted(range(n_gears), key=lambda x: km.cluster_centers_[x], reverse=True)
        )
    }

    vehicle_df = vehicle_df.copy()
    vehicle_df['est_gear'] = [
        center_rank[c]
        for c in km.predict(vehicle_df['gear_ratio'].fillna(0).values.reshape(-1, 1))
    ]
    return vehicle_df


def add_gear_estimates(df: pd.DataFrame, excluded_devices: list) -> pd.DataFrame:
    """
    Apply estimate_gears_kmeans to all valid vehicles and return
    concatenated DataFrame with 'est_gear' column.
    """
    df = df.copy()
    df['gear_ratio'] = df['rpm'] / df['speed']

    parts = []
    for dev_id, vdf in df.groupby('deviceID'):
        if dev_id in excluded_devices:
            continue
        if vdf['kpl'].sum() == 0:
            continue
        parts.append(estimate_gears_kmeans(vdf))

    return pd.concat(parts).reset_index(drop=True)


# ─── Feature Engineering ──────────────────────────────────────────────────────

def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features used in the CatBoost fuel efficiency model.

    Features added:
        rpm_per_speed  — continuous gear ratio signal (RPM / speed+1)
        throttle_load  — driving aggressiveness (tPos x eLoad / 100)
        gear_ratio     — raw RPM/speed ratio (used for KMeans)
    """
    df = df.copy()
    df['gear_ratio']    = df['rpm'] / df['speed']
    df['rpm_per_speed'] = df['rpm'] / (df['speed'] + 1)
    df['throttle_load'] = df['tPos'] * df['eLoad'] / 100

    if 'cTemp' in df.columns:
        df['cTemp'] = df['cTemp'].fillna(df['cTemp'].median())

    return df


def add_speed_zone(df: pd.DataFrame, ordinal: bool = False) -> pd.DataFrame:
    """
    Add speed zone column.
    ordinal=False → string labels ('0-10', '10-20', ...)
    ordinal=True  → integer labels (0, 1, 2, ...) for model input
    """
    df = df.copy()
    if ordinal:
        df['speed_zone_ord'] = pd.cut(
            df['speed'], bins=SPEED_BINS, labels=list(range(12)), right=False
        ).astype(float)
    else:
        df['speed_zone'] = pd.cut(
            df['speed'], bins=SPEED_BINS, labels=SPEED_LABELS, right=False
        )
    return df
