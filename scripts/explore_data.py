"""
AirGuard AI — Quick Data Exploration Script
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.data_loader import load_city_day, load_pollution_sources, load_city_coordinates, load_traffic


def explore():
    print("=" * 65)
    print("  🌬️  AirGuard AI — Data Exploration Report")
    print("=" * 65)

    df = load_city_day()
    print(f"\n📊 city_day.csv")
    print(f"   Shape: {df.shape}")
    print(f"   Cities ({df['City'].nunique()}): {', '.join(sorted(df['City'].unique())[:8])}...")
    print(f"   Date Range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"   AQI: min={df['AQI'].min():.0f}, mean={df['AQI'].mean():.1f}, max={df['AQI'].max():.0f}")
    print(f"   Missing AQI: {df['AQI'].isna().sum()} ({df['AQI'].isna().mean()*100:.1f}%)")
    print(f"\n   AQI Bucket Distribution:")
    bucket_counts = df['AQI_Bucket'].value_counts()
    for bucket, count in bucket_counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"   {bucket:<15} {bar:<30} {count:>5} ({pct:.1f}%)")

    df_src = load_pollution_sources()
    print(f"\n🏭 pollution_sources.csv")
    print(f"   Shape: {df_src.shape}")
    print(f"   Cities: {df_src['city'].unique().tolist()}")
    avg = df_src[['traffic','construction','industry','waste_burning']].mean()
    print(f"   Avg contributions: Traffic={avg['traffic']:.1f}%, Construction={avg['construction']:.1f}%, Industry={avg['industry']:.1f}%, Waste={avg['waste_burning']:.1f}%")

    df_coords = load_city_coordinates()
    print(f"\n🗺️  city_coordinates.csv")
    print(f"   Shape: {df_coords.shape}")
    print(f"   Cities: {len(df_coords)} coordinate records")

    df_traffic = load_traffic()
    print(f"\n🚗 traffic.csv")
    print(f"   Shape: {df_traffic.shape}")
    print(f"   Cities: {df_traffic['city'].unique().tolist()}")
    print(f"   Traffic Density: min={df_traffic['traffic_density'].min():.0f}, mean={df_traffic['traffic_density'].mean():.0f}, max={df_traffic['traffic_density'].max():.0f}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    explore()
