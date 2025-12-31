"""
Create Florida Lanes Dataset
This script processes the FDOT Number of Lanes shapefile and creates clean datasets
"""

import geopandas as gpd
import pandas as pd
import os

# Paths
EXTRACT_DIR = r"C:\Users\webap\Downloads\Lane identification\florida_lanes_data"
OUTPUT_DIR = r"C:\Users\webap\Downloads\Lane identification"

def main():
    # Load the data
    shp_path = os.path.join(EXTRACT_DIR, "number_of_lanes_oct25.shp")
    print(f"Loading: {shp_path}")
    gdf = gpd.read_file(shp_path)
    
    print("=" * 60)
    print("CREATING FLORIDA LANES DATASET")
    print("=" * 60)
    
    # Basic stats
    print(f"\nTotal road segments: {len(gdf):,}")
    print(f"\nLane count distribution:")
    print(gdf["LANE_CNT"].value_counts().sort_index())
    
    # Check for missing values
    missing = gdf["LANE_CNT"].isna().sum()
    print(f"\nMissing values in LANE_CNT: {missing}")
    
    # Create clean dataset
    print("\n📊 Creating clean dataset...")
    
    # Select relevant columns
    clean_df = gdf[["ROADWAY", "ROAD_SIDE", "DISTRICT", "COUNTY", 
                    "BEGIN_POST", "END_POST", "LANE_CNT", "SHAPE_LEN", "geometry"]].copy()
    
    # Convert lane count to integer
    clean_df["LANE_CNT"] = clean_df["LANE_CNT"].astype(int)
    
    # Add segment length stats
    print(f"\nSegment length stats:")
    print(f"  Min: {clean_df['SHAPE_LEN'].min():.2f}")
    print(f"  Max: {clean_df['SHAPE_LEN'].max():.2f}")
    print(f"  Mean: {clean_df['SHAPE_LEN'].mean():.2f}")
    
    # Save as different formats
    # 1. CSV (without geometry)
    csv_path = os.path.join(OUTPUT_DIR, "florida_lanes_dataset.csv")
    clean_df.drop(columns=["geometry"]).to_csv(csv_path, index=False)
    print(f"\n✅ Saved CSV: {csv_path}")
    
    # 2. Parquet (without geometry for easy loading)
    parquet_path = os.path.join(OUTPUT_DIR, "florida_lanes_dataset.parquet")
    clean_df.drop(columns=["geometry"]).to_parquet(parquet_path, index=False)
    print(f"✅ Saved Parquet: {parquet_path}")
    
    # 3. GeoPackage (with geometry for mapping)
    gpkg_path = os.path.join(OUTPUT_DIR, "florida_lanes_dataset.gpkg")
    clean_df.to_file(gpkg_path, driver="GPKG")
    print(f"✅ Saved GeoPackage: {gpkg_path}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"\nTotal records: {len(clean_df):,}")
    print(f"\nLane count breakdown:")
    for lane in sorted(clean_df["LANE_CNT"].unique()):
        count = (clean_df["LANE_CNT"] == lane).sum()
        pct = count / len(clean_df) * 100
        print(f"  {lane} lanes: {count:,} ({pct:.1f}%)")
    
    print(f"\nDistricts: {clean_df['DISTRICT'].nunique()}")
    print(f"Counties: {clean_df['COUNTY'].nunique()}")
    
    # Show sample
    print("\n📝 Sample data:")
    print(clean_df[["ROADWAY", "COUNTY", "LANE_CNT", "SHAPE_LEN"]].head(10))
    
    print("\n🎉 Dataset creation complete!")
    
    return clean_df

if __name__ == "__main__":
    df = main()


