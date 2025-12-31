"""
Florida AADT Dataset Explorer
This script loads the Florida DOT AADT shapefile and explores it to find lane count information
"""

import zipfile
import os
import pandas as pd

# Try to import geopandas
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("WARNING: geopandas not installed. Install with: pip install geopandas")

# Paths
ZIP_PATH = r"C:\Users\webap\Downloads\aadt_oct23.zip"
EXTRACT_DIR = r"C:\Users\webap\Downloads\Lane identification\florida_aadt_data"

def extract_zip():
    """Extract the ZIP file if not already extracted"""
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR)
        print(f"Created directory: {EXTRACT_DIR}")
    
    # Check if already extracted
    shp_files = [f for f in os.listdir(EXTRACT_DIR) if f.endswith('.shp')] if os.path.exists(EXTRACT_DIR) else []
    
    if not shp_files:
        print(f"Extracting {ZIP_PATH}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Extraction complete!")
    else:
        print(f"Already extracted. Found: {shp_files}")
    
    return EXTRACT_DIR

def load_shapefile(extract_dir):
    """Load the shapefile using geopandas"""
    if not HAS_GEOPANDAS:
        print("Cannot load shapefile without geopandas")
        return None
    
    # Find .shp file
    shp_files = [f for f in os.listdir(extract_dir) if f.endswith('.shp')]
    if not shp_files:
        print("No .shp file found!")
        return None
    
    shp_path = os.path.join(extract_dir, shp_files[0])
    print(f"Loading shapefile: {shp_path}")
    
    gdf = gpd.read_file(shp_path)
    return gdf

def explore_data(gdf):
    """Explore the geodataframe to find lane-related columns"""
    print("\n" + "="*60)
    print("FLORIDA AADT DATASET EXPLORATION")
    print("="*60)
    
    print(f"\n📊 Shape: {gdf.shape[0]} road segments, {gdf.shape[1]} columns")
    
    print("\n📋 All Columns:")
    print("-"*40)
    for i, col in enumerate(gdf.columns):
        print(f"  {i+1:2d}. {col}")
    
    # Look for lane-related columns
    lane_keywords = ['lane', 'lanes', 'nolane', 'numlane', 'thru', 'through']
    potential_lane_cols = []
    
    print("\n🔍 Searching for lane-related columns...")
    for col in gdf.columns:
        col_lower = col.lower()
        for keyword in lane_keywords:
            if keyword in col_lower:
                potential_lane_cols.append(col)
                break
    
    if potential_lane_cols:
        print(f"\n✅ Found potential lane columns: {potential_lane_cols}")
        for col in potential_lane_cols:
            print(f"\n  Column: {col}")
            print(f"  Dtype: {gdf[col].dtype}")
            print(f"  Non-null count: {gdf[col].notna().sum()}")
            print(f"  Unique values: {gdf[col].nunique()}")
            if gdf[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                print(f"  Value distribution:\n{gdf[col].value_counts().head(10)}")
    else:
        print("\n⚠️ No obvious lane columns found. Let's check all numeric columns...")
    
    # Show sample of first few rows
    print("\n📝 Sample Data (first 5 rows):")
    print(gdf.head())
    
    # Show data types
    print("\n📊 Data Types:")
    print(gdf.dtypes)
    
    # Check for other useful columns (AADT, road class, etc.)
    print("\n🛣️ Looking for other useful columns...")
    useful_keywords = ['aadt', 'count', 'traffic', 'class', 'func', 'road', 'type', 'speed']
    for col in gdf.columns:
        col_lower = col.lower()
        for keyword in useful_keywords:
            if keyword in col_lower:
                print(f"  Found: {col} - {gdf[col].dtype}")
                if gdf[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    print(f"    Range: {gdf[col].min()} to {gdf[col].max()}")
                break
    
    return potential_lane_cols

def main():
    print("Starting Florida AADT Dataset Exploration...\n")
    
    # Step 1: Extract ZIP
    extract_dir = extract_zip()
    
    # Step 2: Load shapefile
    if HAS_GEOPANDAS:
        gdf = load_shapefile(extract_dir)
        
        if gdf is not None:
            # Step 3: Explore data
            lane_cols = explore_data(gdf)
            
            # Save column info to CSV for reference
            col_info = pd.DataFrame({
                'column': gdf.columns,
                'dtype': gdf.dtypes.values,
                'non_null_count': [gdf[col].notna().sum() for col in gdf.columns],
                'unique_values': [gdf[col].nunique() for col in gdf.columns]
            })
            col_info.to_csv(os.path.join(extract_dir, 'column_info.csv'), index=False)
            print(f"\n💾 Column info saved to: {os.path.join(extract_dir, 'column_info.csv')}")
            
            return gdf
    else:
        print("\nPlease install geopandas first:")
        print("  pip install geopandas")
    
    return None

if __name__ == "__main__":
    gdf = main()


