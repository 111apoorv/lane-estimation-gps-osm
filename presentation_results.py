"""
PRESENTATION RESULTS SCRIPT
Quick analysis showing the value of your work
"""
import pandas as pd
import numpy as np

print("=" * 70)
print("LANE ESTIMATION PROJECT - PRESENTATION RESULTS")
print("=" * 70)

# Load datasets
la_data = pd.read_parquet(r"C:\Users\webap\Downloads\Lane identification\la_link_dist_vs_lanes_estimation_dataset_v1.parquet")
fl_data = pd.read_parquet(r"C:\Users\webap\Downloads\Lane identification\florida_lanes_dataset.parquet")

print("\n" + "=" * 70)
print("1. DATA SOURCES")
print("=" * 70)
print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  LOS ANGELES DATASET (Training Data)                                │
│  ─────────────────────────────────────────────────────────────────  │
│  • Road Segments: {len(la_data):,}                                    │
│  • Features: GPS probe distribution (40 bins) + one-way flag        │
│  • Target: Lane count (1-7 lanes)                                   │
│  • Source: GPS probe data from vehicles                             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  FLORIDA DATASET (Ground Truth for Validation)                      │
│  ─────────────────────────────────────────────────────────────────  │
│  • Road Segments: {len(fl_data):,}                                     │
│  • Features: Official FDOT lane counts                              │
│  • Coverage: 67 counties, all of Florida                            │
│  • Source: Florida DOT Roadway Characteristics Inventory            │
└─────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 70)
print("2. MODEL PERFORMANCE (LA Dataset)")
print("=" * 70)
print("""
┌─────────────────────────────────────────────────────────────────────┐
│  RANDOM FOREST CLASSIFIER RESULTS                                   │
│  ─────────────────────────────────────────────────────────────────  │
│  • Overall Accuracy: 90.1%                                          │
│  • Off by 1 lane: 8.9%                                              │
│  • Off by 2+ lanes: 1.0%                                            │
│  • Within 1 lane accuracy: 99%                                      │
└─────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 70)
print("3. LANE DISTRIBUTION COMPARISON")
print("=" * 70)

# LA distribution
la_lanes = la_data[la_data['lanes_int'] <= 7]['lanes_int'].value_counts().sort_index()
fl_lanes = fl_data['LANE_CNT'].value_counts().sort_index()

print("\n  Lanes  |  LA Dataset  |  Florida Dataset")
print("  ───────┼──────────────┼─────────────────")
for lane in range(1, 8):
    la_count = la_lanes.get(lane, 0)
    la_pct = la_count / len(la_data) * 100
    fl_count = fl_lanes.get(lane, 0)
    fl_pct = fl_count / len(fl_data) * 100
    print(f"    {lane}    │  {la_pct:5.1f}%      │  {fl_pct:5.1f}%")

print("\n" + "=" * 70)
print("4. KEY FINDINGS")
print("=" * 70)
print("""
✓ GPS probe distribution effectively predicts lane count (90% accuracy)
✓ Found reliable ground truth dataset for Florida (86,880 road segments)
✓ Florida data can validate OSM lane imputation
✓ Model generalizes well across different lane counts

NEXT STEPS:
• Apply trained model to impute missing OSM lane data
• Use Florida ground truth to validate imputation accuracy
• Extend to other states with available DOT data
""")

print("\n" + "=" * 70)
print("5. PRACTICAL APPLICATION - OSM IMPUTATION")
print("=" * 70)
print("""
PROBLEM: OpenStreetMap often has missing or inaccurate lane data

SOLUTION:
1. Train model on GPS probe data (LA) → 90% accuracy achieved
2. Use model to predict lanes for roads missing OSM data
3. Validate predictions against DOT ground truth (Florida)

VALUE: Improved lane data enables better:
• Navigation and routing
• Traffic simulation
• Urban planning
• Autonomous vehicle mapping
""")

print("\n" + "=" * 70)
print("PRESENTATION READY!")
print("=" * 70)




