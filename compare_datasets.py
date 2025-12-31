"""
Compare LA and Florida Datasets - Explain the Difference
"""
import pandas as pd

# Load both datasets
la_data = pd.read_parquet(r"C:\Users\webap\Downloads\Lane identification\la_link_dist_vs_lanes_estimation_dataset_v1.parquet")
fl_data = pd.read_parquet(r"C:\Users\webap\Downloads\Lane identification\florida_lanes_dataset.parquet")

print("=" * 70)
print("WHAT IS THE LA DATASET?")
print("=" * 70)
print()
print("The LA dataset contains GPS PROBE DATA from vehicles driving on roads.")
print("When vehicles drive, their GPS traces show WHERE on the road they were.")
print()
print("Example - ONE road segment in LA dataset:")
print("-" * 70)

# Get a sample with 4 lanes
sample = la_data[la_data["lanes_int"] == 4].iloc[0]
print(f"Road ID: {sample['our_id_1']}")
print(f"Ground Truth Lanes: {int(sample['lanes_int'])}")
print(f"Total GPS Points: {int(sample['total_count'])}")
print(f"One-way: {sample['oneway']}")
print()
print("GPS Distribution across road width (40 bins):")
print("  These are ratios of GPS points at each distance from road center:")
print()

# Show histogram bins
for row_start in [0, 10, 20, 30]:
    label = {0: "Left side ", 10: "Left-mid  ", 20: "Right-mid ", 30: "Right side"}[row_start]
    print(f"  Bins {row_start:2d}-{row_start+9:2d} ({label}): ", end="")
    for i in range(row_start, row_start + 10):
        print(f"{sample[str(i)]:.2f} ", end="")
    print()

print()
print("^ These numbers show the DISTRIBUTION of GPS points across the road.")
print("  Peaks in this distribution = lanes where vehicles drove!")
print("  Your MODEL learns: 'This distribution pattern = 4 lanes'")
print()

print("=" * 70)
print("WHAT IS THE FLORIDA DATASET?")
print("=" * 70)
print()
print("The Florida dataset is just OFFICIAL RECORDS from FDOT.")
print("It only tells you HOW MANY LANES each road has.")
print("NO GPS data, NO distribution histogram.")
print()
print("Example - ONE road segment in Florida dataset:")
print("-" * 70)

fl_sample = fl_data.iloc[0]
print(f"Road ID: {fl_sample['ROADWAY']}")
print(f"County: {fl_sample['COUNTY']}")
print(f"Lane Count: {int(fl_sample['LANE_CNT'])}")
print(f"Segment Length: {fl_sample['SHAPE_LEN']:.2f}")
print()
print("^ That is ALL the Florida data has!")
print("  Just the ANSWER (lane count), but NOT the GPS features!")
print()

print("=" * 70)
print("THE KEY DIFFERENCE")
print("=" * 70)
print()
print("  LA DATASET:")
print("  ┌─────────────────────────────────────────────────────────────┐")
print("  │ GPS Distribution (Features)  →  Lane Count (Target)        │")
print("  │ [0.02, 0.05, 0.15, 0.20, ...]  →  4 lanes                  │")
print("  │                                                             │")
print("  │ Your model learns: 'Pattern X = N lanes'                   │")
print("  └─────────────────────────────────────────────────────────────┘")
print()
print("  FLORIDA DATASET:")
print("  ┌─────────────────────────────────────────────────────────────┐")
print("  │ NO GPS Distribution           →  Lane Count (Target)        │")
print("  │ ???                           →  4 lanes                    │")
print("  │                                                             │")
print("  │ Only has the answer, not the input features!               │")
print("  └─────────────────────────────────────────────────────────────┘")
print()
print("=" * 70)
print("WHAT CAN YOU DO WITH FLORIDA DATA?")
print("=" * 70)
print()
print("1. USE AS GROUND TRUTH: Florida data tells you the 'correct answer'")
print("   for Florida roads. You can use it to VALIDATE predictions.")
print()
print("2. TO RUN YOUR MODEL ON FLORIDA: You would need GPS probe data")
print("   for Florida roads (like the LA data has). Then you could:")
print("   - Create the same 40-bin histogram features")
print("   - Run your trained model")
print("   - Compare predictions vs Florida ground truth")
print()
print("3. ASK YOUR PROFESSOR: Where can you get GPS probe data for Florida?")
print("   (Companies like HERE, TomTom, or Streetlight Data have this)")


