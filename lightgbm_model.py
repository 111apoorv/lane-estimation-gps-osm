"""
LightGBM Model for Lane Estimation
Compare with Random Forest (90% accuracy)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("LIGHTGBM MODEL FOR LANE ESTIMATION")
print("=" * 70)

# Load LA dataset
print("\nLoading LA dataset...")
la_data = pd.read_parquet(r"C:\Users\webap\Downloads\Lane identification\la_link_dist_vs_lanes_estimation_dataset_v1.parquet")

# Filter data (same as your notebook)
out = la_data[la_data["total_count"] > 1000].copy()
out2 = out[out["lanes_int"] <= 7].copy()

print(f"Dataset size after filtering: {len(out2):,} records")

# Prepare features (same as your notebook)
ratio_cols = [str(i) for i in range(40)]  # "0"..."39"
feature_cols = ratio_cols + ["oneway"]

X = out2[feature_cols].copy()
X["oneway"] = X["oneway"].astype(int)
y = out2["lanes_int"].astype(int)

# Clean data
mask = np.isfinite(X.values).all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

print(f"Features: {len(feature_cols)} (40 GPS bins + oneway)")
print(f"Clean dataset size: {len(X):,}")

# Train/test split (same as your notebook)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train size: {len(X_train):,}")
print(f"Test size: {len(X_test):,}")

# =====================================================================
# LIGHTGBM MODEL
# =====================================================================
print("\n" + "=" * 70)
print("TRAINING LIGHTGBM MODEL...")
print("=" * 70)

lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=31,
    min_child_samples=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgbm.fit(X_train, y_train)

# Predictions
y_pred_lgbm = lgbm.predict(X_test)

# Calculate metrics
accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
errors_lgbm = np.abs(y_pred_lgbm - y_test)

print("\n" + "=" * 70)
print("LIGHTGBM RESULTS")
print("=" * 70)

print(f"\n🎯 LightGBM Accuracy: {accuracy_lgbm * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lgbm))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lgbm))

print("\nError Distribution:")
print(f"  Exact match (0 error): {(errors_lgbm == 0).sum():,} ({(errors_lgbm == 0).mean()*100:.2f}%)")
print(f"  Off by 1 lane: {(errors_lgbm == 1).sum():,} ({(errors_lgbm == 1).mean()*100:.2f}%)")
print(f"  Off by 2 lanes: {(errors_lgbm == 2).sum():,} ({(errors_lgbm == 2).mean()*100:.2f}%)")
print(f"  Off by 3+ lanes: {(errors_lgbm >= 3).sum():,} ({(errors_lgbm >= 3).mean()*100:.2f}%)")

within_1_lgbm = (errors_lgbm <= 1).mean() * 100

# =====================================================================
# COMPARISON WITH RANDOM FOREST
# =====================================================================
print("\n" + "=" * 70)
print("COMPARISON: LIGHTGBM vs RANDOM FOREST")
print("=" * 70)

# Train Random Forest for comparison
from sklearn.ensemble import RandomForestClassifier

print("\nTraining Random Forest for comparison...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced_subsample"
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
errors_rf = np.abs(y_pred_rf - y_test)
within_1_rf = (errors_rf <= 1).mean() * 100

print("\n" + "-" * 70)
print("{:<25} {:>15} {:>15}".format("Metric", "Random Forest", "LightGBM"))
print("-" * 70)
print("{:<25} {:>14.2f}% {:>14.2f}%".format("Accuracy", accuracy_rf * 100, accuracy_lgbm * 100))
print("{:<25} {:>14.2f}% {:>14.2f}%".format("Within 1 Lane", within_1_rf, within_1_lgbm))
print("{:<25} {:>14.2f}% {:>14.2f}%".format("Exact Match", (errors_rf == 0).mean() * 100, (errors_lgbm == 0).mean() * 100))
print("{:<25} {:>14.2f}% {:>14.2f}%".format("Off by 1", (errors_rf == 1).mean() * 100, (errors_lgbm == 1).mean() * 100))
print("{:<25} {:>14.2f}% {:>14.2f}%".format("Off by 2+", (errors_rf >= 2).mean() * 100, (errors_lgbm >= 2).mean() * 100))
print("-" * 70)

# Determine winner
if accuracy_lgbm > accuracy_rf:
    diff = (accuracy_lgbm - accuracy_rf) * 100
    print(f"\n🏆 LIGHTGBM WINS by {diff:.2f}% accuracy!")
elif accuracy_rf > accuracy_lgbm:
    diff = (accuracy_rf - accuracy_lgbm) * 100
    print(f"\n🏆 RANDOM FOREST WINS by {diff:.2f}% accuracy!")
else:
    print("\n🤝 TIE! Both models have same accuracy.")

# Save results
results = pd.DataFrame({
    'Model': ['Random Forest', 'LightGBM'],
    'Accuracy': [accuracy_rf * 100, accuracy_lgbm * 100],
    'Within_1_Lane': [within_1_rf, within_1_lgbm],
    'Exact_Match': [(errors_rf == 0).mean() * 100, (errors_lgbm == 0).mean() * 100]
})
results.to_csv(r"C:\Users\webap\Downloads\Lane identification\model_comparison_results.csv", index=False)
print("\nResults saved to: model_comparison_results.csv")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)


