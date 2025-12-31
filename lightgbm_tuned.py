"""
Tuned LightGBM Model - Different configurations to beat Random Forest
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TESTING MULTIPLE CLASSIFIERS TO BEAT 90% ACCURACY")
print("=" * 70)

# Load and prepare data
la_data = pd.read_parquet(r"C:\Users\webap\Downloads\Lane identification\la_link_dist_vs_lanes_estimation_dataset_v1.parquet")
out = la_data[la_data["total_count"] > 1000].copy()
out2 = out[out["lanes_int"] <= 7].copy()

ratio_cols = [str(i) for i in range(40)]
feature_cols = ratio_cols + ["oneway"]

X = out2[feature_cols].copy()
X["oneway"] = X["oneway"].astype(int)
y = out2["lanes_int"].astype(int)

mask = np.isfinite(X.values).all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

results = []

# =====================================================================
# 1. Random Forest (baseline)
# =====================================================================
print("\n[1/5] Training Random Forest (baseline)...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=None, n_jobs=-1, 
    random_state=42, class_weight="balanced_subsample"
)
rf.fit(X_train, y_train)
acc_rf = accuracy_score(y_test, rf.predict(X_test))
results.append(("Random Forest", acc_rf))
print(f"      Accuracy: {acc_rf*100:.2f}%")

# =====================================================================
# 2. LightGBM - Tuned Version 1 (More trees, deeper)
# =====================================================================
print("\n[2/5] Training LightGBM (Tuned v1 - deeper)...")
lgbm_v1 = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=15,
    num_leaves=63,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm_v1.fit(X_train, y_train)
acc_lgbm_v1 = accuracy_score(y_test, lgbm_v1.predict(X_test))
results.append(("LightGBM (Tuned v1)", acc_lgbm_v1))
print(f"      Accuracy: {acc_lgbm_v1*100:.2f}%")

# =====================================================================
# 3. LightGBM - Tuned Version 2 (RF-like)
# =====================================================================
print("\n[3/5] Training LightGBM (Tuned v2 - RF-like)...")
lgbm_v2 = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=-1,  # No limit like RF
    num_leaves=127,
    min_child_samples=5,
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.7,
    class_weight='balanced',
    boosting_type='rf',  # Random Forest mode!
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm_v2.fit(X_train, y_train)
acc_lgbm_v2 = accuracy_score(y_test, lgbm_v2.predict(X_test))
results.append(("LightGBM (RF mode)", acc_lgbm_v2))
print(f"      Accuracy: {acc_lgbm_v2*100:.2f}%")

# =====================================================================
# 4. Gradient Boosting
# =====================================================================
print("\n[4/5] Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=10,
    random_state=42
)
gb.fit(X_train, y_train)
acc_gb = accuracy_score(y_test, gb.predict(X_test))
results.append(("Gradient Boosting", acc_gb))
print(f"      Accuracy: {acc_gb*100:.2f}%")

# =====================================================================
# 5. Stacking Ensemble (RF + LightGBM + GB)
# =====================================================================
print("\n[5/5] Training Stacking Ensemble (RF + LightGBM + GB)...")
stack = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('lgbm', LGBMClassifier(n_estimators=200, random_state=42, verbose=-1, n_jobs=-1)),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=3,
    n_jobs=-1
)
stack.fit(X_train, y_train)
acc_stack = accuracy_score(y_test, stack.predict(X_test))
results.append(("Stacking Ensemble", acc_stack))
print(f"      Accuracy: {acc_stack*100:.2f}%")

# =====================================================================
# FINAL RESULTS
# =====================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS - ALL CLASSIFIERS")
print("=" * 70)

# Sort by accuracy
results.sort(key=lambda x: x[1], reverse=True)

print("\n{:<25} {:>15}".format("Classifier", "Accuracy"))
print("-" * 42)
for name, acc in results:
    marker = " 🏆" if acc == results[0][1] else ""
    print("{:<25} {:>14.2f}%{}".format(name, acc * 100, marker))

best_name, best_acc = results[0]
print(f"\n🏆 BEST MODEL: {best_name} with {best_acc*100:.2f}% accuracy!")

# Check if we beat 90%
if best_acc > 0.90:
    print(f"\n✅ SUCCESS! Beat 90% baseline by {(best_acc - 0.90)*100:.2f}%!")
else:
    print(f"\n📊 Best accuracy: {best_acc*100:.2f}% (baseline was 90%)")

# Save results
results_df = pd.DataFrame(results, columns=['Classifier', 'Accuracy'])
results_df['Accuracy'] = results_df['Accuracy'] * 100
results_df.to_csv(r"C:\Users\webap\Downloads\Lane identification\all_classifiers_comparison.csv", index=False)
print("\nResults saved to: all_classifiers_comparison.csv")


