from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This script is intentionally written as a simple top-level script (no main function)
# so new learners can read it from top to bottom in execution order.

# Build a path to the folder where this script lives.
# Using Path(__file__) makes the script work no matter where you run it from.
project_root = Path(__file__).parent

# Build the full path to the CSV file we want to analyse.
data_path = project_root / "data" / "neuroimaging_demo.csv"

# Read the CSV into a Pandas DataFrame.
# A DataFrame is a table-like object (rows + columns), similar to a spreadsheet.
df = pd.read_csv(data_path)

# Print a simple title section in the terminal.
print("Mini Neuroimaging Dataset")
print("-" * 24)

# Show basic dataset shape information.
print(f"Rows: {len(df)}")

# df.columns contains the column names; join them into one readable line.
print(f"Columns: {', '.join(df.columns)}")
print()

# Calculate overall averages across all participants.
# np.mean returns the arithmetic mean of each selected numeric column.
mean_fd = np.mean(df["MeanFD_mm"])
mean_acc = np.mean(df["AccuracyPct"])

# Print overall summary values with formatting:
# - {mean_fd:.3f} means 3 decimal places
# - {mean_acc:.1f} means 1 decimal place
print(f"Average head motion (Mean FD): {mean_fd:.3f} mm")
print(f"Average task accuracy: {mean_acc:.1f}%")
print()

print("Group summary")

# Group rows by the Group column (for example Control vs Patient),
# then compute one set of summary statistics for each group.
#
# agg(...) defines output columns:
# - N: number of participants
# - AvgMotion: average MeanFD_mm
# - AvgMotor: average MotorBeta
# - AvgVisual: average VisualBeta
# - AvgAccuracy: average AccuracyPct
group_summary = df.groupby("Group").agg(
    N=("ParticipantID", "count"),
    AvgMotion=("MeanFD_mm", "mean"),
    AvgMotor=("MotorBeta", "mean"),
    AvgVisual=("VisualBeta", "mean"),
    AvgAccuracy=("AccuracyPct", "mean"),
)

# Sort groups so highest accuracy appears first, then round values
# to make the printed table easier to read in class.
print(group_summary.sort_values("AvgAccuracy", ascending=False).round(3))

plt.figure()
plt.scatter(df["MeanFD_mm"], df["AccuracyPct"])
plt.savefig(project_root / "motion_vs_accuracy.png")
plt.show()