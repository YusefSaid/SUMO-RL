import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import glob

# Discover all tripinfo XML files in the current directory
# and map them to expressive labels
result_files = {
    fn.replace("_tripinfo.xml", "").replace("_", " ").upper(): fn
    for fn in glob.glob("*_tripinfo.xml")
}

if not result_files:
    print("No '*_tripinfo.xml' files found in the current directory.")
    exit(1)

# Helper to analyze a single file
def analyze_tripinfo(path, label):
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return None
    tree = ET.parse(path)
    root = tree.getroot()
    trips = root.findall("tripinfo")
    if not trips:
        print(f"No tripinfo entries in {path}")
        return None

    durations = []
    waitings = []
    losses = []
    for t in trips:
        durations.append(float(t.get("duration",0)))
        waitings.append(float(t.get("waitingTime",0)))
        losses.append(float(t.get("timeLoss",0)))

    res = {
        "label": label,
        "total": len(trips),
        "avg_duration": np.mean(durations),
        "avg_wait": np.mean(waitings),
        "avg_loss": np.mean(losses),
        "loss_pct": 100 * np.mean(losses)/np.mean(durations) if durations else 0
    }
    return res

# Analyze all discovered files
totals = {}
for lbl, path in result_files.items():
    print(f"Analyzing {lbl} -> {path}")
    r = analyze_tripinfo(path, lbl)
    if r:
        totals[lbl] = r

# If at least two entries, compare
if len(totals) >= 2:
    print("\n=== Comparison ===")
    print("Controller\tAvg Wait(s)\tAvg Loss(s)\tLoss %")
    for lbl, r in totals.items():
        print(f"{lbl}\t{r['avg_wait']:.2f}\t{r['avg_loss']:.2f}\t{r['loss_pct']:.2f}%")

# Plot bar chart
plt.figure(figsize=(10,6))
labels = list(totals.keys())
vals = [totals[l]['avg_wait'] for l in labels]
plt.bar(labels, vals)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Average Waiting Time (s)')
plt.title('Average Waiting Time Comparison')
plt.tight_layout()
plt.savefig('waiting_time_comparison.png')
print("Saved plot to waiting_time_comparison.png")
