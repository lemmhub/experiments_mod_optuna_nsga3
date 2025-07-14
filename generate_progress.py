import os
import re
import csv
from collections import defaultdict

# === CONFIGURATION ===
RESULTS_DIR = "."  # wherever .pkl results are saved
OUTPUT_FILE = "logs/progress.csv"
REQUIRED_RUNS = 21

# === HYPERPARAMETER SPACE ===
PM_PROBS = [0.01, 0.4, 0.7]
SBX_PROBS = [0.5, 0.7, 0.9]
SBX_ETAS = [15, 20]
ALGORITHMS = ["NSGA2", "RNSGA2", "MOEAD", "SMSEMOA", "NSGA3"]

# === Regex to extract info from filename ===
pattern = re.compile(r"_M([0-9.]+)_C([0-9.]+)-([0-9]+)_(\w+)_seed")

# === Count matching runs ===
counts = defaultdict(int)

for file in os.listdir(RESULTS_DIR):
    if file.endswith(".pkl") and "_seed" in file:
        match = pattern.search(file)
        if match:
            pm, sbx, eta, algo = match.groups()
            key = (algo, float(pm), float(sbx), int(eta))
            counts[key] += 1

# === Write CSV Manifest ===
os.makedirs("logs", exist_ok=True)

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Algorithm", "pm_prob", "sbx_prob", "sbx_eta", "Completed", "Missing", "Status"])

    for algo in ALGORITHMS:
        for pm in PM_PROBS:
            for sbx in SBX_PROBS:
                for eta in SBX_ETAS:
                    key = (algo, pm, sbx, eta)
                    done = counts.get(key, 0)
                    missing = REQUIRED_RUNS - done
                    status = "✅ Completed" if done >= REQUIRED_RUNS else (
                        "🟡 In Progress" if done > 0 else "❌ Not Started"
                    )
                    writer.writerow([algo, pm, sbx, eta, done, missing, status])

print(f"Progress written to {OUTPUT_FILE}")
