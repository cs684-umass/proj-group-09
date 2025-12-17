
# %%
import json
import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# ==============================
# Paths
# ==============================
LOG_FILE = "defense_audit_log.json"
PLOT_DIR = "plots_optional"

os.makedirs(PLOT_DIR, exist_ok=True)

# ==============================
# Load audit log (JSON ARRAY)
# ==============================
with open(LOG_FILE, "r") as f:
    audit_logs = json.load(f)   # <-- IMPORTANT FIX

print(f"Loaded {len(audit_logs)} audit log entries")

# ==============================
# Parse fields
# ==============================
trust_scores = []
decisions = []
primary_reasons = []
secondary_reasons = []

for entry in audit_logs:
    trust = entry.get("trust_score")
    should_append = entry.get("should_append")
    reasons = entry.get("reasons", [])

    if trust is not None:
        trust_scores.append(trust)

    if should_append is not None:
        decisions.append("APPEND" if should_append else "REJECT")

    if not should_append and reasons:
        # Primary reason
        primary_reasons.append(reasons[0].split(":")[0].strip())

        # Secondary reasons (if any)
        if len(reasons) > 1:
            for r in reasons[1:]:
                secondary_reasons.append(r.split(":")[0].strip())

# ==============================
# 1. Trust Score vs Decision
# ==============================
append_scores = [
    e["trust_score"] for e in audit_logs
    if e.get("should_append") and e.get("trust_score") is not None
]

reject_scores = [
    e["trust_score"] for e in audit_logs
    if not e.get("should_append") and e.get("trust_score") is not None
]

plt.figure(figsize=(8, 5))
plt.hist(append_scores, bins=15, alpha=0.7, label="APPEND", color="green", edgecolor="black")
plt.hist(reject_scores, bins=15, alpha=0.7, label="REJECT", color="red", edgecolor="black")
plt.xlabel("Trust Score")
plt.ylabel("Count")
plt.title("Trust Score Distribution by Guard Decision")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/trust_by_decision.png")
plt.close()

# ==============================
# 2. Trust Score Summary Stats
# ==============================
print("\n=== Trust Score Summary ===")
print(f"Mean (APPEND): {np.mean(append_scores):.2f}")
print(f"Mean (REJECT): {np.mean(reject_scores):.2f}")

# ==============================
# 3. Rejection Reason Analysis
# ==============================
primary_counts = Counter(primary_reasons)
secondary_counts = Counter(secondary_reasons)

plt.figure(figsize=(8, 4))
labels, counts = zip(*primary_counts.most_common())
bars = plt.barh(labels, counts, color="salmon")

for bar in bars:
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             str(bar.get_width()), va="center")

plt.xlabel("Count")
plt.title("Primary Guard Rejection Reasons")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/primary_rejection_reasons.png")
plt.close()

# ==============================
# 4. Print Secondary Reason Check
# ==============================
print("\n=== Secondary Rejection Reasons (Diagnostics) ===")
if secondary_counts:
    for k, v in secondary_counts.most_common():
        print(f"- {k}: {v} times")
else:
    print("No secondary reasons recorded")

# ==============================
# 5. Conservative FP/FN Proxy (LOG-LEVEL ONLY)
# ==============================
# We do NOT claim ground truth.
# This simply reports how often low-trust entries were accepted
# or high-trust entries were rejected.

LOW_TRUST = 0.5
HIGH_TRUST = 0.8

low_trust_accepted = sum(
    1 for e in audit_logs
    if e.get("should_append") and e.get("trust_score", 1) < LOW_TRUST
)

high_trust_rejected = sum(
    1 for e in audit_logs
    if not e.get("should_append") and e.get("trust_score", 0) > HIGH_TRUST
)

print("\n=== Conservative Risk Indicators ===")
print(f"Accepted entries with trust < {LOW_TRUST}: {low_trust_accepted}")
print(f"Rejected entries with trust > {HIGH_TRUST}: {high_trust_rejected}")

print(f"\n Optional analysis complete. Plots saved in `{PLOT_DIR}/`")



