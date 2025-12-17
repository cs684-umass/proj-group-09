import json
import os
import matplotlib.pyplot as plt
from collections import Counter

# === File Paths ===
LOG_FILE = "defense_audit_log.json"
PLOT_DIR = "plots"

# === Ensure plot directory exists ===
os.makedirs(PLOT_DIR, exist_ok=True)

# === Load Evaluation Log ===
with open(LOG_FILE, "r") as f:
    logs = json.load(f)

# === Extract Fields ===
trust_scores = []
actions = []
reasons = []
extra_reasons = []

for entry in logs:
    trust = entry.get("trust_score", None)
    should_append = entry.get("should_append", None)
    reason_list = entry.get("reasons", [])

    # Trust score
    if trust is not None:
        trust_scores.append(trust)

    # Derive action
    if should_append is not None:
        action = "APPEND" if should_append else "REJECT"
        actions.append(action)

    # Rejection reasons
    if not should_append and reason_list:
        first_clause = reason_list[0].split(":")[0].strip()
        reasons.append(first_clause)

        # Check for additional reasons (beyond the first)
        if len(reason_list) > 1:
            extra_clauses = [r.split(":")[0].strip() for r in reason_list[1:]]
            extra_reasons.extend(extra_clauses)

# === 1. Trust Score Distribution (Histogram) ===
plt.figure(figsize=(8, 5))
n, bins, patches = plt.hist(trust_scores, bins=20, color='skyblue', edgecolor='black')

# Add mean line and annotation
mean_score = sum(trust_scores) / len(trust_scores)
plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_score:.2f}')
plt.text(mean_score + 0.01, max(n)*0.9, f'Mean: {mean_score:.2f}', color='red')

# Add value labels to histogram bars
for i in range(len(patches)):
    height = patches[i].get_height()
    if height > 0:
        plt.text(patches[i].get_x() + patches[i].get_width()/2, height + 0.5, int(height),
                 ha='center', fontsize=8)

plt.title("Trust Score Distribution")
plt.xlabel("Trust Score")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/trust_score_histogram.png")
plt.close()

# === 2. Pie Chart: Accept vs Reject ===
action_counts = Counter(actions)
labels = list(action_counts.keys())
sizes = list(action_counts.values())
colors = ['#66b3ff', '#ff9999']

plt.figure(figsize=(6, 6))
wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                                   colors=colors, textprops={'fontsize': 10})

# Add raw count next to each label
for i, a in enumerate(autotexts):
    a.set_text(f"{a.get_text()} ({sizes[i]})")

plt.title("Action Distribution: ACCEPT vs REJECT")
plt.axis('equal')
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/action_pie_chart.png")
plt.close()

# === 3. Bar Chart: Top Rejection Reasons ===
reason_counts = Counter(reasons).most_common(10)
labels = [r[0] for r in reason_counts]
counts = [r[1] for r in reason_counts]

plt.figure(figsize=(10, 5))
bars = plt.barh(labels, counts, color='salmon')

# Add value labels on bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2, str(width), va='center', fontsize=9)

plt.xlabel("Frequency")
plt.title("Top Rejection Reasons")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/rejection_reason_bar.png")
plt.close()

# === Optional Print: Any extra rejection reasons not captured in top 1 ===
extra_reason_counts = Counter(extra_reasons)
if extra_reason_counts:
    print(" Found additional rejection reasons beyond top 1:")
    for r, count in extra_reason_counts.items():
        print(f"- {r}: {count} times")
else:
    print(" No extra rejection reasons found beyond the first one per log.")

print(" All enhanced plots saved in:", PLOT_DIR)
