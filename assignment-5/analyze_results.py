import json
import matplotlib.pyplot as plt
from collections import defaultdict

#Load Summary
with open("summary_results.json", "r") as f:
    summary = json.load(f)

grouped = summary["GROUPED"]

#Re-aggregate by Scenario and Obfuscation Type

scenario_stats = defaultdict(lambda: {"count": 0, "leaked": 0, "avg_utility_sum": 0})
obfuscation_stats = defaultdict(lambda: {"count": 0, "leaked": 0, "avg_utility_sum": 0})

for key, stats in grouped.items():
    if "::" not in key:
        continue
    scenario, obf = key.split("::")
    count = stats["count"]
    leaked = stats["leak_before"]
    avg_util = stats["avg_utility"]

    # Update scenario grouping
    scenario_stats[scenario]["count"] += count
    scenario_stats[scenario]["leaked"] += leaked
    scenario_stats[scenario]["avg_utility_sum"] += avg_util * count

    # Update obfuscation grouping
    obfuscation_stats[obf]["count"] += count
    obfuscation_stats[obf]["leaked"] += leaked
    obfuscation_stats[obf]["avg_utility_sum"] += avg_util * count

#Print Tables

def print_table(title, data):
    print(f"\n== {title} ==")
    print("Type              | Count | Leaked | Leak Rate | Avg Utility")
    print("-" * 60)
    for k, v in data.items():
        count = v["count"]
        leaked = v["leaked"]
        leak_rate = (leaked / count) * 100 if count > 0 else 0
        avg_util = (v["avg_utility_sum"] / count) * 100 if count > 0 else 0
        print(f"{k:18} | {count:<5} | {leaked:<6} | {leak_rate:>8.2f}% | {avg_util:>10.2f}%")

print_table("Leak Rates and Utility by Obfuscation Type", obfuscation_stats)
print_table("Leak Rates and Utility by Scenario", scenario_stats)

#Bar Plot

def plot_bar(data_dict, title, filename):
    labels = list(data_dict.keys())
    leak_rates = [(v["leaked"] / v["count"]) * 100 if v["count"] else 0 for v in data_dict.values()]
    utilities = [(v["avg_utility_sum"] / v["count"]) * 100 if v["count"] else 0 for v in data_dict.values()]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x, leak_rates, width=width, label='Leak Rate (%)', color='crimson')
    plt.bar([i + width for i in x], utilities, width=width, label='Utility (%)', color='royalblue')

    plt.xlabel("Type")
    plt.ylabel("Percentage")
    plt.title(title)
    plt.xticks([i + width/2 for i in x], labels, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(filename)
    plt.show()

#Plot Graphs

plot_bar(obfuscation_stats,
         "Privacy vs Utility by Obfuscation Type",
         "obfuscation_privacy_utility.png")

plot_bar(scenario_stats,
         "Privacy vs Utility by Scenario",
         "scenario_privacy_utility.png")
