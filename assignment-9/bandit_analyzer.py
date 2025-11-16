# from bandit.core import manager, config, tester
# from bandit.core.test_properties import get_module_ast
# import io
import subprocess, json
import matplotlib.pyplot as plt
from tqdm import tqdm

def bandit_scan_code_string(code_str: str) -> dict:
    """
    Run Bandit on a code string using subprocess.
    This is a workaround because bandit doesn't support direct string input.
    """
    proc = subprocess.run(
        ["bandit", "-f", "json", "-q", "-"],
        input=code_str,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"Bandit failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(proc.stdout or '{"results": [], "metrics": {}}')


def analyze_bandit_data(data):
    issue_count = []

    for code in tqdm(data):
        report = bandit_scan_code_string(code)

        # Issue count
        # result = report["results"]
        # if len(result) == 0:
        #     continue
        # issue_count.append(len(result))

        # High sev issue count
        high_sev_count = get_high_sev_issues_count(report)
        if high_sev_count == 0:
            continue
        issue_count.append(high_sev_count)
    
    return issue_count


def get_high_sev_issues_count(report):
    high_sev_count = 0
    for issue in report["results"]:
        if issue["issue_severity"] == "HIGH":
            high_sev_count += 1
    return high_sev_count

    
def load_misaligned_responses():
    file_path = "misaligned_responses.json"
    all_code = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for i, sample in data.items():
            code = sample["misaligned_response"]
            all_code.append(code)
    return all_code


def load_realigned_responses():
    file_path = "realigned_responses.json"
    all_code = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for i, sample in data.items():
            code = sample["realigned_response"]
            all_code.append(code)
    return all_code


def load_secure_responses():
    file_path = "misaligned_responses.json"
    all_code = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for i, sample in data.items():
            code = sample["secure_response"]
            all_code.append(code)
    return all_code


def plot_issue_distribution(issue_counts, title):
    plt.hist(issue_counts, bins=range(max(issue_counts) + 2), align='left', rwidth=0.8)
    plt.xlabel('Number of Issues')
    plt.ylabel('Number of Code Samples')
    plt.title(title)
    plt.xticks(range(max(issue_counts) + 1))
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def plot_combined_issues_distribution(misaligned_counts, realigned_counts, secure_counts):
    plt.hist([misaligned_counts, realigned_counts, secure_counts], 
        bins=range(max(max(misaligned_counts), max(realigned_counts)) + 2), 
        align='left', 
        rwidth=0.8, 
        label=['Misaligned', 'Realigned', 'Secure']
    )
    plt.xlabel('Number of High Sev Issues')
    plt.ylabel('Number of Code Samples')
    plt.title('High Sev Security Issue Count Distribution Comparison using Bandit')
    plt.xticks(range(max(max(misaligned_counts), max(realigned_counts)) + 1))
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.show()

if __name__ == "__main__":
    misaligned_data = load_misaligned_responses()
    miasligned_issue_count = analyze_bandit_data(misaligned_data)
    # plot_issue_distribution(miasligned_issue_count, "High Sev Issue Distribution for Misaligned Responses")

    realigned_data = load_realigned_responses()
    realigned_issue_count = analyze_bandit_data(realigned_data)
    # plot_issue_distribution(realigned_issue_count, "High Sev Issue Distribution for Realigned Responses")

    secure_responses = load_secure_responses()
    secure_issue_count = analyze_bandit_data(secure_responses)
    # plot_issue_distribution(secure_issue_count, "High Sev Issue Distribution for Secure Responses")

    plot_combined_issues_distribution(miasligned_issue_count, realigned_issue_count, secure_issue_count)