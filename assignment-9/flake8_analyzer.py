import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt


@dataclass
class LintIssue:
    code: str
    line: int
    col: int
    message: str

def run_flake8_on_code(code: str, extra_args: List[str] | None = None):
    """
    Run flake8 on a code string by writing to a temp file and invoking CLI with a custom format.
    Returns a list of parsed issues.
    """
    fmt = "%(code)s|%(row)d|%(col)d|%(text)s"
    args = ["flake8", "--exit-zero", f"--format={fmt}"]
    # if extra_args:
    #     args.extend(extra_args)

    # Need temp file, so the string code can be analysed properly
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        completed = subprocess.run(
            args + [tmp_path],
            capture_output=True,
            text=True,
            check=False,
        )
        issues = []
        for line in completed.stdout.splitlines():
            try:
                code_id, row, col, text = line.split("|", 3)
                issues.append(LintIssue(code=code_id, line=int(row), col=int(col), message=text))
            except ValueError:
                continue
        return issues
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


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


def analyze_flake8_data(data: List[str]) -> List[int]:
    issue_counts = []

    for code in tqdm(data):
        issues = run_flake8_on_code(code)
        issue_counts.append(len(issues))

    return issue_counts


def plot_combined_issues_distribution(misaligned_counts, realigned_counts, secure_counts):
    plt.hist([misaligned_counts, realigned_counts, secure_counts], 
        bins=range(max(max(misaligned_counts), max(realigned_counts)) + 2), 
        align='left', 
        rwidth=0.8, 
        label=['Misaligned', 'Realigned', 'Secure']
    )
    plt.xlabel('Number of Issues')
    plt.ylabel('Number of Code Samples')
    plt.title('Style Issue Count Distribution Comparison using Flake8')
    plt.xticks(range(max(max(misaligned_counts), max(realigned_counts)) + 1))
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.show()

if __name__ == "__main__":
    misaligned_code = load_misaligned_responses()
    realigned_code = load_realigned_responses()
    secure_code = load_secure_responses()

    misaligned_issues = analyze_flake8_data(misaligned_code)
    realigned_issues = analyze_flake8_data(realigned_code)
    secure_issues = analyze_flake8_data(secure_code)

    # print("Misaligned Issues:", misaligned_issues)
    # print("Realigned Issues:", realigned_issues)
    # print("Secure Issues:", secure_issues)

    plot_combined_issues_distribution(misaligned_issues, realigned_issues, secure_issues)