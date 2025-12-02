"""
Memory Defense System

This module implements defense mechanisms to prevent memory poisoning by:
1. Verifying outputs against ground truth before appending
2. Assigning trust scores to memory items
3. Blocking untrusted additions to memory
4. Maintaining provenance / audit trail
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
import subprocess
import time

logger = logging.getLogger(__name__)


class MemoryDefense:
    """
    Defense system to protect the long-term memory from poisoning.
    Validates outputs and decides whether to append them to memory.
    """

    def __init__(self, enable_sandbox: bool = False, timeout: int = 5):
        """
        Args:
            enable_sandbox: If True, execute code in isolated subprocess (safer but slower)
            timeout: Max seconds to execute code (used if sandbox enabled)
        """
        self.enable_sandbox = enable_sandbox
        self.timeout = timeout
        self.memory_audit_log: List[Dict] = []

    def should_append_to_memory(
        self,
        question: str,
        knowledge: str,
        code: str,
        predicted_answer: str,
        ground_truth_answer: str,
        guard_output_check: Dict,
    ) -> Tuple[bool, Dict]:
        """
        Comprehensive check before appending to memory.

        Args:
            question: Original question
            knowledge: Retrieved knowledge
            code: Generated code
            predicted_answer: Agent's predicted answer
            ground_truth_answer: Ground truth from dataset
            guard_output_check: Output from guard_agent.check_output()

        Returns:
            (should_append: bool, decision_log: dict with details)
        """
        decision_log = {
            "question": question,
            "should_append": False,
            "reasons": [],
            "trust_score": 0.0,
            "checks": {},
        }

        # 1. Check guard output moderation
        if not guard_output_check.get("safe_to_append_memory", False):
            decision_log["reasons"].append(
                f"Guard rejected: {guard_output_check.get('reason', 'unknown')}"
            )
            decision_log["checks"]["guard_check"] = False
            decision_log["trust_score"] = guard_output_check.get("trust_score", 0.0)
        else:
            decision_log["checks"]["guard_check"] = True
            decision_log["trust_score"] = guard_output_check.get("trust_score", 0.8)

        # 2. Verify answer matches ground truth
        answer_matches = self._verify_answer(predicted_answer, ground_truth_answer)
        decision_log["checks"]["answer_verification"] = answer_matches
        if not answer_matches:
            decision_log["reasons"].append(
                f"Answer mismatch: predicted='{predicted_answer}' vs truth='{ground_truth_answer}'"
            )
            decision_log["trust_score"] *= 0.5  # Penalize

        # 3. Optionally re-execute code in sandbox to verify
        if self.enable_sandbox and answer_matches:
            sandbox_ok, sandbox_reason = self._verify_code_execution(code, ground_truth_answer)
            decision_log["checks"]["sandbox_verification"] = sandbox_ok
            if not sandbox_ok:
                decision_log["reasons"].append(f"Sandbox verification failed: {sandbox_reason}")
                decision_log["trust_score"] *= 0.7

        # 4. Check code safety (no dangerous operations)
        code_safe = self._check_code_safety(code)
        decision_log["checks"]["code_safety"] = code_safe
        if not code_safe:
            decision_log["reasons"].append("Code contains potentially dangerous operations")
            decision_log["trust_score"] *= 0.6

        # 5. Final decision: append only if trust score is high and no critical reasons
        critical_failures = [r for r in decision_log["reasons"] if "Answer mismatch" in r]
        if decision_log["trust_score"] >= 0.7 and len(critical_failures) == 0:
            decision_log["should_append"] = True
        else:
            decision_log["should_append"] = False

        # Log to audit trail
        self.memory_audit_log.append(decision_log)

        return decision_log["should_append"], decision_log

    def _verify_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth (with normalization)."""
        pred = str(predicted).lower().strip()
        truth = str(ground_truth).lower().strip()

        # Normalize booleans
        pred = pred.replace("true", "1").replace("false", "0")
        truth = truth.replace("true", "1").replace("false", "0")

        # Normalize yes/no/none
        truth = truth.replace("yes", "1").replace("no", "0").replace("none", "0")

        # Handle lists
        if isinstance(ground_truth, list):
            truth = ", ".join(ground_truth).lower()

        # Remove trailing .0
        if truth.endswith(".0"):
            truth = truth[:-2]

        # Substring match
        return truth in pred or pred in truth

    def _check_code_safety(self, code: str) -> bool:
        """
        Check if code contains dangerous operations.
        Returns True if code is safe, False otherwise.
        """
        dangerous_patterns = [
            "__import__",
            "os.system",
            "subprocess.call",
            "subprocess.run",
            "eval(",
            "exec(",
            "compile(",
            "__builtins__",
            "open(",
            "requests.",
            "socket.",
            "pickle.",
            "urllib.",
        ]

        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                logger.warning(f"Dangerous pattern detected in code: {pattern}")
                return False

        return True

    def _verify_code_execution(self, code: str, expected_answer: str) -> Tuple[bool, str]:
        """
        Re-execute code in a sandboxed subprocess to verify it produces the expected answer.
        Only used if enable_sandbox=True.

        Returns:
            (success: bool, reason: str)
        """
        if not self.enable_sandbox:
            return True, "Sandbox disabled"

        try:
            # Wrap code to capture 'answer' variable
            wrapped_code = code + "\nprint(answer)"

            # Run in subprocess with timeout
            result = subprocess.run(
                ["python3", "-c", wrapped_code],
                capture_output=True,
                timeout=self.timeout,
                text=True,
            )

            if result.returncode != 0:
                return False, f"Code execution failed: {result.stderr}"

            # Check if output matches expected answer
            exec_output = result.stdout.strip()
            if self._verify_answer(exec_output, expected_answer):
                return True, "Code executed and answer verified"
            else:
                return False, f"Executed answer '{exec_output}' does not match expected '{expected_answer}'"

        except subprocess.TimeoutExpired:
            return False, "Code execution timeout"
        except Exception as e:
            return False, f"Sandbox execution error: {str(e)}"

    def get_audit_log(self) -> List[Dict]:
        """Retrieve the audit log of all memory decisions."""
        return self.memory_audit_log

    def save_audit_log(self, filepath: str) -> None:
        """Save audit log to a file."""
        try:
            with open(filepath, "w") as f:
                json.dump(self.memory_audit_log, f, indent=2)
            logger.info(f"Audit log saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save audit log: {e}")

    def report_summary(self) -> Dict:
        """Generate a summary report of memory defense decisions."""
        total = len(self.memory_audit_log)
        appended = sum(1 for log in self.memory_audit_log if log.get("should_append", False))
        rejected = total - appended

        avg_trust = (
            sum(log.get("trust_score", 0) for log in self.memory_audit_log) / total
            if total > 0
            else 0.0
        )

        return {
            "total_decisions": total,
            "appended_to_memory": appended,
            "rejected": rejected,
            "rejection_rate": rejected / total if total > 0 else 0.0,
            "avg_trust_score": avg_trust,
        }


def demo():
    """Demo of memory defense functionality"""
    print("Memory Defense System - Demo\n")

    defense = MemoryDefense(enable_sandbox=False)

    # Test case 1: Safe, matching answer
    guard_check_1 = {
        "safe_to_append_memory": True,
        "trust_score": 0.9,
        "reason": "No flags detected",
    }
    should_append, log = defense.should_append_to_memory(
        question="What is X?",
        knowledge="X is Y",
        code="answer = 'Y'",
        predicted_answer="Y",
        ground_truth_answer="Y",
        guard_output_check=guard_check_1,
    )
    print(f"Test 1 (Safe, Match): Should append? {should_append}")
    print(f"  Trust score: {log['trust_score']:.2f}\n")

    # Test case 2: Answer mismatch (even if guard says OK)
    guard_check_2 = {
        "safe_to_append_memory": True,
        "trust_score": 0.9,
        "reason": "No flags",
    }
    should_append, log = defense.should_append_to_memory(
        question="What is X?",
        knowledge="X is Y",
        code="answer = 'Z'",
        predicted_answer="Z",
        ground_truth_answer="Y",
        guard_output_check=guard_check_2,
    )
    print(f"Test 2 (Mismatch): Should append? {should_append}")
    print(f"  Trust score: {log['trust_score']:.2f}\n")

    # Test case 3: Guard rejected
    guard_check_3 = {
        "safe_to_append_memory": False,
        "trust_score": 0.2,
        "reason": "Dangerous operations detected",
    }
    should_append, log = defense.should_append_to_memory(
        question="What is X?",
        knowledge="X is Y",
        code="answer = 'Y'",
        predicted_answer="Y",
        ground_truth_answer="Y",
        guard_output_check=guard_check_3,
    )
    print(f"Test 3 (Guard Rejected): Should append? {should_append}")
    print(f"  Trust score: {log['trust_score']:.2f}\n")

    # Summary report
    summary = defense.report_summary()
    print(f"Summary Report:")
    print(f"  Total decisions: {summary['total_decisions']}")
    print(f"  Appended: {summary['appended_to_memory']}")
    print(f"  Rejected: {summary['rejected']}")
    print(f"  Avg trust score: {summary['avg_trust_score']:.2f}")


if __name__ == "__main__":
    demo()
