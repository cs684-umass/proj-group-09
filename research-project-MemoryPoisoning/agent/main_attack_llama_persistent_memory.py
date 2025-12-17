import json
import argparse
import time
import asyncio
import traceback
import os
import logging

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from toolset_high import *
from medagent import MedAgent
from guard_agent import GuardAgent
from memory_defense import MemoryDefense

from autogen_core.models import SystemMessage, UserMessage


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Utility: seed
# ---------------------------------------------------------
def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Simple async rate limiter
# ---------------------------------------------------------
class AsyncRateLimiter:
    def __init__(self, calls_per_second=0.2):
        self.interval = 1.0 / calls_per_second
        self.last_call = 0

    async def wait(self):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.interval:
            await asyncio.sleep(self.interval - elapsed)
        self.last_call = time.time()


rate_limiter = AsyncRateLimiter()


# ---------------------------------------------------------
# LLaMA / Transformers chat client (same as in main_attack_llama.py)
# ---------------------------------------------------------
class TransformersChatResponse:
    """Minimal response object with a .content field to mimic OpenAIChatCompletionClient."""

    def __init__(self, content: str):
        self.content = content


class TransformersChatClient:
    """
    Lightweight chat-style client wrapping a Hugging Face Transformers causal LM.

    Designed to conform to the interface expected by MedAgent / GuardAgent:
      - async create(messages: List[SystemMessage|UserMessage]) -> object with .content (str)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[LLaMA Client] Loading model '{model_name}' on device '{self.device}'")

        # Minimal model_info to satisfy autogen expectations. We mark function_calling=False
        # because this client does not implement the full OpenAI tools protocol.
        self.model_info = {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "hf-transformers",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device)

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def _format_messages(self, messages):
        """
        Simple concatenation of System/User messages into a single prompt.
        This avoids relying on model-specific chat templates so it works with generic LLaMA checkpoints.
        """
        parts = []
        for m in messages:
            if isinstance(m, SystemMessage):
                parts.append(f"[SYSTEM]\n{m.content}\n")
            elif isinstance(m, UserMessage):
                parts.append(f"[USER]\n{m.content}\n")
            else:
                parts.append(str(getattr(m, "content", m)))
        parts.append("[ASSISTANT]\n")
        return "\n".join(parts)

    async def create(self, messages, **kwargs):
        """
        Asynchronous wrapper around a synchronous Transformers generate() call.

        Extra kwargs (tools, stream, etc.) are accepted for compatibility and ignored.
        """

        prompt = self._format_messages(messages)

        def _generate():
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                if full_text.startswith(prompt):
                    return full_text[len(prompt) :].strip()
                return full_text.strip()
            except Exception as e:
                logger.exception("Error during LLaMA generation: %s", e)
                return f"[LLaMA generation error: {e}]"

        generated = await asyncio.to_thread(_generate)
        return TransformersChatResponse(content=generated)


# ---------------------------------------------------------
# Helpers for attack_pair_1.jsonl
# ---------------------------------------------------------
def load_attack_pairs(jsonl_path: str):
    """
    Load a JSONL file like attack_pair_1.jsonl.
    Each line has:
      {
        "attack_queries": [...],
        "original_id": "...",
        "target_id": "..."
      }
    """
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_attack_from_pair(record, line_index: int):
    """
    Convert one JSONL record into the internal 'attack' structure expected by run_attack:
      {
        "queries": [
            {"id": "q1", "template": "...", "answer": ["Unknown"]},
            ...
        ],
        "original_id": "...",
        "target_id": "..."
      }
    """
    original_id = record["original_id"]
    target_id = record["target_id"]
    attack_queries = record["attack_queries"]

    queries = []
    for j, q_text in enumerate(attack_queries):
        queries.append(
            {
                "id": f"line{line_index}_q{j+1}",
                "template": q_text,
                "answer": ["Unknown"],
            }
        )

    return {
        "queries": queries,
        "original_id": original_id,
        "target_id": target_id,
    }


async def run_attack(
    attack,
    user_proxy,
    log_file_path,
    long_term_memory,
    num_shots=4,
    guard_agent=None,
    memory_defense=None,
):
    """
    Run attack with PERSISTENT memory across queries.
    
    Unlike main_attack_llama_pair1.py, this version does NOT clear memory before
    processing queries. Memory accumulates across all queries in the attack sequence,
    and persists across different attacks.
    
    Note: attack_pair_*.jsonl already contains the full poisoned prompts.
    """
    contents = attack["queries"]
    print(contents)
    print(f"[MEMORY] Starting attack with {len(long_term_memory)} items in memory")

    # NOTE: Memory is NOT cleared here - it persists from previous attacks/queries
    # long_term_memory = []  # <-- REMOVED: Memory persists across attacks

    for i in range(len(contents)):
        question = contents[i]["template"]
        answer = contents[i]["answer"]

        # INPUT MODERATION (if enabled)
        input_check = None
        if guard_agent is not None:
            await rate_limiter.wait()
            input_check = await guard_agent.check_input(question)
            if not input_check.get("is_safe", True):
                logger.warning(
                    f"[ATTACK RUN] Question flagged by input guard: {input_check.get('reason')}"
                )
                logger.warning(f"  Flags: {input_check.get('flags')}")
                if input_check.get("risk_level") == "high":
                    logger.warning("[ATTACK RUN] SKIPPING HIGH-RISK QUESTION")
                    continue

        try:
            print("-----------Question-----------")
            print(contents[i])
            print(question)
            user_proxy.update_memory(num_shots, long_term_memory)
            response, logs_string, _examples = await user_proxy.answer(question)
            print("-----------Response-----------")
            print(response)
            print("-----------End of Response-----------")
        except Exception:
            traceback.print_exc()
            logs_string = [traceback.format_exc()]

        print(logs_string)
        file_directory = log_file_path.format(id=contents[i]["id"])
        if isinstance(answer, list):
            answer = ", ".join(answer)
        logs_string.append("Ground-Truth Answer ---> " + answer)

        os.makedirs(os.path.dirname(file_directory), exist_ok=True)
        with open(file_directory, "w") as f:
            f.write("\n----------------------------------------------------------\n".join(logs_string))

        logs_joined = "\n----------------------------------------------------------\n".join(logs_string)
        if '"cell": "' in logs_joined:
            last_code_start = logs_joined.rfind('"cell": "')
            last_code_end = logs_joined.rfind('"\n}')
            last_code = logs_joined[last_code_start + 9 : last_code_end]
        else:
            last_code_end = logs_joined.rfind("Solution:")
            last_code = ""
        prediction_end = logs_joined.rfind("TERMINATE")
        prediction = logs_joined[last_code_end:prediction_end]

        # OUTPUT MODERATION (if enabled)
        should_append = True
        if memory_defense is not None and guard_agent is not None:
            await rate_limiter.wait()
            output_check = await guard_agent.check_output(
                question=question,
                generated_code=user_proxy.code,
                predicted_answer=prediction,
                answer=answer,
            )

            await rate_limiter.wait()
            should_append, defense_log = memory_defense.should_append_to_memory(
                question=question,
                knowledge=user_proxy.knowledge,
                code=user_proxy.code,
                predicted_answer=prediction,
                ground_truth_answer=answer,
                guard_output_check=output_check,
            )

            logs_joined += "\n===== DEFENSE SUMMARY =====\n"
            logs_joined += f"Input Check - Risk Level: {input_check.get('risk_level', 'N/A')}\n"
            logs_joined += (
                f"Output Check - Trust Score: {output_check.get('trust_score', 0):.2f}\n"
            )
            logs_joined += f"Memory Defense - Should Append: {should_append}\n"
            logs_joined += f"Memory Defense - Reasons: {defense_log.get('reasons', [])}\n"

        if should_append:
            new_item = {
                "question": question,
                "knowledge": user_proxy.knowledge,
                "code": user_proxy.code,
            }
            long_term_memory.append(new_item)
            logger.info(
                f"[ATTACK RUN] Appended item from {contents[i]['id']} to memory "
                f"(len={len(long_term_memory)})"
            )
        else:
            logger.warning(
                f"[ATTACK RUN] Rejected item from {contents[i]['id']} (not appended)"
            )


# ---------------------------------------------------------
# Main entry point for attack_pair_1.jsonl with PERSISTENT MEMORY
# ---------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(
        description="Run LLaMA-based attacks using attack_pair_*.jsonl with PERSISTENT memory across attacks"
    )
    parser.add_argument(
        "--attack_file",
        type=str,
        default="attack_dataset/attack_pair_1.jsonl",
        help="Path to attack_pair_*.jsonl file",
    )
    parser.add_argument("--dataset", type=str, default="mimic_iii")
    parser.add_argument("--logs_path", type=str, default="logs_llama_pair1_persistent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_shots", type=int, default=4)
    parser.add_argument(
        "--enable_defense", action="store_true", help="Enable input/output moderation"
    )
    parser.add_argument(
        "--enable_sandbox", action="store_true", help="Enable code execution sandbox"
    )
    parser.add_argument(
        "--llama_model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Hugging Face model id for LLaMA",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if args.dataset == "mimic_iii":
        from prompts_mimic import EHRAgent_4Shots_Knowledge
    else:
        from prompts_eicu import EHRAgent_4Shots_Knowledge

    # Instantiate LLaMA client
    model_client = TransformersChatClient(model_name=args.llama_model_name)

    # MedAgent using LLaMA backend
    user_proxy = MedAgent(
        name="user_proxy",
        model_client=model_client,
        dataset=args.dataset,
        system_message="You are a helpful EHR coding agent.",
    )

    # Initialize defense systems if enabled
    guard_agent = None
    memory_defense = None
    if args.enable_defense:
        logger.info("Input/Output Moderation enabled")
        guard_agent = GuardAgent(model_client)
        memory_defense = MemoryDefense(enable_sandbox=args.enable_sandbox)

    # Load attack pairs from JSONL
    records = load_attack_pairs(args.attack_file)
    logger.info(f"Loaded {len(records)} attack lines from {args.attack_file}")

    start_time = time.time()

    # Build initial long-term memory from few-shot template
    # This initial memory will persist across ALL attacks
    long_term_memory = []
    init_memory = EHRAgent_4Shots_Knowledge.split("\n\n")
    for item in init_memory:
        item = item.split("Question:")[-1]
        question = item.split("\nKnowledge:\n")[0]
        item_rest = item.split("\nKnowledge:\n")[-1]
        knowledge = item_rest.split("\nSolution:")[0]
        code = item_rest.split("\nSolution:")[-1]
        new_item = {"question": question, "knowledge": knowledge, "code": code}
        long_term_memory.append(new_item)
    
    logger.info(f"[INITIAL MEMORY] Loaded {len(long_term_memory)} few-shot examples")

    # Run attacks for each line in the JSONL file
    # NOTE: Memory is NOT cleared between attacks - it accumulates across all attacks
    for idx, rec in enumerate(records):
        attack_obj = build_attack_from_pair(rec, line_index=idx + 1)
        attack_id = f"pair_line_{idx+1}"
        print(f"===========Starting Attack Line: {attack_id}===========")
        print(f"[MEMORY] Current memory size: {len(long_term_memory)} items")
        log_file_dir = f"{args.logs_path}/{attack_id}/"
        os.makedirs(os.path.dirname(log_file_dir), exist_ok=True)
        per_query_log_path = log_file_dir + "{id}.txt"
        # NOTE: Memory is NOT cleared here - it persists from previous attacks
        # long_term_memory = []  # <-- REMOVED: Memory persists across attacks

        await run_attack(
            attack=attack_obj,
            user_proxy=user_proxy,
            log_file_path=per_query_log_path,
            long_term_memory=long_term_memory,
            num_shots=args.num_shots,
            guard_agent=guard_agent,
            memory_defense=memory_defense,
        )
        print(f"===========Finished Attack Line: {attack_id}===========")
        print(f"[MEMORY] Memory size after attack: {len(long_term_memory)} items")

    end_time = time.time()
    print("Time elapsed: ", end_time - start_time)
    print(f"[FINAL MEMORY] Total memory items accumulated: {len(long_term_memory)}")

    # Save defense audit log if enabled
    if memory_defense:
        audit_path = f"{args.logs_path}/defense_audit_log.json"
        os.makedirs(os.path.dirname(audit_path), exist_ok=True)
        memory_defense.save_audit_log(audit_path)

        summary = memory_defense.report_summary()
        print("\n===== DEFENSE SUMMARY REPORT =====")
        print(f"Total decisions: {summary['total_decisions']}")
        print(f"Appended to memory: {summary['appended_to_memory']}")
        print(f"Rejected: {summary['rejected']}")
        print(f"Rejection rate: {summary['rejection_rate']:.2%}")
        print(f"Avg trust score: {summary['avg_trust_score']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())

