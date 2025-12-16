"""
EhrAgent-compatible MedAgent using autogen_agentchat.
Based on https://github.com/wshi83/EhrAgent
"""

import json
import time
import logging
import asyncio
from typing import Dict, List

import Levenshtein
from termcolor import colored

from autogen_core.tools import FunctionTool
from autogen_core.models import SystemMessage, UserMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import (
    TextMessage,
    ToolCallRequestEvent,
    ToolCallExecutionEvent,
)
from autogen_core import CancellationToken


from toolset_high import run_code

logger = logging.getLogger(__name__)


class AsyncRateLimiter:
    """Rate limiter for API calls to avoid 429 errors."""
    def __init__(self, calls_per_second=0.1):  # 10 seconds between calls
        self.interval = 1.0 / calls_per_second
        self.last_call = 0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                sleep_time = self.interval - elapsed
                logger.debug(f"[MEDAGENT RATE LIMITER] Sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
            self.last_call = time.time()


async def retry_with_backoff(func, max_retries=5, base_delay=5.0):
    """Retry a function with exponential backoff for 429 errors."""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "rate limit" in error_str or "quota" in error_str
            
            if attempt == max_retries - 1:
                logger.error(f"[RETRY] Max retries ({max_retries}) exceeded. Raising error.")
                raise
            
            if is_rate_limit:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"[RETRY] Rate limit error (attempt {attempt + 1}/{max_retries}). Waiting {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                delay = base_delay * (attempt + 1)
                logger.warning(f"[RETRY] Error (attempt {attempt + 1}/{max_retries}): {e}. Waiting {delay:.1f}s...")
                await asyncio.sleep(delay)
    
    raise Exception("Retry logic failed unexpectedly")


_rate_limiter = AsyncRateLimiter(calls_per_second=0.05)

class MedAgent:
    def __init__(
        self,
        name: str,
        model_client,
        system_message: str = "",
        dataset: str = "mimic_iii",
    ):
        self.name = name
        self.model_client = model_client
        self.dataset = dataset

        # Few-shot memory stuff
        self.num_shots = 0
        self.memory: List[Dict[str, str]] = []

        # Last question/code/knowledge for debugging
        self.question: str = ""
        self.code: str = ""
        self.knowledge: str = ""

        # Build tools and the underlying AssistantAgent
        self._build_agent(system_message or "You are a helpful medical coding assistant.")


    def update_memory(self, num_shots: int, memory: List[Dict[str, str]]):
        """
        memory: list of dicts with keys: question, knowledge, code
        """
        self.num_shots = num_shots
        self.memory = memory


    def register_dataset(self, dataset: str):
        self.dataset = dataset


    def _build_agent(self, system_message: str):
        async def execute_cell(cell: str) -> str:
            """Execute code and call error debugger on failure."""
            self.code = cell
            try:
                raise RuntimeError("Demo error from execute_cell – plug in your executor.")
            except Exception as e:
                error_str = f"Error: {e}"
                reasons = await self.error_debugger(self.code, error_str)
                return error_str + "\nPotential Reasons: " + reasons

        execute_tool = FunctionTool(
            execute_cell,
            description="Execute user-provided code and, on failure, explain the most likely cause.",
            name="execute_cell",
        )

        code_executor_tool = FunctionTool(
            run_code,
            name="python",
            description="Tool for function run_code",
        )

        self.assistant = AssistantAgent(
            name=self.name,
            model_client=self.model_client,
            system_message=system_message,
            tools=[execute_tool, code_executor_tool],
            reflect_on_tool_use=True,
        )

    async def retrieve_knowledge(self, query: str) -> str:
        """Retrieve knowledge using model client."""
        if self.dataset == "mimic_iii":
            from prompts_mimic import RetrKnowledge
        else:
            from prompts_eicu import RetrKnowledge

        query_message = RetrKnowledge.format(question=query)

        messages = [
            SystemMessage(content="You are an AI assistant that helps people find information."),
            UserMessage(content=query_message, source="user"),
        ]

        await _rate_limiter.wait()

        async def _make_request():
            resp = await self.model_client.create(messages)
            if isinstance(resp.content, str) and resp.content.strip():
                return resp.content.strip()
            return "Fail to retrieve related knowledge, please try again later."

        try:
            result = await retry_with_backoff(_make_request, max_retries=5, base_delay=3.0)
            return result
        except Exception as e:
            logger.error(f"retrieve_knowledge failed after retries: {e}")
            return "Fail to retrieve related knowledge, please try again later."


    def retrieve_examples(self, query: str) -> str:
        """Retrieve high-trust examples for few-shot learning."""
        if not self.memory or self.num_shots <= 0:
            return ""

        now = time.time()
        scored: List[tuple[int, float, float]] = []

        for idx, m in enumerate(self.memory):
            question = m.get("question", "")
            dist = Levenshtein.distance(query, question)
            effective_trust = self._compute_effective_trust(m, now=now)

            stored_question = m.get("question", "").lower()
            has_poison_pattern = (
                "refer" in stored_question and "to" in stored_question
            ) or "do not use" in stored_question or "use patient" in stored_question and "instead" in stored_question
            
            if has_poison_pattern:
                logger.warning(f"[RETRIEVE] BLOCKING memory {idx} - contains poison pattern (trust={effective_trust:.2f})")
                continue

            if effective_trust < 0.5:
                logger.debug(f"[RETRIEVE] Skipping memory {idx} due to low trust ({effective_trust:.2f})")
                continue

            score = dist / (effective_trust + 1e-3)
            scored.append((idx, score, effective_trust))

        if not scored:
            logger.warning("[RETRIEVE] No trusted examples found after filtering")
            return ""

        scored.sort(key=lambda x: x[1])
        selected_indexes = [idx for idx, _, _ in scored[: self.num_shots]]

        examples: List[str] = []
        for i in selected_indexes:
            m = self.memory[i]
            template = "Question: {}\nKnowledge:\n{}\nSolution:\n{}\n".format(
                m.get("question", ""),
                m.get("knowledge", ""),
                m.get("code", ""),
            )
            examples.append(template)

        logger.debug(f"[RETRIEVE] Selected {len(examples)} examples from {len(scored)} trusted candidates")
        return "\n".join(examples)

    def _compute_effective_trust(self, memory_item: Dict, now: float = None, half_life_hours: float = 24.0) -> float:
        """Compute effective trust score with temporal decay."""
        if now is None:
            now = time.time()

        base_trust = memory_item.get("trust_score", 0.8)
        try:
            base_trust = float(base_trust)
        except (TypeError, ValueError):
            base_trust = 0.8

        base_trust = max(0.0, min(1.0, base_trust))

        ts = memory_item.get("last_updated_at") or memory_item.get("created_at")
        if ts is None:
            return base_trust

        age_hours = (now - ts) / 3600.0
        
        if half_life_hours <= 0:
            return base_trust

        decay_factor = 0.5 ** (age_hours / half_life_hours)
        effective_trust = base_trust * decay_factor

        return max(0.0, min(1.0, effective_trust))


    async def generate_init_message(self, message: str) -> str:
        """Create prompt with RAG and examples."""
        if self.dataset == "mimic_iii":
            from prompts_mimic import EHRAgent_Message_Prompt
        else:
            from prompts_eicu import EHRAgent_Message_Prompt

        self.question = message

        knowledge = await self.retrieve_knowledge(message)
        self.knowledge = knowledge
        print("Knowledge", len(knowledge))

        examples = self.retrieve_examples(message)
        print("Examples", len(examples))

        init_message = EHRAgent_Message_Prompt.format(
            examples=examples, knowledge=knowledge, question=message
        )
        return init_message, examples
    

    async def error_debugger(self, code: str, error_info: str) -> str:
        if self.dataset == "mimic_iii":
            from prompts_mimic import CodeDebugger
        else:
            from prompts_eicu import CodeDebugger

        query_message = CodeDebugger.format(
            question=self.question, code=code, error_info=error_info
        )

        messages = [
            SystemMessage(
                content=(
                    "You are an AI assistant that helps people debug their code. "
                    "Only list one most possible reason to the errors."
                )
            ),
            UserMessage(content=query_message, source="user"),
        ]

        await _rate_limiter.wait()

        async def _make_request():
            resp = await self.model_client.create(messages)
            if isinstance(resp.content, str) and resp.content.strip():
                return resp.content.strip()
            return "Fail to diagnose the reasons to the errors."

        try:
            result = await retry_with_backoff(_make_request, max_retries=5, base_delay=3.0)
            return result
        except Exception as e:
            logger.error(f"error_debugger failed after retries: {e}")
            return "Fail to diagnose the reasons to the errors."

    async def answer(self, user_question: str) -> str:
        """Build init message with RAG, invoke assistant, return response."""
        init_msg, examples = await self.generate_init_message(user_question)
        print("-----------Initial Message to AssistantAgent-----------")
        print(init_msg)

        await _rate_limiter.wait()

        async def _call_assistant():
            return await self.assistant.on_messages(
                [TextMessage(content=init_msg, source="user")],
                cancellation_token=CancellationToken(),
            )

        response = await retry_with_backoff(_call_assistant, max_retries=5, base_delay=3.0)

        final_msg = response.chat_message
        answer = str(final_msg.content)

        logs_string: list[str] = []
        logs_string.append(str(user_question))
        logs_string.append(init_msg)
        logs_string.append(answer)

        for m in response.inner_messages:
            if isinstance(m, TextMessage):
                if m.content is not None:
                    logs_string.append(str(m.content))
            elif isinstance(m, ToolCallRequestEvent):
                for fcall in m.content:
                    args_raw = fcall.arguments
                    try:
                        args = json.loads(args_raw)
                    except Exception:
                        logs_string.append(str(args_raw))
                        continue

                    if isinstance(args, dict) and "cell" in args:
                        logs_string.append(str(args["cell"]))
                    else:
                        logs_string.append(str(args))
            elif isinstance(m, ToolCallExecutionEvent):
                for result in m.content:
                    if result.content is not None:
                        logs_string.append(str(result.content))

        return str(final_msg.content), logs_string, examples