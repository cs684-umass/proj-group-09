"""
Note: this code is to recreate the logic used in EhrAgent/ehragent/medagent.py.
https://github.com/wshi83/EhrAgent
But modified to use autogen_agentchat (AssistantAgent and FunctionTool) which is the newer version.
Additionly, this is for Gemini models, not OpenAI models.
"""

import json
import time
import logging
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
        # This is an example on how to attach a tool to the assistant
        # Tool to execute code + run error debugger on failure.
        async def execute_cell(cell: str) -> str:
            """
            Execute a Python cell (or any code) and, if there is an error,
            call the LLM-based error debugger to explain the most likely reason.
            """
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

        # This is the same as register_function, as the authors code.
        code_executor_tool = FunctionTool(
            run_code,
            name="python",
            description="Tool for function run_code",
        )

        # Create AssistantAgent that will call tools and generate code.
        self.assistant = AssistantAgent(
            name=self.name,
            model_client=self.model_client,
            system_message=system_message,
            tools=[execute_tool, code_executor_tool],
            reflect_on_tool_use=True,
        )

    """
    RAG - function to retrieve knowledge and examples
    """
    async def retrieve_knowledge(self, query: str) -> str:
        """
        Invoke Gemini model_client with retrieval prompt template.
        Cannot use the original logic to the authors, as that uses OpenAI: OpenAIChatCompletionClient.
        """
        if self.dataset == "mimic_iii":
            from prompts_mimic import RetrKnowledge
        else:
            from prompts_eicu import RetrKnowledge

        query_message = RetrKnowledge.format(question=query)

        messages = [
            SystemMessage(content="You are an AI assistant that helps people find information."),
            UserMessage(content=query_message, source="user"),
        ]

        patience = 2
        sleep_time = 3

        while patience > 0:
            patience -= 1
            try:
                resp = await self.model_client.create(messages)
                if isinstance(resp.content, str) and resp.content.strip():
                    return resp.content.strip()
            except Exception as e:
                logger.warning(f"retrieve_knowledge error: {e}")
                if sleep_time > 0:
                    time.sleep(sleep_time)

        return "Fail to retrieve related knowledge, please try again later."


    def retrieve_examples(self, query: str) -> str:
        if not self.memory or self.num_shots <= 0:
            return ""

        levenshtein_dist = {}
        for i in range(len(self.memory)):
            question = self.memory[i]["question"]
            levenshtein_dist[i] = Levenshtein.distance(query, question)

        levenshtein_dist = sorted(levenshtein_dist.items(), key=lambda x: x[1])
        selected_indexes = [
            levenshtein_dist[i][0] for i in range(min(self.num_shots, len(levenshtein_dist)))
        ]

        examples = []
        for i in selected_indexes:
            m = self.memory[i]
            template = "Question: {}\nKnowledge:\n{}\nSolution:\n{}\n".format(
                m["question"], m["knowledge"], m["code"]
            )
            examples.append(template)

        return "\n".join(examples)


    async def generate_init_message(self, message: str) -> str:
        """
        Create prompt with RAG and examples - prepended to question
        """
        if self.dataset == "mimic_iii":
            from prompts_mimic import EHRAgent_Message_Prompt
        else:
            from prompts_eicu import EHRAgent_Message_Prompt

        self.question = message

        knowledge = await self.retrieve_knowledge(message)
        self.knowledge = knowledge

        examples = self.retrieve_examples(message)

        init_message = EHRAgent_Message_Prompt.format(
            examples=examples, knowledge=knowledge, question=message
        )
        return init_message
    

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

        patience = 2
        sleep_time = 3

        while patience > 0:
            patience -= 1
            try:
                resp = await self.model_client.create(messages)
                if isinstance(resp.content, str) and resp.content.strip():
                    return resp.content.strip()
            except Exception as e:
                logger.warning(f"error_debugger error: {e}")
                if sleep_time > 0:
                    time.sleep(sleep_time)

        return "Fail to diagnose the reasons to the errors."
    

    async def answer(self, user_question: str) -> str:
        """
        Full flow:
        1. Build init message with RAG with examples.
        2. Invoke AssistantAgent to get response
        3. Return the final message text.
        """
        init_msg = await self.generate_init_message(user_question)

        # Here we let the AssistantAgent do its thing (may call tools).
        # result = await self.assistant.run(task=init_msg)
        response = await self.assistant.on_messages(
            [TextMessage(content=init_msg, source="user")],
            cancellation_token=CancellationToken(),
        )


        # `result` is an AgentChat Response. The final message is in `chat_message`.
        final_msg = response.chat_message
        answer = str(final_msg.content)

        # Logging logic
        logs_string: list[str] = []
        logs_string.append(str(user_question))
        logs_string.append(answer)

        for m in response.inner_messages:
            if isinstance(m, TextMessage):
                if m.content is not None:
                    logs_string.append(str(m.content))

            # Tool calls (function calls)
            elif isinstance(m, ToolCallRequestEvent):
                # m.content is a list of FunctionCall
                for fcall in m.content:
                    args_raw = fcall.arguments
                    try:
                        args = json.loads(args_raw)
                    except Exception:
                        # if it isn't valid JSON, just log the raw string
                        logs_string.append(str(args_raw))
                        continue

                    if isinstance(args, dict) and "cell" in args:
                        logs_string.append(str(args["cell"]))
                    else:
                        logs_string.append(str(args))

            # Tool execution results (outputs, errors, etc.)
            elif isinstance(m, ToolCallExecutionEvent):
                # m.content is a list of FunctionExecutionResult
                for result in m.content:
                    if result.content is not None:
                        logs_string.append(str(result.content))

        return str(final_msg.content), logs_string