from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable

AGENT_BASE_URL = os.getenv("VAP_LLM_BASE_URL", "https://llm-api.amd.com/OpenAI")
AGENT_MODEL = os.getenv("VAP_LLM_MODEL", "gpt-5.5")
AGENT_ENV_KEY_NAME = "VAP_LLM_SUBSCRIPTION_KEY"
AGENT_TIMEOUT_SEC = float(os.getenv("VAP_LLM_TIMEOUT_SEC", "180"))
AGENT_MAX_TOOL_ROUNDS = int(os.getenv("VAP_AGENT_MAX_TOOL_ROUNDS", "8"))


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    parameters: dict[str, Any]
    safety: str
    handler: ToolHandler

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"[{self.safety}] {self.description}",
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class PendingAction:
    approval_id: str
    tool_name: str
    arguments: dict[str, Any]


class VAPAgentRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscription_key: str | None = None
        self._tools: dict[str, AgentTool] = {}
        self._pending_actions: dict[str, PendingAction] = {}

    def register_tool(self, tool: AgentTool) -> None:
        self._tools[tool.name] = tool

    def get_subscription_key(self) -> tuple[str | None, str | None]:
        env_key = os.getenv(AGENT_ENV_KEY_NAME)
        if env_key:
            return env_key, "env"
        with self._lock:
            memory_key = self._subscription_key
        if memory_key:
            return memory_key, "memory"
        return None, None

    def status(self) -> dict[str, Any]:
        _, key_source = self.get_subscription_key()
        return {
            "unlocked": key_source is not None,
            "key_source": key_source,
            "base_url": AGENT_BASE_URL,
            "model": AGENT_MODEL,
            "env_key_name": AGENT_ENV_KEY_NAME,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "safety": tool.safety,
                }
                for tool in self._tools.values()
            ],
        }

    def unlock(self, subscription_key: str) -> dict[str, Any]:
        cleaned_key = subscription_key.strip()
        self.validate_key(cleaned_key)
        with self._lock:
            self._subscription_key = cleaned_key
        return {
            **self.status(),
            "message": "Agent unlocked for this server session.",
        }

    def validate_key(self, subscription_key: str) -> None:
        if not subscription_key.strip():
            raise ValueError("Subscription key is required")
        client = self._create_client(subscription_key.strip())
        try:
            client.chat.completions.create(
                model=AGENT_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": "Reply with OK to validate this VAP agent connection.",
                    }
                ],
                max_completion_tokens=4,
            )
        except Exception as exc:
            raise ValueError(f"Agent key validation failed: {exc}") from exc

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        subscription_key, key_source = self.get_subscription_key()
        if not subscription_key:
            raise ValueError("Agent is locked. Provide a subscription key first.")

        messages = self.normalize_messages(payload.get("messages"))
        max_completion_tokens = self._parse_max_tokens(
            payload.get("max_completion_tokens", 700)
        )
        client = self._create_client(subscription_key)
        loop_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            *messages,
        ]

        tool_events: list[dict[str, Any]] = []
        for _ in range(AGENT_MAX_TOOL_ROUNDS):
            response = self._chat_completion(
                client,
                messages=loop_messages,
                tools=[tool.to_openai_tool() for tool in self._tools.values()],
                max_completion_tokens=max_completion_tokens,
            )
            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            if not tool_calls:
                return {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": message.content or "",
                    },
                    "tool_events": tool_events,
                    "model": AGENT_MODEL,
                    "key_source": key_source,
                }

            loop_messages.append(self._assistant_tool_call_message(message))
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments = self._parse_tool_arguments(tool_call.function.arguments)
                tool = self._tools.get(tool_name)
                if tool is None:
                    result = {"ok": False, "message": f"Unknown tool: {tool_name}"}
                elif tool.safety == "requires_approval":
                    approval = self._create_pending_action(tool_name, arguments)
                    return {
                        "type": "approval_required",
                        "message": {
                            "role": "assistant",
                            "content": f"Approval required before running tool `{tool_name}`.",
                        },
                        "approval": {
                            "approval_id": approval.approval_id,
                            "tool_name": approval.tool_name,
                            "arguments": approval.arguments,
                        },
                        "tool_events": tool_events,
                        "model": AGENT_MODEL,
                        "key_source": key_source,
                    }
                else:
                    result = self._execute_tool(tool, arguments)
                    tool_events.append(
                        {
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "result": result,
                        }
                    )
                loop_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        return {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": (
                    f"I reached the tool-call limit ({AGENT_MAX_TOOL_ROUNDS}) while working on this request. "
                    "Try asking for a narrower analysis, or increase VAP_AGENT_MAX_TOOL_ROUNDS."
                ),
            },
            "tool_events": tool_events,
            "model": AGENT_MODEL,
            "key_source": key_source,
        }

    def stream_chat(self, payload: dict[str, Any]):
        subscription_key, key_source = self.get_subscription_key()
        if not subscription_key:
            raise ValueError("Agent is locked. Provide a subscription key first.")

        messages = self.normalize_messages(payload.get("messages"))
        max_completion_tokens = self._parse_max_tokens(
            payload.get("max_completion_tokens", 700)
        )
        client = self._create_client(subscription_key)
        loop_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            *messages,
        ]

        for _ in range(AGENT_MAX_TOOL_ROUNDS):
            stream = self._chat_completion(
                client,
                messages=loop_messages,
                tools=[tool.to_openai_tool() for tool in self._tools.values()],
                max_completion_tokens=max_completion_tokens,
                stream=True,
            )
            content_parts: list[str] = []
            tool_calls_by_index: dict[int, dict[str, Any]] = {}

            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = choices[0].delta
                content_delta = getattr(delta, "content", None)
                if content_delta:
                    content_parts.append(content_delta)
                    yield {"type": "delta", "content": content_delta}
                for tool_call in getattr(delta, "tool_calls", None) or []:
                    index = int(getattr(tool_call, "index", 0) or 0)
                    record = tool_calls_by_index.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if getattr(tool_call, "id", None):
                        record["id"] = tool_call.id
                    function_delta = getattr(tool_call, "function", None)
                    if function_delta is not None:
                        if getattr(function_delta, "name", None):
                            record["function"]["name"] += function_delta.name
                        if getattr(function_delta, "arguments", None):
                            record["function"]["arguments"] += function_delta.arguments

            tool_calls = [
                tool_calls_by_index[index] for index in sorted(tool_calls_by_index)
            ]
            if not tool_calls:
                yield {
                    "type": "done",
                    "message": {"role": "assistant", "content": "".join(content_parts)},
                    "model": AGENT_MODEL,
                    "key_source": key_source,
                }
                return

            loop_messages.append(
                {
                    "role": "assistant",
                    "content": "".join(content_parts) or None,
                    "tool_calls": tool_calls,
                }
            )
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                arguments = self._parse_tool_arguments(
                    tool_call["function"]["arguments"]
                )
                tool = self._tools.get(tool_name)
                if tool is None:
                    result = {"ok": False, "message": f"Unknown tool: {tool_name}"}
                elif tool.safety == "requires_approval":
                    approval = self._create_pending_action(tool_name, arguments)
                    yield {
                        "type": "approval_required",
                        "message": {
                            "role": "assistant",
                            "content": f"Approval required before running tool `{tool_name}`.",
                        },
                        "approval": {
                            "approval_id": approval.approval_id,
                            "tool_name": approval.tool_name,
                            "arguments": approval.arguments,
                        },
                        "model": AGENT_MODEL,
                        "key_source": key_source,
                    }
                    return
                else:
                    result = self._execute_tool(tool, arguments)
                    yield {
                        "type": "tool_event",
                        "tool_event": {
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "result": result,
                        },
                    }
                loop_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        yield {
            "type": "done",
            "message": {
                "role": "assistant",
                "content": (
                    f"I reached the tool-call limit ({AGENT_MAX_TOOL_ROUNDS}) while working on this request. "
                    "Try asking for a narrower analysis, or increase VAP_AGENT_MAX_TOOL_ROUNDS."
                ),
            },
            "model": AGENT_MODEL,
            "key_source": key_source,
        }

    def approve(self, approval_id: str) -> dict[str, Any]:
        with self._lock:
            action = self._pending_actions.pop(approval_id, None)
        if action is None:
            raise ValueError(
                "Approval request was not found or has already been handled."
            )
        tool = self._tools.get(action.tool_name)
        if tool is None:
            raise ValueError(f"Tool no longer exists: {action.tool_name}")
        result = self._execute_tool(tool, action.arguments)
        return {
            "type": "action_result",
            "message": {
                "role": "assistant",
                "content": f"Approved action `{action.tool_name}` executed.",
            },
            "tool_event": {
                "tool_name": action.tool_name,
                "arguments": action.arguments,
                "result": result,
            },
        }

    def cancel(self, approval_id: str) -> dict[str, Any]:
        with self._lock:
            action = self._pending_actions.pop(approval_id, None)
        if action is None:
            raise ValueError(
                "Approval request was not found or has already been handled."
            )
        return {
            "type": "action_cancelled",
            "message": {
                "role": "assistant",
                "content": f"Cancelled pending action `{action.tool_name}`.",
            },
        }

    def normalize_messages(self, raw_messages: Any) -> list[dict[str, str]]:
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("messages must be a non-empty list")
        normalized: list[dict[str, str]] = []
        allowed_roles = {"system", "user", "assistant"}
        for index, item in enumerate(raw_messages):
            if not isinstance(item, dict):
                raise ValueError(f"messages[{index}] must be an object")
            role = item.get("role")
            content = item.get("content")
            if role not in allowed_roles:
                raise ValueError(
                    f"messages[{index}].role must be system, user, or assistant"
                )
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"messages[{index}].content is required")
            normalized.append({"role": role, "content": content.strip()})
        return normalized

    def _create_client(self, subscription_key: str):
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI SDK is not installed. Run install.sh again."
            ) from exc

        return openai.OpenAI(
            base_url=AGENT_BASE_URL,
            api_key="dummy",
            timeout=AGENT_TIMEOUT_SEC,
            default_headers={
                "Ocp-Apim-Subscription-Key": subscription_key,
                "user": self._user_header(),
            },
        )

    def _chat_completion(
        self,
        client: Any,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_completion_tokens: int,
        stream: bool = False,
    ) -> Any:
        try:
            kwargs: dict[str, Any] = {
                "model": AGENT_MODEL,
                "messages": messages,
                "max_completion_tokens": max_completion_tokens,
                "stream": stream,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if "timed out" in str(exc).lower():
                raise TimeoutError(
                    f"LLM request exceeded {AGENT_TIMEOUT_SEC:.0f}s. Try again, ask a narrower question, or increase VAP_LLM_TIMEOUT_SEC."
                ) from exc
            raise

    def _create_pending_action(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> PendingAction:
        action = PendingAction(
            approval_id=uuid.uuid4().hex,
            tool_name=tool_name,
            arguments=arguments,
        )
        with self._lock:
            self._pending_actions[action.approval_id] = action
        return action

    def _execute_tool(
        self, tool: AgentTool, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            return {"ok": True, "data": tool.handler(arguments)}
        except Exception as exc:
            return {"ok": False, "message": str(exc)}

    def _assistant_tool_call_message(self, message: Any) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls or []
            ],
        }

    def _parse_tool_arguments(self, raw_arguments: str | None) -> dict[str, Any]:
        if not raw_arguments:
            return {}
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tool arguments are not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must be a JSON object")
        return parsed

    def _parse_max_tokens(self, value: Any) -> int:
        if not isinstance(value, int) or value < 1 or value > 8192:
            raise ValueError("max_completion_tokens must be an integer from 1 to 8192")
        return value

    def _system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            f"- {tool.name} [{tool.safety}]: {tool.description}"
            for tool in self._tools.values()
        )
        return (
            "You are the VAP Profiling Agent, a Hermes-style tool agent dedicated "
            "to vLLM profiling workflows. Your job is to guide the user step by "
            "step from profiling intent to a validated run. Start by identifying "
            "the model they want to profile, then refine model path, Docker image, "
            "GPU devices, tensor parallel size, benchmark prompt/concurrency "
            "settings, profiler options, and visualization needs. Prefer asking "
            "one focused question at a time when required details are missing. "
            "Use tools to inspect the current config, status, logs, validation, "
            "port checks, and resource checks before recommending execution. "
            "When the config is ready, summarize the final run plan and request "
            "approval for tools marked requires_approval. Never execute run or "
            "stop actions without explicit user approval. Explain profiling risks "
            "and tradeoffs clearly. When the user asks to download run logs or "
            "trace artifacts, use the safe download artifact tool instead of "
            "inventing file paths. For detailed trace analysis, prefer Perfetto "
            "SQL tools over raw trace previews. Prefer the TorchProfilerTraceSkill "
            "workflow tool for trace reports. For broad trace analysis, call "
            "run_torchprofiler_skill once with workflow=full_report instead of "
            "issuing many individual SQL tools. Use individual Perfetto SQL "
            "queries only when the user asks for deeper evidence.\n\nAvailable VAP tools:\n"
            f"{tool_descriptions}"
        )

    def _user_header(self) -> str:
        try:
            return os.getlogin()
        except OSError:
            return os.getenv("USER") or os.getenv("USERNAME") or "unknown"
