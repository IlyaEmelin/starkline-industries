import json
import sys
from typing import Any

from ai_client import call_model
from tools import TOOL_DEFINITIONS, ToolError, call_tool


MAX_STEPS = 6


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_configure_stdio()


def _get_messages(
    system_prompt: str,
    conversation_history: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if conversation_history is None:
        return [{"role": "system", "content": system_prompt}]

    if not conversation_history:
        conversation_history.append({"role": "system", "content": system_prompt})
    elif conversation_history[0].get("role") != "system":
        conversation_history.insert(0, {"role": "system", "content": system_prompt})

    return conversation_history


def _tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }


def _assistant_message_to_dict(message: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": "assistant",
        "content": message.content,
    }

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = [_tool_call_to_dict(tool_call) for tool_call in tool_calls]

    return result


def _parse_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        raise ToolError("Модель вернула аргументы tool не в JSON.") from exc

    if not isinstance(parsed, dict):
        raise ToolError("Аргументы tool должны быть JSON-объектом.")

    return parsed


def _safe_arguments_for_log(arguments: dict[str, Any]) -> str:
    safe: dict[str, Any] = {}
    for key, value in arguments.items():
        if key == "content" and isinstance(value, str):
            safe[key] = f"<content {len(value.encode('utf-8'))} bytes>"
        else:
            safe[key] = value
    return json.dumps(safe, ensure_ascii=True)


def run_agent(
    user_task: str,
    system_prompt: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> str:
    print(f"[task] {user_task}")

    messages = _get_messages(system_prompt, conversation_history)
    messages.append({"role": "user", "content": user_task})

    for step_number in range(1, MAX_STEPS + 1):
        print(f"[loop] step {step_number}")

        assistant_message = call_model(messages, tools=TOOL_DEFINITIONS)
        tool_calls = getattr(assistant_message, "tool_calls", None)

        if not tool_calls:
            print("[final] model returned final answer")
            answer = assistant_message.content or ""
            messages.append({"role": "assistant", "content": answer})
            return answer

        messages.append(_assistant_message_to_dict(assistant_message))

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            print(f"[tool] selected: {tool_name}")

            try:
                arguments = _parse_arguments(tool_call.function.arguments)
                print(f"[tool] arguments: {_safe_arguments_for_log(arguments)}")

                result = call_tool(tool_name, arguments)
                tool_payload = {
                    "ok": True,
                    "result": result,
                }

                if tool_name == "read_lead_file":
                    path = result.get("path") if isinstance(result, dict) else ""
                    print(f"[tool] read ok: {path}")
                elif tool_name == "write_lead_review":
                    path = result.get("path") if isinstance(result, dict) else ""
                    print(f"[tool] write ok: {path}")
            except Exception as exc:
                tool_payload = {
                    "ok": False,
                    "error": str(exc),
                }
                print(f"[refusal] {exc}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_payload, ensure_ascii=False),
                }
            )

    answer = "Агент остановился: достигнут лимит шагов runtime."
    messages.append({"role": "assistant", "content": answer})
    return answer