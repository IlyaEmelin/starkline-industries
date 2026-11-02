import sys
from pathlib import Path
from typing import Any

from agent_runtime import run_agent


PROJECT_ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "agent" / "soul.md"


def configure_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def main() -> None:
    configure_stdio()
    system_prompt = load_system_prompt()
    session_messages: list[dict[str, Any]] = []

    print("LeadReview Agent — Пятница")
    print("Локальный терминальный агент Starkline Industries.")
    print("Введите задачу или exit для выхода.")

    while True:
        try:
            user_task = input("\nfriday> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if user_task.lower() in {"exit", "quit"}:
            print("bye")
            break

        if not user_task:
            continue

        answer = run_agent(
            user_task=user_task,
            system_prompt=system_prompt,
            conversation_history=session_messages,
        )
        print("\n[answer]")
        print(answer)


if __name__ == "__main__":
    main()