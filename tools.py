from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
LEADS_DIR = (PROJECT_ROOT / "leads").resolve()
REVIEWS_DIR = (PROJECT_ROOT / "reviews").resolve()


class ToolError(Exception):
    """Ошибка безопасного tool-call."""


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_lead_file",
            "description": (
                "Безопасно читает Markdown-заявку только из папки leads/. "
                "filename должен быть относительным: lead_001.md или leads/lead_001.md."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Относительное имя .md-файла заявки внутри leads/.",
                    }
                },
                "required": ["filename"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_lead_review",
            "description": (
                "Безопасно создаёт или перезаписывает Markdown-review только в папке reviews/. "
                "filename должен быть относительным: lead_001_review.md или reviews/lead_001_review.md."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Относительное имя .md-файла review внутри reviews/.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Markdown-содержимое review.",
                    },
                },
                "required": ["filename", "content"],
                "additionalProperties": False,
            },
        },
    },
]


def _validate_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ToolError(f"`{field_name}` должен быть строкой.")

    stripped = value.strip()
    if not stripped:
        raise ToolError(f"`{field_name}` не должен быть пустым.")

    return stripped


def _normalize_relative_markdown_path(raw_path: str, allowed_prefix: str, base_dir: Path) -> Path:
    normalized = raw_path.replace("\\", "/")
    posix_path = PurePosixPath(normalized)

    if posix_path.is_absolute() or ":" in normalized:
        raise ToolError("Абсолютные пути запрещены.")

    parts = [part for part in posix_path.parts if part not in {"", "."}]
    if not parts:
        raise ToolError("Путь не должен быть пустым.")

    if ".." in parts:
        raise ToolError("Выход через `..` запрещён.")

    if parts[0] == allowed_prefix:
        parts = parts[1:]

    if not parts:
        raise ToolError(f"Нужно указать файл внутри `{allowed_prefix}/`.")

    relative_path = Path(*parts)
    if relative_path.suffix.lower() != ".md":
        raise ToolError("Разрешены только Markdown-файлы с расширением `.md`.")

    candidate = (base_dir / relative_path).resolve()
    try:
        candidate.relative_to(base_dir)
    except ValueError as exc:
        raise ToolError(f"Путь должен оставаться внутри `{allowed_prefix}/`.") from exc

    return candidate


def _display_path(prefix: str, resolved_path: Path, base_dir: Path) -> str:
    relative = resolved_path.relative_to(base_dir).as_posix()
    return f"{prefix}/{relative}"


def read_lead_file(filename: str) -> dict[str, str]:
    safe_filename = _validate_string(filename, "filename")
    file_path = _normalize_relative_markdown_path(safe_filename, "leads", LEADS_DIR)

    if not file_path.exists():
        raise ToolError(f"Файл не найден: {_display_path('leads', file_path, LEADS_DIR)}")

    if not file_path.is_file():
        raise ToolError("Путь должен указывать на файл.")

    return {
        "path": _display_path("leads", file_path, LEADS_DIR),
        "content": file_path.read_text(encoding="utf-8"),
    }


def write_lead_review(filename: str, content: str) -> dict[str, str | int]:
    safe_filename = _validate_string(filename, "filename")
    if not isinstance(content, str):
        raise ToolError("`content` должен быть строкой.")

    review_path = _normalize_relative_markdown_path(safe_filename, "reviews", REVIEWS_DIR)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(content, encoding="utf-8")

    return {
        "path": _display_path("reviews", review_path, REVIEWS_DIR),
        "bytes": len(content.encode("utf-8")),
    }


def _require_only(arguments: dict[str, Any], required: set[str]) -> None:
    missing = required - set(arguments)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ToolError(f"Не хватает аргументов tool: {joined}.")

    extra = set(arguments) - required
    if extra:
        joined = ", ".join(sorted(extra))
        raise ToolError(f"Лишние аргументы tool: {joined}.")


def call_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ToolError("Аргументы tool должны быть словарём.")

    if tool_name == "read_lead_file":
        _require_only(arguments, {"filename"})
        return read_lead_file(arguments["filename"])

    if tool_name == "write_lead_review":
        _require_only(arguments, {"filename", "content"})
        return write_lead_review(arguments["filename"], arguments["content"])

    raise ToolError(f"Tool `{tool_name}` не разрешён для LeadReview Agent.")