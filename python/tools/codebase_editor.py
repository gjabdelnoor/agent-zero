"""Repository-aware editing helper for self-repair workflows."""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Optional

from python.helpers import files
from python.helpers.tool import Response, Tool


@dataclass
class _WriteResult:
    path: str
    created: bool
    bytes_written: int
    preview: Optional[str]


class CodebaseEditor(Tool):
    """Expose safe file-management actions rooted at the repository base."""

    async def execute(self, **_: str) -> Response:
        action = (self.args.get("action") or "").strip().lower()
        if not action:
            return self._error("Missing required 'action' argument.")

        handlers = {
            "list": self._handle_list,
            "read": self._handle_read,
            "write": self._handle_write,
            "append": self._handle_append,
        }

        handler = handlers.get(action)
        if not handler:
            supported = ", ".join(sorted(handlers))
            return self._error(
                f"Unsupported action '{action}'. Expected one of: {supported}."
            )

        try:
            message = handler()
        except Exception as exc:  # pragma: no cover - defensive safety net
            return self._error(f"codebase_editor failed: {exc}")

        return Response(message=message, break_loop=False)

    # ------------------------------------------------------------------
    # action handlers
    # ------------------------------------------------------------------
    def _handle_list(self) -> str:
        rel_path = (self.args.get("path") or ".").strip() or "."
        pattern = (self.args.get("pattern") or "*").strip() or "*"
        abs_path = self._resolve_path(rel_path)

        if not os.path.exists(abs_path):
            return f"Path `{rel_path}` does not exist."
        if not os.path.isdir(abs_path):
            return f"Path `{rel_path}` is not a directory."

        entries: list[str] = []
        for name in sorted(os.listdir(abs_path)):
            if not fnmatch(name, pattern):
                continue
            full = os.path.join(abs_path, name)
            kind = "dir" if os.path.isdir(full) else "file"
            entries.append(f"- ({kind}) {name}")

        if not entries:
            return f"No entries in `{rel_path}` match pattern `{pattern}`."

        return "Contents of `{}` matching `{}`:\n{}".format(
            rel_path, pattern, "\n".join(entries)
        )

    def _handle_read(self) -> str:
        rel_path = (self.args.get("path") or "").strip()
        if not rel_path:
            raise ValueError("Missing 'path' for read action.")
        abs_path = self._resolve_path(rel_path)

        if not os.path.exists(abs_path):
            return f"File `{rel_path}` does not exist."
        if not os.path.isfile(abs_path):
            return f"Path `{rel_path}` is not a regular file."

        encoding = (self.args.get("encoding") or "utf-8").strip() or "utf-8"
        start_line = self._parse_int(self.args.get("start_line"), minimum=1)
        end_line = self._parse_int(self.args.get("end_line"), minimum=1)

        with open(abs_path, "r", encoding=encoding) as fh:
            lines = fh.read().splitlines()

        start = start_line - 1 if start_line is not None else 0
        stop = end_line if end_line is not None else len(lines)
        snippet = "\n".join(lines[start:stop])

        header = f"Contents of `{rel_path}`"
        if start_line is not None or end_line is not None:
            header += f" (lines {start_line or 1}-{end_line or len(lines)})"

        return f"{header}:\n```\n{snippet}\n```"

    def _handle_write(self) -> str:
        rel_path = (self.args.get("path") or "").strip()
        if not rel_path:
            raise ValueError("Missing 'path' for write action.")
        content = self.args.get("content")
        if content is None:
            raise ValueError("Missing 'content' for write action.")

        encoding = (self.args.get("encoding") or "utf-8").strip() or "utf-8"
        preview = self._parse_bool(self.args.get("preview"), default=True)

        result = self._write_file(
            rel_path,
            content,
            encoding=encoding,
            append=False,
            preview=preview,
        )

        return self._format_write_response("write", result)

    def _handle_append(self) -> str:
        rel_path = (self.args.get("path") or "").strip()
        if not rel_path:
            raise ValueError("Missing 'path' for append action.")
        content = self.args.get("content")
        if content is None:
            raise ValueError("Missing 'content' for append action.")

        encoding = (self.args.get("encoding") or "utf-8").strip() or "utf-8"
        preview = self._parse_bool(self.args.get("preview"), default=True)

        result = self._write_file(
            rel_path,
            content,
            encoding=encoding,
            append=True,
            preview=preview,
        )

        return self._format_write_response("append", result)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _write_file(
        self,
        rel_path: str,
        content: str,
        *,
        encoding: str,
        append: bool,
        preview: bool,
    ) -> _WriteResult:
        abs_path = self._resolve_path(rel_path)
        os.makedirs(os.path.dirname(abs_path) or abs_path, exist_ok=True)

        existing = ""
        created = not os.path.exists(abs_path)
        if not created:
            if not os.path.isfile(abs_path):
                raise ValueError(f"Path `{rel_path}` is not a regular file.")
            with open(abs_path, "r", encoding=encoding) as fh:
                existing = fh.read()

        mode = "a" if append else "w"
        with open(abs_path, mode, encoding=encoding) as fh:
            fh.write(content)

        preview_text: Optional[str] = None
        if preview:
            if append:
                preview_text = content if content else "(no content appended)"
            else:
                diff_lines = difflib.unified_diff(
                    existing.splitlines(),
                    content.splitlines(),
                    fromfile=f"a/{rel_path}",
                    tofile=f"b/{rel_path}",
                    lineterm="",
                )
                preview_text = "\n".join(diff_lines) or "(no changes)"

        preview_text = self._truncate_preview(preview_text)

        return _WriteResult(
            path=rel_path,
            created=created,
            bytes_written=len(content.encode(encoding)),
            preview=preview_text,
        )

    def _format_write_response(self, action: str, result: _WriteResult) -> str:
        status = "created" if result.created else "updated"
        message = (
            f"{action.title()} {status} `{result.path}` ({result.bytes_written} bytes)."
        )
        if result.preview:
            fence = "diff" if action == "write" else ""
            fence_header = f"```{fence}\n" if fence else "```\n"
            message += f"\n\nPreview:\n{fence_header}{result.preview}\n```"
        return message

    def _truncate_preview(self, preview: Optional[str], limit: int = 4000) -> Optional[str]:
        if not preview:
            return preview
        if len(preview) <= limit:
            return preview
        return preview[:limit] + "\n... [preview truncated]"

    def _resolve_path(self, rel_path: str) -> str:
        sanitized = rel_path.strip().lstrip("/")
        abs_path = files.get_abs_path(sanitized)
        abs_path = os.path.abspath(abs_path)
        if not files.is_in_base_dir(abs_path):
            raise ValueError("Path escapes repository root.")
        return abs_path

    def _parse_int(self, value: str | None, *, minimum: int | None = None) -> Optional[int]:
        if value is None:
            return None
        try:
            parsed = int(str(value))
        except ValueError as exc:  # pragma: no cover - arguments validated upstream
            raise ValueError(f"Invalid integer value: {value}") from exc
        if minimum is not None and parsed < minimum:
            raise ValueError(f"Value {parsed} must be >= {minimum}.")
        return parsed

    def _parse_bool(self, value: str | None, *, default: bool) -> bool:
        if value is None:
            return default
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
        raise ValueError(f"Invalid boolean value: {value}")

    def _error(self, message: str) -> Response:
        return Response(message=message, break_loop=False)


ToolClass = CodebaseEditor
