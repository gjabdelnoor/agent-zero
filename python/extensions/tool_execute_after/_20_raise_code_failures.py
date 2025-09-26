import re
from typing import Optional

from python.helpers import errors
from python.helpers.errors import RepairableException
from python.helpers.extension import Extension
from python.helpers.tool import Response


class RaiseCodeFailures(Extension):
    """Convert failed code_execution_tool runs into repairable exceptions."""

    MAX_SNIPPET = 4000
    TRACEBACK_MARKER = "Traceback (most recent call last):"

    async def execute(
        self,
        response: Optional[Response] = None,
        tool_name: Optional[str] = None,
        **_: str,
    ) -> None:
        if tool_name != "code_execution_tool" or not response:
            return

        snippet = self._extract_failure_snippet(response.message or "")
        if not snippet:
            return

        snippet = self._truncate(snippet, self.MAX_SNIPPET)

        try:
            raise RuntimeError(snippet)
        except RuntimeError as exc:
            formatted = errors.format_error(exc, start_entries=0, end_entries=0)

        message = snippet or formatted
        if formatted and formatted not in message:
            message = f"{snippet}\n\n{formatted}" if snippet else formatted
        raise RepairableException(message)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _extract_failure_snippet(self, message: str) -> Optional[str]:
        if not message:
            return None

        traceback_snippet = self._extract_traceback(message)
        if traceback_snippet:
            return traceback_snippet

        return self._extract_shell_error(message)

    def _extract_traceback(self, message: str) -> Optional[str]:
        if self.TRACEBACK_MARKER not in message:
            return None

        lines = message.splitlines()
        for idx, line in enumerate(lines):
            if self.TRACEBACK_MARKER in line:
                start = max(0, idx - 5)
                snippet_lines = [
                    ln for ln in lines[start:] if not ln.startswith("[SYSTEM:")
                ]
                snippet = "\n".join(snippet_lines).strip()
                return snippet or None
        return None

    def _extract_shell_error(self, message: str) -> Optional[str]:
        lines = message.splitlines()
        for idx in range(len(lines) - 1, -1, -1):
            candidate = lines[idx].strip()
            if not candidate:
                continue
            if self._looks_like_shell_error(candidate):
                start = max(0, idx - 10)
                snippet_lines = [
                    ln
                    for ln in lines[start : idx + 1]
                    if ln.strip() and not ln.startswith("[SYSTEM:")
                ]
                snippet = "\n".join(snippet_lines).strip()
                if snippet:
                    return snippet
        return None

    ERROR_KEYWORDS = (
        "command not found",
        "not recognized as an internal or external command",
        "no such file or directory",
        "permission denied",
        "segmentation fault",
        "non-zero exit status",
        "returned non-zero exit status",
        "fatal:",
        "exception:",
        "cannot ",
        "can't ",
        "killed",
    )

    ERROR_REGEX = re.compile(r"\b(error|failed?)\b", re.IGNORECASE)

    def _looks_like_shell_error(self, line: str) -> bool:
        lower = line.lower()
        if lower.endswith(": not found"):
            return True
        for keyword in self.ERROR_KEYWORDS:
            if keyword in lower:
                if "error" in keyword and ("0 error" in lower or "no error" in lower):
                    continue
                return True
        if "error" in lower and "0 error" not in lower and "no error" not in lower:
            return True
        if self.ERROR_REGEX.search(line) and "0 fail" not in lower:
            return True
        return False

    def _truncate(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n... [snippet truncated]"
