import asyncio
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

from python.helpers.tool import Tool, Response
from python.helpers.print_style import PrintStyle

AwaitableResult = Tuple[str, dict[str, Any] | None]


class DesktopAutomation(Tool):
    """Control the supervised desktop session running inside the container."""

    DEFAULT_DISPLAY = ":99"
    DEFAULT_VNC_PORT = "5901"
    DEFAULT_NOVNC_PORT = "6080"
    SCREENSHOT_DIR = Path("tmp/desktop")

    async def execute(self, **kwargs) -> Response:
        payload: dict[str, Any] = {**self.args, **kwargs}
        method = (self.method or payload.get("method") or payload.get("action") or "").lower()

        if not method:
            return Response(
                message="No desktop automation method provided. Specify an action via the tool method or an 'action' argument.",
                break_loop=False,
            )

        handler_name = f"_handle_{method}"
        handler = getattr(self, handler_name, None)

        if not handler or not callable(handler):
            return Response(
                message=f"Unsupported desktop automation method '{method}'.",
                break_loop=False,
            )

        try:
            message, additional = await handler(payload)  # type: ignore[misc]
        except FileNotFoundError as exc:
            error = f"Required desktop binary not found: {exc}. Ensure the desktop stack is installed."
            PrintStyle().error(error)
            return Response(message=error, break_loop=False)
        except Exception as exc:
            error = f"Desktop automation '{method}' failed: {exc}"
            PrintStyle().error(error)
            return Response(message=error, break_loop=False)

        return Response(message=message, break_loop=False, additional=additional)

    # -- utility helpers -------------------------------------------------

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("DISPLAY", os.environ.get("DESKTOP_DISPLAY", self.DEFAULT_DISPLAY))
        env.setdefault("XDG_RUNTIME_DIR", "/var/run/desktop-session/root")
        env.setdefault("DESKTOP_VNC_PORT", self.DEFAULT_VNC_PORT)
        env.setdefault("DESKTOP_NOVNC_PORT", self.DEFAULT_NOVNC_PORT)
        return env

    async def _run_command(
        self,
        *command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        check: bool = True,
    ) -> Tuple[str, str]:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env or self._environment(),
            cwd=cwd,
        )
        stdout, stderr = await process.communicate()
        stdout_text = stdout.decode().strip()
        stderr_text = stderr.decode().strip()
        if check and process.returncode != 0:
            raise RuntimeError(
                f"Command '{' '.join(shlex.quote(c) for c in command)}' failed with code {process.returncode}: {stderr_text or stdout_text}"
            )
        return stdout_text, stderr_text

    async def _run_shell(
        self,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        start_new_session: bool = False,
    ) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_shell(
            command,
            env=env or self._environment(),
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=start_new_session,
        )

    async def _ensure_display(self, env: dict[str, str]) -> None:
        display = env.get("DISPLAY", self.DEFAULT_DISPLAY)
        try:
            await self._run_command("xset", "-display", display, "q", env=env)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Display {display} is not reachable. Confirm that the desktop supervisor services are running."
            ) from exc

    def _resolve_bool(self, payload: dict[str, Any], key: str, default: bool = False) -> bool:
        raw = payload.get(key, default)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            value = raw.strip().lower()
            if value in {"1", "true", "yes", "y"}:
                return True
            if value in {"0", "false", "no", "n"}:
                return False
        return default

    # -- handlers --------------------------------------------------------

    async def _handle_status(self, payload: dict[str, Any]) -> AwaitableResult:
        env = self._environment()
        await self._ensure_display(env)
        wmctrl, _ = await self._run_command("wmctrl", "-m", env=env)
        sessions = [line for line in wmctrl.splitlines() if line]
        ports = f"VNC: {env.get('DESKTOP_VNC_PORT', self.DEFAULT_VNC_PORT)}, noVNC: {env.get('DESKTOP_NOVNC_PORT', self.DEFAULT_NOVNC_PORT)}"
        return ("Desktop session is reachable.\n" + "\n".join(sessions) + f"\n{ports}", None)

    async def _handle_key(self, payload: dict[str, Any]) -> AwaitableResult:
        keys = payload.get("keys")
        if not keys:
            raise ValueError("Provide a 'keys' argument, e.g., 'ctrl+alt+t'.")
        env = self._environment()
        await self._ensure_display(env)
        await self._run_command("xdotool", "key", "--delay", str(payload.get("delay", 12)), str(keys), env=env)
        return (f"Sent key sequence '{keys}'.", None)

    async def _handle_type(self, payload: dict[str, Any]) -> AwaitableResult:
        text = payload.get("text")
        if text is None:
            raise ValueError("Provide a 'text' argument to type.")
        env = self._environment()
        await self._ensure_display(env)
        args = ["xdotool", "type", "--delay", str(payload.get("delay", 12))]
        if self._resolve_bool(payload, "clearmodifiers", True):
            args.extend(["--clearmodifiers"])
        args.append(str(text))
        await self._run_command(*args, env=env)
        return (f"Typed {len(str(text))} characters.", None)

    async def _handle_click(self, payload: dict[str, Any]) -> AwaitableResult:
        button = str(payload.get("button", 1))
        env = self._environment()
        await self._ensure_display(env)
        await self._run_command("xdotool", "click", button, env=env)
        return (f"Clicked mouse button {button}.", None)

    async def _handle_move(self, payload: dict[str, Any]) -> AwaitableResult:
        x = payload.get("x")
        y = payload.get("y")
        if x is None or y is None:
            raise ValueError("Provide integer 'x' and 'y' coordinates.")
        env = self._environment()
        await self._ensure_display(env)
        await self._run_command("xdotool", "mousemove", str(x), str(y), env=env)
        return (f"Moved pointer to ({x}, {y}).", None)

    async def _handle_drag(self, payload: dict[str, Any]) -> AwaitableResult:
        def _coerce_coordinate(name: str) -> int:
            value = payload.get(name)
            if value is None:
                raise ValueError("Provide integer 'start_x', 'start_y', 'end_x', and 'end_y' coordinates.")
            try:
                return int(round(float(value)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Coordinate '{name}' must be numeric.") from exc

        start_x = _coerce_coordinate("start_x")
        start_y = _coerce_coordinate("start_y")
        end_x = _coerce_coordinate("end_x")
        end_y = _coerce_coordinate("end_y")

        button = str(payload.get("button", 1))
        duration_raw = payload.get("duration", 0)
        try:
            duration = max(float(duration_raw), 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("Duration must be numeric if provided.") from exc

        env = self._environment()
        await self._ensure_display(env)

        await self._run_command("xdotool", "mousemove", str(start_x), str(start_y), env=env)

        pressed = False
        try:
            await self._run_command("xdotool", "mousedown", button, env=env)
            pressed = True

            if duration > 0:
                steps = max(int(duration * 30), 1)
                interval = duration / steps
                for step in range(1, steps + 1):
                    progress = step / steps
                    x = round(start_x + (end_x - start_x) * progress)
                    y = round(start_y + (end_y - start_y) * progress)
                    await self._run_command("xdotool", "mousemove", str(x), str(y), env=env)
                    if step < steps:
                        await asyncio.sleep(interval)
            else:
                await self._run_command("xdotool", "mousemove", str(end_x), str(end_y), env=env)
        finally:
            if pressed:
                await self._run_command("xdotool", "mouseup", button, env=env, check=False)

        return (
            f"Dragged mouse button {button} from ({start_x}, {start_y}) to ({end_x}, {end_y})"
            + (f" over {duration:.2f}s." if duration > 0 else "."),
            None,
        )

    async def _handle_scroll(self, payload: dict[str, Any]) -> AwaitableResult:
        amount = int(payload.get("amount", 1))
        direction = str(payload.get("direction", "down")).lower()
        if direction not in {"up", "down"}:
            raise ValueError("Scroll direction must be 'up' or 'down'.")
        button = "4" if direction == "up" else "5"
        env = self._environment()
        await self._ensure_display(env)
        for _ in range(abs(amount)):
            await self._run_command("xdotool", "click", button, env=env)
        return (f"Scrolled {direction} {amount} step(s).", None)

    async def _handle_focus(self, payload: dict[str, Any]) -> AwaitableResult:
        target = payload.get("window") or payload.get("title")
        if not target:
            raise ValueError("Provide a 'window' or 'title' argument to focus.")
        env = self._environment()
        await self._ensure_display(env)
        await self._run_command("wmctrl", "-a", str(target), env=env)
        return (f"Focused window matching '{target}'.", None)

    async def _handle_windows(self, payload: dict[str, Any]) -> AwaitableResult:
        env = self._environment()
        await self._ensure_display(env)
        output, _ = await self._run_command("wmctrl", "-lx", env=env)
        if not output:
            return ("No windows reported by wmctrl.", None)
        return ("Active windows:\n" + output, None)

    async def _handle_launch(self, payload: dict[str, Any]) -> AwaitableResult:
        command = payload.get("command")
        if not command:
            raise ValueError("Provide a 'command' to launch an application.")
        env = self._environment()
        await self._ensure_display(env)

        loop = asyncio.get_running_loop()

        def _spawn() -> subprocess.Popen[Any]:
            return subprocess.Popen(  # noqa: S603,S607
                ["bash", "-lc", str(command)],
                env=env,
                cwd=payload.get("cwd"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        process = await loop.run_in_executor(None, _spawn)
        return (f"Launched '{command}' (pid {process.pid}).", None)

    async def _handle_screenshot(self, payload: dict[str, Any]) -> AwaitableResult:
        env = self._environment()
        await self._ensure_display(env)
        path = await self._capture_screenshot(env, payload.get("filename"))
        message = (
            f"Captured screenshot to {path}. Use `desktop_automation` with the `inspect` method for OCR or `vision_load` for "
            "detailed viewing."
        )
        return (message, {"attachments": [str(path)]})

    async def _handle_inspect(self, payload: dict[str, Any]) -> AwaitableResult:
        env = self._environment()
        await self._ensure_display(env)
        path = await self._capture_screenshot(env, payload.get("filename"))

        args = ["tesseract", str(path), "stdout"]
        language = payload.get("lang") or payload.get("language")
        if language:
            args.extend(["-l", str(language)])
        psm = payload.get("psm")
        if psm is not None:
            args.extend(["--psm", str(psm)])

        ocr_text, _ = await self._run_command(*args, env=env)
        ocr_text = ocr_text.strip()

        limit = 500
        preview = ocr_text[:limit]
        if not preview:
            preview = "[no text detected]"
            truncated = False
        else:
            truncated = len(ocr_text) > limit
        if truncated:
            preview += "… (truncated)"

        message = (
            f"Captured screenshot to {path}.\n"
            f"OCR preview (first {limit} chars):\n{preview}"
        )

        return (message, {"attachments": [str(path)], "ocr_text": ocr_text})

    async def _handle_run(self, payload: dict[str, Any]) -> AwaitableResult:
        command = payload.get("command")
        if not command:
            raise ValueError("Provide a 'command' to execute within the desktop session.")
        env = self._environment()
        await self._ensure_display(env)
        output, _ = await self._run_shell_and_capture(command, env=env, cwd=payload.get("cwd"))
        return (output or "Command executed with no output.", None)

    async def _run_shell_and_capture(
        self, command: str, env: dict[str, str], cwd: str | None
    ) -> Tuple[str, str]:
        process = await self._run_shell(command, env=env, cwd=cwd)
        stdout, stderr = await process.communicate()
        stdout_text = stdout.decode().strip()
        stderr_text = stderr.decode().strip()
        if process.returncode != 0:
            raise RuntimeError(
                f"Shell command '{command}' failed with code {process.returncode}: {stderr_text or stdout_text}"
            )
        return stdout_text, stderr_text

    async def _capture_screenshot(self, env: dict[str, str], filename: Any) -> Path:
        DesktopAutomation.SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        if filename:
            target = DesktopAutomation.SCREENSHOT_DIR / str(filename)
        else:
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            target = DesktopAutomation.SCREENSHOT_DIR / f"screenshot-{timestamp}.png"

        command = f"xwd -root -silent | convert xwd:- {shlex.quote(str(target))}"
        await self._run_shell_and_capture(command, env=env, cwd=None)
        return target
