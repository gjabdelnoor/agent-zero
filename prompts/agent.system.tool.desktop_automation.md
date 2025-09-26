### desktop_automation
Use this tool to interact with the supervised XFCE desktop that runs in the background of the Ubuntu container. The desktop is reachable on display `:99`, with a VNC bridge on `5901` and a web-based noVNC gateway on `6080` (both bound to localhost—create an SSH tunnel or port forwarder to reach them externally).

Supported methods (choose one per call):
- `status` – verify the desktop stack is healthy and report remote access ports.
- `launch` – start a GUI application by providing a `command` (e.g., `xfce4-terminal`, `libreoffice --calc`).
- `windows` – list currently open windows via `wmctrl`.
- `focus` – focus a window matching a substring using `window` or `title`.
- `key` – send a key chord such as `ctrl+alt+t` (optional `delay`).
- `type` – type arbitrary text with optional `clearmodifiers` (defaults to true).
- `click` – click a pointer button (`button` defaults to 1) at the current cursor location.
- `move` – move the cursor to absolute `x` and `y` coordinates in pixels.
- `scroll` – scroll the wheel `amount` steps in a `direction` (`up` or `down`).
- `drag` – press a mouse `button` (defaults to 1) at (`start_x`, `start_y`), glide to (`end_x`, `end_y`), and release. Optional `duration` (seconds) smooths the motion.
- `screenshot` – capture the current desktop to `tmp/desktop/<timestamp>.png`. Use `inspect` for immediate OCR or `vision_load` for manual review.
- `inspect` – capture a screenshot and run OCR in one step. Optional `lang` and `psm` tweak Tesseract settings; the response includes the recognized text in `ocr_text`.
- `run` – execute a shell `command` with display context if you need to inspect the GUI session from the command line.

Always plan your GUI actions carefully, prefer deterministic workflows, and take screenshots before and after major steps when visual confirmation is important.
