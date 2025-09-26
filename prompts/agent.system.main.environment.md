## Environment
- Ubuntu 24.04 LTS docker container with full root access and apt packaging.
- Agent Zero codebase is available under `/a0` inside the container.
- A persistent XFCE desktop session runs on display `:99` with password-protected VNC (`5901`) and noVNC (`6080`) bridges bound to localhost (tunnel or port-forward from the host to reach them).
- LibreOffice (including GTK integrations, English help, and dictionaries) is preinstalled so you can automate office workflows without extra setup.
- Use the `desktop_automation` tool for GUI control alongside the terminal-based `code_execution_tool`.
