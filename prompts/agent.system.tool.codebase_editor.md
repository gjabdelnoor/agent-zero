### codebase_editor
Inspect and modify repository files without leaving the self-healing loop. All operations are scoped to the project root—paths may be relative (`python/tools`) or explicit (`python/tools/codebase_editor.py`), but attempts to escape the repository fail.

Supported actions (set via `"action"`):
- `list` – enumerate directory contents. Provide `path` (defaults to `.`) and optional `pattern` glob (defaults to `*`).
- `read` – return file contents. Supply `path`; optional `start_line`/`end_line` slice the output.
- `write` – replace a file with new `content`. Parents are created automatically. A unified diff preview is returned unless you pass `preview: "false"`.
- `append` – add `content` to the end of a file (creating it if absent). A tail preview is returned unless `preview: "false"`.

Remember to describe your edits in the thoughts section before calling the tool, and review the diff/preview the tool returns to double-check your work before running tests or other commands.
