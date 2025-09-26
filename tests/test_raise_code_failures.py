import os
import sys

import asyncio
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.extensions.tool_execute_after._20_raise_code_failures import (  # type: ignore  # noqa: E402
    RaiseCodeFailures,
)
from python.helpers.errors import RepairableException  # type: ignore  # noqa: E402
from python.helpers.tool import Response  # type: ignore  # noqa: E402


def test_traceback_triggers_repairable_exception():
    extension = RaiseCodeFailures(agent=None)
    response = Response(
        message=(
            "bash> python script.py\n"
            "Traceback (most recent call last):\n"
            "  File \"script.py\", line 1, in <module>\n"
            "ValueError: boom"
        ),
        break_loop=False,
    )

    with pytest.raises(RepairableException) as excinfo:
        asyncio.run(
            extension.execute(response=response, tool_name="code_execution_tool")
        )

    assert "Traceback (most recent call last):" in str(excinfo.value)


def test_non_error_output_is_ignored():
    extension = RaiseCodeFailures(agent=None)
    response = Response(message="bash> echo ok\nok", break_loop=False)
    asyncio.run(extension.execute(response=response, tool_name="code_execution_tool"))


def test_shell_error_is_detected():
    extension = RaiseCodeFailures(agent=None)
    response = Response(
        message="bash> ls missing\nls: cannot access 'missing': No such file or directory",
        break_loop=False,
    )

    with pytest.raises(RepairableException):
        asyncio.run(
            extension.execute(response=response, tool_name="code_execution_tool")
        )
