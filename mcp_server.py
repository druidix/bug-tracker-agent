#!/usr/bin/env python3
"""
Anime 101 bug tracker as an MCP server, over stdio.

An MCP host (Claude Code, Claude Desktop, MCP Inspector) starts this file as
a subprocess and talks to it through stdin/stdout. The host runs the model and
the agent loop; this server only exposes the tracker's tools. No API key needed.

Requires:
    pip install "mcp[cli]"
    cp bugs.sample.csv bugs.csv

Try it in MCP Inspector:
    mcp dev mcp_server.py
"""

import functools

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import tracker

mcp = MCPServer("anime101-bugs")


def expose(func):
    """Register a tracker function as an MCP tool, turning its ValueErrors into ToolErrors."""

    @functools.wraps(func)  # keeps the name, docstring and type hints the SDK builds the schema from
    def tool(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            # ToolError means "anticipated failure": the model sees this message.
            # Any other exception reaches the model only as a generic error.
            raise ToolError(str(e)) from e

    return mcp.tool()(tool)


expose(tracker.list_bugs)
expose(tracker.add_bug)
expose(tracker.summarize_bugs)


if __name__ == "__main__":
    mcp.run()  # stdio by default. Never print() to stdout here: it would corrupt the protocol.
