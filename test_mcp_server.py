#!/usr/bin/env python3
"""MCP server tests, through the SDK's in-memory client: the real protocol, no subprocess."""

import json

import pytest
from mcp import Client

import mcp_server
import tracker

pytestmark = pytest.mark.anyio

GOOD_BUG = {
    "summary": "Crash",
    "description": "Steps",
    "issue_type": "Bug",
    "priority": "High",
    "component": "Canvas",
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def bugs_file(tmp_path, monkeypatch):
    path = tmp_path / "bugs.csv"
    monkeypatch.setattr(tracker, "BUGS_CSV", path)
    return path


@pytest.fixture
async def client():
    async with Client(mcp_server.mcp) as c:
        yield c


async def test_exposes_the_three_tools_with_model_facing_descriptions(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"list_bugs", "add_bug", "summarize_bugs"}
    assert tools["add_bug"].description == tracker.add_bug.__doc__


async def test_add_bug_schema_comes_from_the_type_hints(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    schema = tools["add_bug"].input_schema
    assert set(schema["required"]) == {"summary", "description", "issue_type", "priority", "component"}
    assert schema["properties"]["priority"]["enum"] == tracker.PRIORITIES
    assert "required" not in tools["list_bugs"].input_schema  # both filters are optional


async def test_add_and_list_round_trip(client):
    result = await client.call_tool("add_bug", GOOD_BUG)
    assert result.is_error is False
    assert result.content[0].text == "Created ANIME-1"
    listed = await client.call_tool("list_bugs", {"priority": "High"})
    assert [bug["key"] for bug in json.loads(listed.content[0].text)] == ["ANIME-1"]


async def test_our_validation_error_reaches_the_model_with_its_message(client, bugs_file):
    result = await client.call_tool("add_bug", {**GOOD_BUG, "summary": "   "})
    assert result.is_error is True
    assert "summary can't be empty" in result.content[0].text
    assert not bugs_file.exists()


@pytest.mark.parametrize("bad_arguments", [
    {**GOOD_BUG, "priority": "Urgent"},
    {key: value for key, value in GOOD_BUG.items() if key != "priority"},
])
async def test_sdk_rejects_bad_arguments_before_our_code_runs(client, bugs_file, bad_arguments):
    result = await client.call_tool("add_bug", bad_arguments)
    assert result.is_error is True
    assert "priority" in result.content[0].text
    assert not bugs_file.exists()
