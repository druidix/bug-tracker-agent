#!/usr/bin/env python3
"""
Anime 101 bug-tracker agent: a raw tool-use loop over a Jira-shaped CSV.

Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    cp bugs.sample.csv bugs.csv

Usage:
    ./bug_agent.py "Which high-priority bugs are still open?"
"""

import csv
import json
import sys
from datetime import date
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-5-5"
BUGS_CSV = Path(__file__).parent / "bugs.csv"
KEY_PREFIX = "ANIME-"
COLUMNS = [
    "key", "summary", "description", "issue_type",
    "priority", "status", "component", "created",
]
ISSUE_TYPES = ["Bug", "Improvement"]
PRIORITIES = ["Highest", "High", "Medium", "Low", "Lowest"]
STATUSES = ["To Do", "In Progress", "Done"]


def read_bugs() -> list[dict]:
    if not BUGS_CSV.exists():
        return []
    with BUGS_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def next_key(bugs: list[dict]) -> str:
    # Highest number + 1, not a row count, so a deleted row can't cause a reused key.
    numbers = [int(bug["key"].removeprefix(KEY_PREFIX)) for bug in bugs]
    return f"{KEY_PREFIX}{max(numbers, default=0) + 1}"


def list_bugs(status: str | None = None, priority: str | None = None) -> str:
    """Return bugs as JSON, optionally filtered by status and priority."""
    if status is not None and status not in STATUSES:
        return f"Error: status must be one of {STATUSES}"
    if priority is not None and priority not in PRIORITIES:
        return f"Error: priority must be one of {PRIORITIES}"
    bugs = [
        bug for bug in read_bugs()
        if (status is None or bug["status"] == status)
        and (priority is None or bug["priority"] == priority)
    ]
    return json.dumps(bugs)


def add_bug(
    summary: str, description: str, issue_type: str, priority: str, component: str
) -> str:
    """Append a new bug. Key, status and created date are set here, not by the model."""
    if not summary.strip():
        return "Error: summary can't be empty"
    if issue_type not in ISSUE_TYPES:
        return f"Error: issue_type must be one of {ISSUE_TYPES}"
    if priority not in PRIORITIES:
        return f"Error: priority must be one of {PRIORITIES}"

    # shortcut: no file locking, safe for one user; move to SQLite if two processes write.
    bugs = read_bugs()
    key = next_key(bugs)
    is_new_file = not BUGS_CSV.exists()
    with BUGS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new_file:
            writer.writeheader()
        writer.writerow({
            "key": key,
            "summary": summary.strip(),
            "description": description,
            "issue_type": issue_type,
            "priority": priority,
            "status": "To Do",
            "component": component,
            "created": date.today().isoformat(),
        })
    return f"Created {key}"


tools = [
    {
        "name": "list_bugs",
        "description": (
            "List Anime 101 bugs and improvements from the tracker. Filter by "
            "status or priority, or omit both to list everything."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": STATUSES},
                "priority": {"type": "string", "enum": PRIORITIES},
            },
        },
    },
    {
        "name": "add_bug",
        "description": (
            "Add a new issue to the Anime 101 tracker. The key, status and "
            "created date are assigned automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "A short one-line headline"},
                "description": {
                    "type": "string",
                    "description": "Steps to reproduce, and expected vs. actual behavior",
                },
                "issue_type": {"type": "string", "enum": ISSUE_TYPES},
                "priority": {"type": "string", "enum": PRIORITIES},
                "component": {
                    "type": "string",
                    "description": "The app area, e.g. 'Drawing canvas'",
                },
            },
            "required": ["summary", "description", "issue_type", "priority", "component"],
        },
    },
]

tool_functions = {"list_bugs": list_bugs, "add_bug": add_bug}


def run_agent(user_message: str) -> None:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=1024, tools=tools, messages=messages
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            print("".join(b.text for b in response.content if b.type == "text"))
            return

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[tool: {block.name}({block.input})]")
                result = tool_functions[block.name](**block.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )
        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('Usage: ./bug_agent.py "your request"')
    run_agent(sys.argv[1])
