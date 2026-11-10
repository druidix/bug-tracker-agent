#!/usr/bin/env python3
"""
Anime 101 issue tracker: the data layer and tool functions, shared by
bug_agent.py (raw API agent) and mcp_server.py (MCP server).

Tool docstrings are written for the model: the MCP server sends them as tool
descriptions, and bug_agent.py uses them in its hand-written schemas.
"""

import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Literal, get_args

BUGS_CSV = Path(__file__).parent / "bugs.csv"
KEY_PREFIX = "ANIME-"
COLUMNS = [
    "key", "summary", "description", "issue_type",
    "priority", "status", "component", "created",
]
# The Literal types are the single source of allowed values. The MCP SDK turns
# them into schema enums; the lists feed bug_agent.py's schemas and validation.
IssueType = Literal["Bug", "Improvement"]
Priority = Literal["Highest", "High", "Medium", "Low", "Lowest"]
Status = Literal["To Do", "In Progress", "Done"]
ISSUE_TYPES = list(get_args(IssueType))
PRIORITIES = list(get_args(Priority))
STATUSES = list(get_args(Status))


def read_bugs() -> list[dict]:
    if not BUGS_CSV.exists():
        return []
    with BUGS_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def next_key(bugs: list[dict]) -> str:
    # Highest number + 1, not a row count, so a deleted row can't cause a reused key.
    numbers = [int(bug["key"].removeprefix(KEY_PREFIX)) for bug in bugs]
    return f"{KEY_PREFIX}{max(numbers, default=0) + 1}"


def list_bugs(status: Status | None = None, priority: Priority | None = None) -> str:
    """List Anime 101 bugs and improvements as JSON. Filter by status or priority, or omit both to list everything."""
    if status is not None and status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    if priority is not None and priority not in PRIORITIES:
        raise ValueError(f"priority must be one of {PRIORITIES}")
    bugs = [
        bug for bug in read_bugs()
        if (status is None or bug["status"] == status)
        and (priority is None or bug["priority"] == priority)
    ]
    return json.dumps(bugs)


def add_bug(
    summary: str, description: str, issue_type: IssueType, priority: Priority, component: str
) -> str:
    """Add a new issue to the Anime 101 tracker.

    summary: a short one-line headline. description: only details the user gave,
    such as steps to reproduce and expected vs. actual behavior. component: the
    app area, e.g. 'Drawing canvas'. The key, status and created date are
    assigned automatically.
    """
    # Key, status and created date are set here, never by the model.
    if not summary.strip():
        raise ValueError("summary can't be empty")
    if issue_type not in ISSUE_TYPES:
        raise ValueError(f"issue_type must be one of {ISSUE_TYPES}")
    if priority not in PRIORITIES:
        raise ValueError(f"priority must be one of {PRIORITIES}")

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


def summarize_bugs() -> str:
    """Get counts for the Anime 101 tracker as JSON: all issues by status, and open issues by priority and component. Use it for overview questions; use list_bugs for the issues themselves."""
    bugs = read_bugs()
    open_bugs = [bug for bug in bugs if bug["status"] != "Done"]
    priorities = Counter(bug["priority"] for bug in open_bugs)
    return json.dumps({
        "total": len(bugs),
        "by_status": dict(Counter(bug["status"] for bug in bugs)),
        "open_by_priority": {p: priorities[p] for p in PRIORITIES if priorities[p]},
        "open_by_component": dict(Counter(bug["component"] for bug in open_bugs).most_common()),
    })
