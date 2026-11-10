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

import sys

import anthropic

import tracker

MODEL = "claude-sonnet-5-5"
MAX_TURNS = 10  # model calls per request; stops a model that never stops asking for tools
MAX_TOKENS = 16000  # per response; you only pay for tokens actually generated
SYSTEM_PROMPT = (
    "You manage the Anime 101 issue tracker. When you add an issue, the summary "
    "and description must contain only what the user told you. Don't invent "
    "reproduction steps, causes or impact. If the user didn't give a priority or "
    "component, pick the best fit and tell them you chose it."
)


tools = [
    {
        "name": "list_bugs",
        "description": tracker.list_bugs.__doc__,
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": tracker.STATUSES},
                "priority": {"type": "string", "enum": tracker.PRIORITIES},
            },
        },
    },
    {
        "name": "add_bug",
        "description": tracker.add_bug.__doc__,
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "A short one-line headline"},
                "description": {
                    "type": "string",
                    "description": (
                        "Only details the user gave, such as steps to reproduce "
                        "and expected vs. actual behavior"
                    ),
                },
                "issue_type": {"type": "string", "enum": tracker.ISSUE_TYPES},
                "priority": {"type": "string", "enum": tracker.PRIORITIES},
                "component": {
                    "type": "string",
                    "description": "The app area, e.g. 'Drawing canvas'",
                },
            },
            "required": ["summary", "description", "issue_type", "priority", "component"],
        },
    },
    {
        "name": "summarize_bugs",
        "description": tracker.summarize_bugs.__doc__,
        "input_schema": {"type": "object", "properties": {}},
    },
]

tool_functions = {
    "list_bugs": tracker.list_bugs,
    "add_bug": tracker.add_bug,
    "summarize_bugs": tracker.summarize_bugs,
}


def run_tool(block) -> dict:
    """Run one tool_use block. Any failure goes back to the model as is_error, never a crash."""
    try:
        func = tool_functions.get(block.name)
        if func is None:
            raise ValueError(f"unknown tool: {block.name}")
        content, is_error = func(**block.input), False
    except Exception as e:  # bad arguments raise TypeError; validation raises ValueError
        content, is_error = f"{type(e).__name__}: {e}", True
        print(f"[tool error: {content}]")
    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": content,
        "is_error": is_error,
    }


def run_agent(user_message: str) -> None:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_message}]

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        text = "".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason == "end_turn":
            print(text)
            return
        if response.stop_reason != "tool_use":
            # max_tokens, refusal, etc.: not a finished answer, and any tool call may be cut off.
            print(f"Stopped early ({response.stop_reason}): {text}")
            return

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[tool: {block.name}({block.input})]")
                tool_results.append(run_tool(block))
        messages.append({"role": "user", "content": tool_results})

    print(f"Stopped after {MAX_TURNS} model calls without a final answer.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('Usage: ./bug_agent.py "your request"')
    run_agent(sys.argv[1])
