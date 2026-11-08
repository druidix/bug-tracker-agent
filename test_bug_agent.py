#!/usr/bin/env python3
import csv
import json
from datetime import date
from types import SimpleNamespace

import pytest

import bug_agent
from bug_agent import add_bug, list_bugs, summarize_bugs


@pytest.fixture(autouse=True)
def bugs_file(tmp_path, monkeypatch):
    path = tmp_path / "bugs.csv"
    monkeypatch.setattr(bug_agent, "BUGS_CSV", path)
    return path


def add(summary="Crash", priority="High"):
    return add_bug(summary, "Steps", "Bug", priority, "Canvas")


def test_list_on_missing_file_is_empty():
    assert json.loads(list_bugs()) == []


def test_add_creates_file_and_sets_code_owned_fields(bugs_file):
    assert add() == "Created ANIME-1"
    [bug] = json.loads(list_bugs())
    assert bug["status"] == "To Do"
    assert bug["created"] == date.today().isoformat()
    assert list(bug) == bug_agent.COLUMNS


def test_keys_increment_and_survive_a_deleted_row(bugs_file):
    add(), add(), add()
    with bugs_file.open(newline="") as f:
        rows = list(csv.DictReader(f))
    with bugs_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=bug_agent.COLUMNS)
        writer.writeheader()
        writer.writerows([rows[0], rows[2]])  # delete ANIME-2
    assert add() == "Created ANIME-4"


def test_filters():
    add("A", "High"), add("B", "Low")
    assert [b["summary"] for b in json.loads(list_bugs(priority="Low"))] == ["B"]
    assert json.loads(list_bugs(status="Done")) == []


def test_commas_quotes_and_newlines_round_trip():
    text = 'Line one, with "quotes"\nLine two'
    add_bug("Crash", text, "Bug", "High", "Canvas")
    assert json.loads(list_bugs())[0]["description"] == text


@pytest.mark.parametrize("args", [
    ("   ", "d", "Bug", "High", "c"),
    ("s", "d", "Epic", "High", "c"),
    ("s", "d", "Bug", "Urgent", "c"),
])
def test_add_rejects_bad_input_and_writes_nothing(args, bugs_file):
    with pytest.raises(ValueError):
        add_bug(*args)
    assert not bugs_file.exists()


def test_list_rejects_bad_filters():
    with pytest.raises(ValueError):
        list_bugs(status="Closed")
    with pytest.raises(ValueError):
        list_bugs(priority="Urgent")


def test_summarize_counts_open_issues_only_for_priority_and_component(bugs_file):
    assert json.loads(summarize_bugs())["total"] == 0
    add("A", "High"), add("B", "High"), add("C", "Low")
    text = bugs_file.read_text().replace("C,Steps,Bug,Low,To Do", "C,Steps,Bug,Low,Done")
    bugs_file.write_text(text)
    assert json.loads(summarize_bugs()) == {
        "total": 3,
        "by_status": {"To Do": 2, "Done": 1},
        "open_by_priority": {"High": 2},
        "open_by_component": {"Canvas": 2},
    }


# --- agent loop, with a fake client: no network, no API key needed ---

def tool_call(name, **tool_input):
    block = SimpleNamespace(type="tool_use", id="toolu_1", name=name, input=tool_input)
    return SimpleNamespace(stop_reason="tool_use", content=[block])


def final(text="done"):
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


def run_with(monkeypatch, next_response):
    """Run the agent; next_response(n) returns the fake reply to model call n."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return next_response(len(calls))

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(bug_agent.anthropic, "Anthropic", lambda: client)
    bug_agent.run_agent("hi")
    return calls


def first_tool_result(calls):
    return calls[-1]["messages"][2]["content"][0]


@pytest.mark.parametrize("bad_call, message", [
    (tool_call("delete_everything"), "unknown tool"),
    (tool_call("add_bug", summary="only one argument"), "TypeError"),
    (tool_call("list_bugs", status="Closed"), "status must be one of"),
])
def test_tool_failures_go_back_to_the_model_as_is_error(monkeypatch, bad_call, message):
    calls = run_with(monkeypatch, lambda n: bad_call if n == 1 else final())
    result = first_tool_result(calls)
    assert result["is_error"] is True
    assert message in result["content"]
    assert result["tool_use_id"] == "toolu_1"


def test_successful_tool_is_not_an_error(monkeypatch):
    calls = run_with(monkeypatch, lambda n: tool_call("summarize_bugs") if n == 1 else final())
    assert first_tool_result(calls)["is_error"] is False


def test_loop_stops_at_the_cap(monkeypatch, capsys):
    calls = run_with(monkeypatch, lambda n: tool_call("list_bugs"))
    assert len(calls) == bug_agent.MAX_TURNS
    assert "Stopped after" in capsys.readouterr().out
