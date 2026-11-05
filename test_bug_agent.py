#!/usr/bin/env python3
import csv
import json
from datetime import date

import pytest

import bug_agent
from bug_agent import add_bug, list_bugs


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
    assert add_bug(*args).startswith("Error:")
    assert not bugs_file.exists()


def test_list_rejects_bad_filters():
    assert list_bugs(status="Closed").startswith("Error:")
    assert list_bugs(priority="Urgent").startswith("Error:")
