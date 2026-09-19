"""The per-leg cost record, and the arithmetic the comparison rests on."""

from __future__ import annotations

from dataclasses import fields

from longline.eval.leg_cost import COST_FIELDS, LegCost, mean_cost, saving


def test_cost_fields_covers_every_field_of_the_dataclass() -> None:
    """A field added to the dataclass and forgotten here would be dropped from
    every row AND every summary mean, silently -- the table would just be missing
    a column nobody remembers was ever there."""
    assert {f.name for f in fields(LegCost)} == set(COST_FIELDS)


def test_to_dict_carries_every_field() -> None:
    cost = LegCost(model_calls=6, tool_calls=4, input_tokens=600, output_tokens=125)
    assert set(cost.to_dict()) == set(COST_FIELDS)
    assert cost.to_dict()["model_calls"] == 6


def test_from_dict_reads_a_worker_report() -> None:
    cost = LegCost.from_dict(
        {
            "model_calls": 2,
            "tool_calls": 1,
            "input_tokens": 200,
            "output_tokens": 45,
            "loop_ms": 31.5,
        }
    )
    assert cost.model_calls == 2
    assert cost.tool_calls == 1
    assert cost.loop_ms == 31.5


def test_from_dict_of_nothing_is_zero_not_an_error() -> None:
    """A killed leg reports nothing, and "no report" has to read as zeroes.

    Raising here would turn the expected death of a child process into a harness
    error -- and a leg that died is the normal case in this suite, not the
    exceptional one.
    """
    for empty in (None, {}, "junk", [], 0):
        assert LegCost.from_dict(empty) == LegCost()


def test_from_dict_survives_a_missing_or_unparseable_field() -> None:
    cost = LegCost.from_dict({"model_calls": "3", "loop_ms": None})
    assert cost.model_calls == 3
    assert cost.loop_ms == 0.0
    assert cost.tool_calls == 0


def test_mean_cost_of_nothing_is_zeroes() -> None:
    assert mean_cost([]) == dict.fromkeys(COST_FIELDS, 0.0)


def test_mean_cost_averages_every_field() -> None:
    mean = mean_cost([LegCost(model_calls=4, tool_calls=3, loop_ms=10.0), LegCost(2, 1, loop_ms=20.0)])
    assert mean["model_calls"] == 3.0
    assert mean["tool_calls"] == 2.0
    assert mean["loop_ms"] == 15.0


def test_saving_is_the_share_of_the_restart_the_resume_did_not_redo() -> None:
    assert saving(6.0, 2.0) == 1.0 - 2.0 / 6.0
    assert saving(4.0, 4.0) == 0.0
    assert saving(4.0, 0.0) == 1.0


def test_saving_is_not_clamped_when_the_resume_did_more() -> None:
    """A resume that redid MORE than a restart is a real possible outcome (a
    checkpoint that forces a repair, say). Clamping it to 0 would hide it, and it
    is exactly the shape a broken checkpoint would take."""
    assert saving(2.0, 3.0) < 0.0
