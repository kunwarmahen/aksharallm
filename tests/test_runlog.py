"""train/runlog.py: the one reader of a run's step log, shared by the dashboard, the sessions
table, scripts/sessions.py and the run report."""

from aksharallm.train import runlog


def test_a_sessions_closing_evaluation_counts_as_a_validation_point():
    """The last eval of a run is written only as `session_end.final_val_loss` (gotcha 24a:
    never a bare `val_loss` on a record with no step). Every reader that skipped it showed a
    finished run's second-to-last validation -- best 1.1401 at step 5,500 for a run whose
    checkpoint holds 1.1387 at 5,999 -- on the tile, the chart and the sessions table."""
    recs = [
        {"event": "session_start", "time": 0, "max_steps": 6000},
        {"step": 5500, "loss": 1.9},
        {"step": 5500, "val_loss": 1.1401},
        {"step": 5999, "loss": 1.8},
        {"event": "session_end", "time": 1, "reason": "max_steps", "last_step": 5999,
         "final_val_loss": 1.1387},
    ]
    last = runlog.latest(recs)
    assert last["best_val"] == 1.1387 and last["val_step"] == 5999
    s = runlog.series(recs)
    assert s["val_step"] == [5500, 5999] and s["val_loss"] == [1.1401, 1.1387]
    (row,) = runlog.summarise_sessions(runlog.split_sessions(recs))
    assert row["best_val"] == 1.1387
    # ...and not twice, when a trainer also wrote the row at that step.
    both = recs[:4] + [{"step": 5999, "val_loss": 1.1387}] + recs[4:]
    assert runlog.series(both)["val_step"] == [5500, 5999]
    # A session record with no step is still never read as an eval on its own.
    assert runlog.val_records([{"event": "session_end", "final_val_loss": 1.0}]) == []
