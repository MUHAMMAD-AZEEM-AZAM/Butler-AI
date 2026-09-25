from importlib import import_module
from types import SimpleNamespace


evaluate_module = import_module("stage7_eval.evaluate")


def test_evaluate_aggregates_successes_and_details(monkeypatch):
    def fake_run_once(command, seed):
        return SimpleNamespace(
            success=seed == 0,
            attempts=1 if seed == 0 else 2,
            log=[
                "[verify]   stage6_verify.verify(...) ",
                f"           -> ok={seed == 0} replan={seed != 0} (fixture)",
            ],
        )

    monkeypatch.setattr(evaluate_module, "run_once", fake_run_once)

    report = evaluate_module.evaluate([0, 1])

    assert report.seeds == [0, 1]
    assert report.successes == 1
    assert report.success_rate == 0.5
    assert report.results[0].success is True
    assert "completed after 1 attempt" in report.results[0].details
    assert report.results[1].success is False
    assert "failed after 2 attempt" in report.results[1].details


def test_evaluate_records_pipeline_exceptions(monkeypatch):
    def failing_run_once(command, seed):
        raise RuntimeError(f"fixture failure {seed}")

    monkeypatch.setattr(evaluate_module, "run_once", failing_run_once)

    report = evaluate_module.evaluate([7])

    assert report.successes == 0
    assert report.success_rate == 0.0
    assert report.results[0].success is False
    assert report.results[0].details == "pipeline error: fixture failure 7"


def test_empty_seed_list_has_zero_rate():
    report = evaluate_module.evaluate([])

    assert report.seeds == []
    assert report.successes == 0
    assert report.success_rate == 0.0
    assert report.results == []
