"""Stage 7: evaluate the integrated pipeline across deterministic seeds."""

from common import EXAMPLE_COMMAND
from common.pipeline import run_once
from common.types import EvalReport, SeedResult


def evaluate(seeds: list[int]) -> EvalReport:
    """Run the full pipeline once per seed and aggregate honest outcomes."""
    results: list[SeedResult] = []
    successes = 0
    for seed in seeds:
        try:
            run = run_once(EXAMPLE_COMMAND, seed=seed)
            success = run.success
            details = _run_details(run.success, run.attempts, run.log)
        except Exception as exc:  # a stage blew up: count as failure
            success = False
            details = f"pipeline error: {exc}"
        successes += success
        results.append(SeedResult(seed=seed, success=success, details=details))

    return EvalReport(
        seeds=seeds,
        successes=successes,
        success_rate=successes / len(seeds) if seeds else 0.0,
        results=results,
    )


def _run_details(success: bool, attempts: int, log: list[str]) -> str:
    """Summarize the final verification result without hiding failures."""
    verification_lines = [line for line in log if line.startswith("           -> ok=")]
    final_verification = verification_lines[-1] if verification_lines else "verification unavailable"
    status = "completed" if success else "failed"
    return f"{status} after {attempts} attempt(s); {final_verification.strip()}"
