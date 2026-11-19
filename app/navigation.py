"""Five-section navigation registry for the simplified research workspace."""

from __future__ import annotations

from typing import Callable, Dict

from app.pages import (
    experiment_lab,
    overview,
    research_runs,
    results_analysis,
    settings,
)

PAGES: Dict[str, Callable[[], None]] = {
    "Overview": overview.render,
    "Experiment Lab": experiment_lab.render,
    "Research Runs": research_runs.render,
    "Results & Analysis": results_analysis.render,
    "Settings": settings.render,
}
