"""Navigation registry mapping the 10 research dashboard pages."""

from __future__ import annotations

from typing import Callable, Dict

from app.pages import (
    annotation,
    api_configuration,
    dataset_management,
    export,
    main_experiment,
    overview,
    pilot_experiment,
    results_dashboard,
    robustness,
    statistical_analysis,
)

PAGES: Dict[str, Callable[[], None]] = {
    "1. Research Overview": overview.render,
    "2. API Configuration": api_configuration.render,
    "3. Dataset Management": dataset_management.render,
    "4. Pilot Experiment": pilot_experiment.render,
    "5. Main Experiment": main_experiment.render,
    "6. Response Annotation": annotation.render,
    "7. Results Dashboard": results_dashboard.render,
    "8. Statistical Analysis": statistical_analysis.render,
    "9. Robustness Analysis": robustness.render,
    "10. Export & Research Files": export.render,
}
