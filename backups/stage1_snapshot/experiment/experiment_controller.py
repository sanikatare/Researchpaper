"""Background thread controller for Pause / Resume / Stop lifecycle in Streamlit."""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from config import ExperimentConfig
from experiment.experiment_runner import build_trial_plan, filter_pending_trials, run_experiment_batch
from experiment.storage import (
    load_experiment_state,
    save_experiment_state,
)


class ExperimentController:
    """Thread-safe controller managing one background experiment run at a time."""

    def __init__(self, experiment_type: str) -> None:
        self.experiment_type = experiment_type
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._pause_requested = False
        self._stop_requested = False
        self._status = "IDLE"
        self._last_error: Optional[str] = None
        self._total_planned = 0
        self._skipped_existing = 0
        self._pending_at_start = 0
        self._executed_in_run = 0
        self._succeeded_in_run = 0
        self._failed_in_run = 0
        self._live_logs: Deque[str] = deque(maxlen=250)

    def _log(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._live_logs.appendleft(f"[{ts}] {msg}")

    def start(
        self,
        scenarios: List[Dict[str, Any]],
        config: ExperimentConfig,
        explicit_approved: bool = False,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        """Launch experiment in a daemon worker thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError(
                    f"A {self.experiment_type} experiment is already running."
                )
            if self.experiment_type == "main" and not explicit_approved:
                raise PermissionError(
                    "Main experiment requires explicit researcher approval before execution."
                )

            plan = build_trial_plan(
                scenarios=scenarios,
                modalities=config.modalities,
                thinking_levels=config.thinking_levels,
                repetitions=config.repetitions,
                randomize_order=config.randomize_order,
                random_seed=config.random_seed,
            )
            pending, skipped = filter_pending_trials(self.experiment_type, plan)

            self._pause_requested = False
            self._stop_requested = False
            self._status = "RUNNING"
            self._last_error = None
            self._total_planned = len(plan)
            self._skipped_existing = skipped
            self._pending_at_start = len(pending)
            self._executed_in_run = 0
            self._succeeded_in_run = 0
            self._failed_in_run = 0

            state = load_experiment_state(self.experiment_type)
            state["status"] = "RUNNING"
            state["approved"] = explicit_approved if self.experiment_type == "main" else True
            state["total_planned"] = len(plan)
            state["skipped_existing"] = skipped
            state.setdefault("history", []).append(
                {
                    "event": "started",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "pending": len(pending),
                    "skipped_existing": skipped,
                }
            )
            save_experiment_state(self.experiment_type, state)
            self._log(
                f"Starting {self.experiment_type}: {len(pending)} pending "
                f"({skipped} already completed)."
            )

            def _worker() -> None:
                try:
                    def _on_complete(rec: Dict[str, Any], idx: int, total: int) -> None:
                        with self._lock:
                            self._executed_in_run += 1
                            if rec["status"] in ("success", "blocked"):
                                self._succeeded_in_run += 1
                            else:
                                self._failed_in_run += 1
                            self._log(
                                f"[{idx}/{total}] {rec['scenario_id']} | {rec['modality']} | "
                                f"{rec['thinking_level']} | r{rec['repetition']} -> "
                                f"{rec['status']} ({rec['latency_ms']} ms)"
                                + (
                                    f" ERR: {rec['error_message']}"
                                    if rec.get("error_message")
                                    else ""
                                )
                            )

                    summary = run_experiment_batch(
                        experiment_type=self.experiment_type,
                        scenarios=scenarios,
                        config=config,
                        should_pause=lambda: self._pause_requested,
                        should_stop=lambda: self._stop_requested,
                        on_trial_complete=_on_complete,
                        api_key=api_key,
                        client=client,
                    )
                    reason = summary["halted_reason"]
                    final_status = {
                        "completed": "COMPLETED",
                        "paused": "PAUSED",
                        "stopped": "STOPPED",
                    }.get(reason, "COMPLETED")
                    with self._lock:
                        self._status = final_status
                        self._log(f"Experiment {final_status}: {summary}")
                        st = load_experiment_state(self.experiment_type)
                        st["status"] = final_status
                        st["completed_count"] = (
                            summary["skipped_existing"] + summary["succeeded"]
                        )
                        st["failed_count"] = summary["failed"]
                        st.setdefault("history", []).append(
                            {
                                "event": final_status.lower(),
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "summary": summary,
                            }
                        )
                        save_experiment_state(self.experiment_type, st)
                except Exception as exc:
                    with self._lock:
                        self._status = "ERROR"
                        self._last_error = str(exc)
                        self._log(f"ERROR: {exc}")
                        st = load_experiment_state(self.experiment_type)
                        st["status"] = "ERROR"
                        st["last_error"] = str(exc)
                        st.setdefault("history", []).append(
                            {
                                "event": "error",
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "error": str(exc),
                            }
                        )
                        save_experiment_state(self.experiment_type, st)

            self._thread = threading.Thread(
                target=_worker,
                name=f"exp-{self.experiment_type}-worker",
                daemon=True,
            )
            self._thread.start()

    def pause(self) -> None:
        with self._lock:
            if self._status == "RUNNING":
                self._pause_requested = True
                self._log("Pause requested; finishing current in-flight call...")

    def stop(self) -> None:
        with self._lock:
            if self._status == "RUNNING":
                self._stop_requested = True
                self._log("Stop requested; halting after current in-flight call...")

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "experiment_type": self.experiment_type,
                "status": self._status,
                "is_alive": self._thread is not None and self._thread.is_alive(),
                "total_planned": self._total_planned,
                "skipped_existing": self._skipped_existing,
                "pending_at_start": self._pending_at_start,
                "executed_in_run": self._executed_in_run,
                "succeeded_in_run": self._succeeded_in_run,
                "failed_in_run": self._failed_in_run,
                "last_error": self._last_error,
                "logs": list(self._live_logs),
            }


_CONTROLLERS: Dict[str, ExperimentController] = {}
_CTRL_LOCK = threading.Lock()


def get_controller(experiment_type: str) -> ExperimentController:
    """Singleton process-level controller so Streamlit reruns preserve running threads."""
    with _CTRL_LOCK:
        if experiment_type not in _CONTROLLERS:
            _CONTROLLERS[experiment_type] = ExperimentController(experiment_type)
        return _CONTROLLERS[experiment_type]
