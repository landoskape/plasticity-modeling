from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import optuna

from src.files import save_repo_snapshot
from src.iaf.analysis import proximal_weight_entropy
from src.iaf.experiments import get_proximal_experiment
from src.utils import create_rng
import src.utils as utils


@dataclass
class ProximalSearchSpace:
    num_synapses_min: int = 720
    num_synapses_max: int = 4320
    num_synapses_step: int = 36
    max_weight_min: float = 1e-13
    max_weight_max: float = 1.5e-9
    max_weight_log: bool = True
    conductance_threshold_min: float = 0.0
    conductance_threshold_max: float = 0.5
    independent_noise_rate_min: float | None = 0.0
    independent_noise_rate_max: float | None = 1.0
    stdp_rate_min: float = 1e-4
    stdp_rate_max: float = 0.1
    stdp_rate_log: bool = True
    dp_ratio_min: float = 0.95
    dp_ratio_max: float = 1.25
    baseline_rate_min: float = 0.0
    baseline_rate_max: float = 50.0
    driven_rate_min: float = 5.0
    driven_rate_max: float = 95.0
    concentration_min: float = 0.25
    concentration_max: float = 5.0
    synapse_weight: float = 0.05


def _maybe_write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_run_dir(base_study_name: str, run_dir: Path) -> tuple[Path, str]:
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return run_dir, metadata["study_name"]

    if run_dir.name:
        study_name = run_dir.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"{base_study_name}_{timestamp}"

    metadata = {
        "base_study_name": base_study_name,
        "study_name": study_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _maybe_write_json(metadata_path, metadata)
    return run_dir, study_name


def _build_storage_url(run_dir: Path) -> str:
    db_path = (run_dir / "optuna.db").resolve()
    return f"sqlite:///{db_path.as_posix()}"


def _acquire_lock(lock_path: Path, timeout_seconds: int) -> None:
    start = time.time()
    while True:
        if lock_path.exists():
            age = time.time() - lock_path.stat().st_mtime
            if age > max(10, timeout_seconds * 2):
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - start > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for lock {lock_path}")
            time.sleep(1)


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _suggest_proximal_params(trial: optuna.Trial, space: ProximalSearchSpace) -> dict[str, Any]:
    num_synapses = trial.suggest_int(
        "num_synapses",
        space.num_synapses_min,
        space.num_synapses_max,
        step=space.num_synapses_step,
    )
    max_weight = trial.suggest_float(
        "max_weight",
        space.max_weight_min,
        space.max_weight_max,
        log=space.max_weight_log,
    )
    conductance_threshold = trial.suggest_float(
        "conductance_threshold",
        space.conductance_threshold_min,
        space.conductance_threshold_max,
    )
    if space.independent_noise_rate_min is None or space.independent_noise_rate_max is None:
        raise ValueError("independent_noise_rate_min and independent_noise_rate_max must be set for range sampling.")
    independent_noise_rate = trial.suggest_float(
        "independent_noise_rate",
        space.independent_noise_rate_min,
        space.independent_noise_rate_max,
    )
    stdp_rate = trial.suggest_float(
        "stdp_rate",
        space.stdp_rate_min,
        space.stdp_rate_max,
        log=space.stdp_rate_log,
    )
    depression_potentiation_ratio = trial.suggest_float(
        "depression_potentiation_ratio",
        space.dp_ratio_min,
        space.dp_ratio_max,
    )
    baseline_rate = trial.suggest_float(
        "baseline_rate",
        space.baseline_rate_min,
        space.baseline_rate_max,
    )
    driven_rate = trial.suggest_float(
        "driven_rate",
        space.driven_rate_min,
        space.driven_rate_max,
    )
    concentration = trial.suggest_float(
        "concentration",
        space.concentration_min,
        space.concentration_max,
    )
    return {
        "num_synapses": num_synapses,
        "max_weight": max_weight,
        "conductance_threshold": conductance_threshold,
        "independent_noise_rate": independent_noise_rate,
        "stdp_rate": stdp_rate,
        "depression_potentiation_ratio": depression_potentiation_ratio,
        "baseline_rate": baseline_rate,
        "driven_rate": driven_rate,
        "concentration": concentration,
    }


def _average_window_steps(duration: int, average_window: float | int) -> int:
    if isinstance(average_window, float):
        return max(1, int(duration * average_window))
    return max(1, min(duration, int(average_window)))


def _normalize_synapses(num_synapses: int, max_synapses: int) -> float:
    return float(num_synapses / max_synapses)


def _make_objective(
    *,
    config_name: str,
    duration: int,
    num_neurons: int,
    average_window: float | int,
    space: ProximalSearchSpace,
) -> Any:
    def objective(trial: optuna.Trial) -> float:
        params = _suggest_proximal_params(trial, space)
        sim, cfg = get_proximal_experiment(
            config_name,
            num_synapses=params["num_synapses"],
            max_weight=params["max_weight"],
            conductance_threshold=params["conductance_threshold"],
            independent_noise_rate=params["independent_noise_rate"],
            stdp_rate=params["stdp_rate"],
            depression_potentiation_ratio=params["depression_potentiation_ratio"],
            baseline_rate=params["baseline_rate"],
            driven_rate=params["driven_rate"],
            concentration=params["concentration"],
            num_simulations=num_neurons,
        )
        results = sim.run(duration=duration, save_source_rates=False)
        results["sim"] = sim
        results["cfg"] = cfg
        window_steps = _average_window_steps(duration, average_window)
        window_start_sec = duration - window_steps
        steps_per_second = int(1 / sim.dt)
        window_start_step = window_start_sec * steps_per_second
        window_end_step = duration * steps_per_second

        averaged_weights = []
        spike_rates = []
        for ineuron in range(len(results["weights"])):
            proximal_weights = results["weights"][ineuron]["proximal"]
            averaged = np.mean(proximal_weights[-window_steps:], axis=0)
            averaged_weights.append(averaged.tolist())

            spike_times = results["spike_times"][ineuron]
            in_window = (spike_times >= window_start_step) & (spike_times < window_end_step)
            spike_count = int(np.sum(in_window))
            spike_rates.append(spike_count / window_steps)

        entropy = proximal_weight_entropy(results, average_window=average_window)
        num_inputs = results["weights"][0]["proximal"].shape[-1]
        max_entropy = float(np.log(num_inputs)) if num_inputs > 0 else 1.0
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
        synapse_norm = _normalize_synapses(
            cfg.synapses["proximal"].num_synapses,
            space.num_synapses_max,
        )
        score = entropy_norm * (1 - space.synapse_weight) + space.synapse_weight * (1.0 - synapse_norm)
        trial.set_user_attr("entropy", entropy)
        trial.set_user_attr("entropy_norm", entropy_norm)
        trial.set_user_attr("synapse_norm", (1 - synapse_norm))
        trial.set_user_attr("score", score)
        trial.set_user_attr("synapse_weight", space.synapse_weight)
        trial.set_user_attr("avg_window_seconds", window_steps)
        trial.set_user_attr("avg_window_start_sec", window_start_sec)
        trial.set_user_attr("avg_proximal_weights", averaged_weights)
        trial.set_user_attr("avg_spike_rate_hz", spike_rates)

        return float(score)

    return objective


def run_optuna_proximal(
    *,
    study_name: str,
    run_dir: Path,
    config_name: str,
    duration: int,
    num_neurons: int,
    n_trials: int,
    timeout: int | None,
    storage_timeout: int,
    average_window: float | int,
    seed: int | None,
    space: ProximalSearchSpace,
) -> tuple[Path, str]:
    if run_dir is None:
        raise ValueError("run_dir must be provided; create it once and share across workers.")
    run_dir, resolved_study_name = _resolve_run_dir(study_name, run_dir)
    storage_url = _build_storage_url(run_dir)
    if seed is not None:
        np.random.seed(seed)
        utils.rng = create_rng(seed)
        sampler = optuna.samplers.TPESampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler()

    lock_path = run_dir / ".optuna_storage.lock"
    _acquire_lock(lock_path, storage_timeout)
    try:
        storage = optuna.storages.RDBStorage(
            storage_url,
            engine_kwargs={"connect_args": {"timeout": storage_timeout}},
        )
        study = optuna.create_study(
            storage=storage,
            study_name=resolved_study_name,
            direction="minimize",
            sampler=sampler,
            load_if_exists=True,
        )
    finally:
        _release_lock(lock_path)

    run_metadata = {
        "study_name": resolved_study_name,
        "storage_url": storage_url,
        "config_name": config_name,
        "duration": duration,
        "num_neurons": num_neurons,
        "n_trials": n_trials,
        "timeout": timeout,
        "storage_timeout": storage_timeout,
        "average_window": average_window,
        "seed": seed,
        "search_space": space.__dict__,
    }
    _maybe_write_json(run_dir / "args.json", run_metadata)
    snapshot_path = run_dir / "repo.zip"
    if not snapshot_path.exists():
        save_repo_snapshot(snapshot_path, verbose=False)

    objective = _make_objective(
        config_name=config_name,
        duration=duration,
        num_neurons=num_neurons,
        average_window=average_window,
        space=space,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return run_dir, resolved_study_name
