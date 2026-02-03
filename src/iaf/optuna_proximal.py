from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
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
    num_synapses_min: int
    num_synapses_max: int
    num_synapses_step: int
    max_weight_min: float
    max_weight_max: float
    max_weight_log: bool
    conductance_threshold_min: float
    conductance_threshold_max: float
    independent_noise_rate_min: float | None
    independent_noise_rate_max: float | None
    stdp_rate_min: float
    stdp_rate_max: float
    stdp_rate_log: bool
    dp_ratio_min: float
    dp_ratio_max: float


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
    return {
        "num_synapses": num_synapses,
        "max_weight": max_weight,
        "conductance_threshold": conductance_threshold,
        "independent_noise_rate": independent_noise_rate,
        "stdp_rate": stdp_rate,
        "depression_potentiation_ratio": depression_potentiation_ratio,
    }


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
            num_simulations=num_neurons,
        )
        results = sim.run(duration=duration, save_source_rates=False)
        results["sim"] = sim
        results["cfg"] = cfg
        entropy = proximal_weight_entropy(results, average_window=average_window)
        trial.set_user_attr("entropy", entropy)
        return float(entropy)

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
    storage = optuna.storages.RDBStorage(
        storage_url,
        engine_kwargs={"connect_args": {"timeout": storage_timeout}},
    )

    if seed is not None:
        np.random.seed(seed)
        utils.rng = create_rng(seed)
        sampler = optuna.samplers.TPESampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(
        storage=storage,
        study_name=resolved_study_name,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

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
