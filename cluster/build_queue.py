from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
import sqlite3


DEFAULT_DISTAL_DP_RATIOS = [1.0, 1.025, 1.05, 1.075, 1.1]
DEFAULT_EDGE_PROBABILITIES = [0.5, 0.75, 1.0]


def parse_args():
    parser = ArgumentParser(description="Build a SQLite task queue for cluster workers.")
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("cluster") / "queue.sqlite",
        help="Path to the SQLite queue file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["correlated", "hofer"],
        required=True,
        help="Which experiment script to target.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Experiment config name (e.g., correlated, hofer_replacement).",
    )
    parser.add_argument(
        "--distal-dp-ratios",
        type=float,
        nargs="+",
        default=DEFAULT_DISTAL_DP_RATIOS,
        help="Distal depression-potentiation ratios.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repeats per ratio.",
    )
    parser.add_argument(
        "--num-neurons",
        type=int,
        default=3,
        help="Number of neurons per simulation.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=2400,
        help="Simulation duration in seconds.",
    )
    parser.add_argument(
        "--no-distal",
        action="store_true",
        help="Run without distal dendrites.",
    )
    parser.add_argument(
        "--save-source-rates",
        action="store_true",
        help="Save source rates in outputs.",
    )
    parser.add_argument(
        "--exp-folder",
        type=str,
        default=None,
        help="Experiment folder name (prefix, timestamp added by scripts).",
    )
    parser.add_argument(
        "--edge-probabilities",
        type=float,
        nargs="+",
        default=DEFAULT_EDGE_PROBABILITIES,
        help="Edge probabilities (hofer mode only).",
    )
    parser.add_argument(
        "--independent-noise-rate",
        type=float,
        default=None,
        help="Independent noise rate (hofer mode only).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear any existing tasks in the queue before creating new ones.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print how many tasks would be created and exit.",
    )
    return parser.parse_args()


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            script TEXT NOT NULL,
            args_json TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            worker_id TEXT,
            started_at TEXT,
            finished_at TEXT,
            last_error TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def build_base_args(args) -> list[str]:
    cli_args = [
        "--config",
        args.config,
        "--repeats",
        str(args.repeats),
        "--duration",
        str(args.duration),
        "--num_neurons",
        str(args.num_neurons),
        "--distal_dp_ratios",
        *[str(value) for value in args.distal_dp_ratios],
    ]

    if args.exp_folder:
        cli_args += ["--exp_folder", args.exp_folder]
    if args.no_distal:
        cli_args.append("--no_distal")
    if args.save_source_rates:
        cli_args.append("--save_source_rates")

    if args.mode == "hofer":
        cli_args += ["--edge_probabilities", *[str(value) for value in args.edge_probabilities]]
        if args.independent_noise_rate is not None:
            cli_args += ["--independent_noise_rate", str(args.independent_noise_rate)]

    return cli_args


def main() -> None:
    args = parse_args()

    script = "scripts/iaf_correlation.py" if args.mode == "correlated" else "scripts/iaf_hofer_reconstruction.py"
    base_args = build_base_args(args)
    task_count = len(args.distal_dp_ratios) * args.repeats

    if args.dry_run:
        print(f"Would create {task_count} tasks in {args.queue}")
        return

    args.queue.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.queue)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    ensure_schema(conn)

    if args.reset:
        conn.execute("DELETE FROM tasks")
        conn.execute("DELETE FROM meta")
    else:
        existing = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        if existing > 0:
            conn.close()
            raise SystemExit(f"Queue {args.queue} already has {existing} tasks. Use --reset or delete the file.")

    created_at = datetime.now().isoformat(timespec="seconds")
    meta_items = {
        "created_at": created_at,
        "mode": args.mode,
        "config": args.config,
        "repeats": str(args.repeats),
        "duration": str(args.duration),
        "num_neurons": str(args.num_neurons),
        "task_count": str(task_count),
        "exp_folder": args.exp_folder or "",
    }
    for key, value in meta_items.items():
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    task_rows = []
    for dp_ratio_index in range(len(args.distal_dp_ratios)):
        for repeat in range(args.repeats):
            task_args = base_args + ["--dp_ratio_index", str(dp_ratio_index), "--repeat", str(repeat)]
            task_rows.append((script, json.dumps(task_args), "pending"))

    conn.executemany(
        "INSERT INTO tasks (script, args_json, status) VALUES (?, ?, ?)",
        task_rows,
    )
    conn.commit()
    conn.close()

    print(f"Created {task_count} tasks in {args.queue}")


if __name__ == "__main__":
    main()
