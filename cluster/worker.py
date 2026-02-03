from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time


def parse_args():
    parser = ArgumentParser(description="Run queued tasks until the queue is empty or time runs out.")
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("cluster") / "queue.sqlite",
        help="Path to the SQLite queue file.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for task logs (defaults to <queue parent>/queue_logs).",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=10,
        help="Seconds to wait before polling the queue again when empty.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum attempts per task before marking it failed.",
    )
    parser.add_argument(
        "--walltime-seconds",
        type=int,
        default=None,
        help="Total walltime available for this worker. If unset, runs until queue is empty.",
    )
    parser.add_argument(
        "--stop-seconds-before",
        type=int,
        default=600,
        help="Stop this many seconds before walltime to exit cleanly.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


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


def claim_task(conn: sqlite3.Connection, worker_id: str, max_attempts: int) -> dict | None:
    conn.execute("BEGIN IMMEDIATE")
    row = conn.execute(
        """
        SELECT id, script, args_json, attempts
        FROM tasks
        WHERE status IN ('pending', 'failed') AND attempts < ?
        ORDER BY id
        LIMIT 1
        """,
        (max_attempts,),
    ).fetchone()

    if row is None:
        conn.execute("COMMIT")
        return None

    task_id, script, args_json, attempts = row
    conn.execute(
        """
        UPDATE tasks
        SET status = 'running', worker_id = ?, started_at = ?, attempts = attempts + 1
        WHERE id = ?
        """,
        (worker_id, now_iso(), task_id),
    )
    conn.execute("COMMIT")

    return {
        "id": task_id,
        "script": script,
        "args_json": args_json,
        "attempts": attempts + 1,
    }


def mark_task_success(conn: sqlite3.Connection, task_id: int) -> None:
    conn.execute(
        """
        UPDATE tasks
        SET status = 'done', finished_at = ?, last_error = NULL
        WHERE id = ?
        """,
        (now_iso(), task_id),
    )
    conn.commit()


def mark_task_failure(conn: sqlite3.Connection, task_id: int, attempts: int, max_attempts: int, error: str) -> None:
    status = "pending" if attempts < max_attempts else "failed"
    conn.execute(
        """
        UPDATE tasks
        SET status = ?, finished_at = ?, last_error = ?
        WHERE id = ?
        """,
        (status, now_iso(), error, task_id),
    )
    conn.commit()


def build_command(script: str, args_json: str, repo_root: Path) -> list[str]:
    args_list = json.loads(args_json)
    script_path = repo_root / script
    return [sys.executable, str(script_path), *args_list]


def should_stop(start_time: float, walltime_seconds: int | None, stop_seconds_before: int) -> bool:
    if walltime_seconds is None:
        return False
    deadline = start_time + walltime_seconds - stop_seconds_before
    return time.time() >= deadline


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    queue_path = args.queue
    log_dir = args.log_dir or (queue_path.parent / "queue_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    worker_id = os.environ.get("SGE_TASK_ID", "0")
    job_id = os.environ.get("JOB_ID", "local")
    worker_tag = f"job{job_id}_task{worker_id}_pid{os.getpid()}"

    conn = sqlite3.connect(queue_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    ensure_schema(conn)

    start_time = time.time()

    while True:
        if should_stop(start_time, args.walltime_seconds, args.stop_seconds_before):
            print("Stopping: nearing walltime limit.")
            break

        task = claim_task(conn, worker_tag, args.max_attempts)
        if task is None:
            remaining = conn.execute(
                """
                SELECT COUNT(*) FROM tasks
                WHERE status IN ('pending', 'failed') AND attempts < ?
                """,
                (args.max_attempts,),
            ).fetchone()[0]
            if remaining == 0:
                print("Queue is empty. Exiting.")
                break
            time.sleep(args.poll_seconds)
            continue

        task_id = task["id"]
        log_path = log_dir / f"task_{task_id}.log"
        command = build_command(task["script"], task["args_json"], repo_root)

        print(f"Starting task {task_id} (attempt {task['attempts']}).")
        try:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"\n[{now_iso()}] {worker_tag} starting task {task_id}\n")
                log_file.write("Command: " + " ".join(command) + "\n")
                log_file.flush()
                result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False)

            if result.returncode == 0:
                mark_task_success(conn, task_id)
                print(f"Task {task_id} completed.")
            else:
                error = f"Exit code {result.returncode}"
                mark_task_failure(conn, task_id, task["attempts"], args.max_attempts, error)
                print(f"Task {task_id} failed: {error}")
        except Exception as exc:
            error = f"Worker exception: {exc}"
            mark_task_failure(conn, task_id, task["attempts"], args.max_attempts, error)
            print(f"Task {task_id} failed: {error}")

    conn.close()


if __name__ == "__main__":
    main()
