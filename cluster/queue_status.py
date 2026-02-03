from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sqlite3


def parse_args():
    parser = ArgumentParser(description="Show status for the SQLite task queue.")
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("cluster") / "queue.sqlite",
        help="Path to the SQLite queue file.",
    )
    parser.add_argument(
        "--show-failed",
        action="store_true",
        help="Show recent failed tasks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of failed tasks to show.",
    )
    parser.add_argument(
        "--requeue-failed",
        action="store_true",
        help="Move failed tasks back to pending.",
    )
    parser.add_argument(
        "--reset-attempts",
        action="store_true",
        help="Reset attempts to 0 when requeuing failed tasks.",
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


def main() -> None:
    args = parse_args()
    conn = sqlite3.connect(args.queue)
    ensure_schema(conn)

    if args.requeue_failed:
        if args.reset_attempts:
            conn.execute(
                """
                UPDATE tasks
                SET status = 'pending', attempts = 0, last_error = NULL, finished_at = NULL, started_at = NULL, worker_id = NULL
                WHERE status = 'failed'
                """
            )
        else:
            conn.execute(
                """
                UPDATE tasks
                SET status = 'pending', last_error = NULL, finished_at = NULL, started_at = NULL, worker_id = NULL
                WHERE status = 'failed'
                """
            )
        conn.commit()

    counts = conn.execute(
        "SELECT status, COUNT(*) FROM tasks GROUP BY status ORDER BY status"
    ).fetchall()

    total = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    print(f"Total tasks: {total}")
    for status, count in counts:
        print(f"{status}: {count}")

    if args.show_failed:
        failed = conn.execute(
            """
            SELECT id, attempts, last_error
            FROM tasks
            WHERE status = 'failed'
            ORDER BY finished_at DESC
            LIMIT ?
            """,
            (args.limit,),
        ).fetchall()

        if failed:
            print("\nRecent failed tasks:")
            for task_id, attempts, last_error in failed:
                print(f"- {task_id} (attempts {attempts}): {last_error}")
        else:
            print("\nNo failed tasks to show.")

    conn.close()


if __name__ == "__main__":
    main()
