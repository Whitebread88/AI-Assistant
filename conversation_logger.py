import atexit
import os
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any

from google.cloud import bigquery

BQ_DATASET = os.getenv("CONVERSATION_BQ_DATASET", "chatbot")
BQ_TABLE = os.getenv("CONVERSATION_BQ_TABLE", "conversations")
LOG_QUEUE_MAX_SIZE = int(os.getenv("CONVERSATION_LOG_QUEUE_SIZE", "5000"))
LOG_BATCH_SIZE = int(os.getenv("CONVERSATION_LOG_BATCH_SIZE", "50"))
LOG_BATCH_FLUSH_SECONDS = float(os.getenv("CONVERSATION_LOG_BATCH_FLUSH_SECONDS", "0.5"))


def _table_id() -> str:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
    if not project_id:
        return ""
    return f"{project_id}.{BQ_DATASET}.{BQ_TABLE}"


class ConversationLogger:
    def __init__(self) -> None:
        self.table_id = _table_id()
        self.enabled = bool(self.table_id)
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=LOG_QUEUE_MAX_SIZE)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._client: bigquery.Client | None = None

        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            atexit.register(self.stop)
        else:
            print("BigQuery conversation logging is disabled: missing GOOGLE_CLOUD_PROJECT/GCP_PROJECT")

    def log(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            # Never block chatbot responses on logging backpressure.
            print("BigQuery log queue is full. Dropping conversation event.")

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._flush_batch(self._drain_queue())

    def _get_client(self) -> bigquery.Client:
        if self._client is None:
            self._client = bigquery.Client()
        return self._client

    def _drain_queue(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        while len(rows) < LOG_BATCH_SIZE:
            try:
                rows.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return rows

    def _run(self) -> None:
        batch: list[dict[str, Any]] = []
        last_flush = time.monotonic()

        while not self._stop_event.is_set():
            timeout = max(0.0, LOG_BATCH_FLUSH_SECONDS - (time.monotonic() - last_flush))
            try:
                event = self._queue.get(timeout=timeout)
                batch.append(event)
            except queue.Empty:
                pass

            if len(batch) >= LOG_BATCH_SIZE or (
                batch and (time.monotonic() - last_flush) >= LOG_BATCH_FLUSH_SECONDS
            ):
                self._flush_batch(batch)
                batch = []
                last_flush = time.monotonic()

        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        try:
            errors = self._get_client().insert_rows_json(self.table_id, rows)
            if errors:
                print(f"Failed to insert conversation batch into BigQuery: {errors}")
        except Exception as exc:
            print(f"Unexpected BigQuery logging error: {exc}")


conversation_logger = ConversationLogger()


def build_conversation_event(
    session_id: str,
    user_message: str,
    assistant_answer: str,
    categories: list[str],
    endpoint: str,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "user_message": user_message,
        "assistant_answer": assistant_answer,
        "categories": categories,
        "endpoint": endpoint,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
