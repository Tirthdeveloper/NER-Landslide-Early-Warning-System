"""
offline_support.py
------------------
Offline / low-network support for the NER Landslide project.

What it does:
- Detects basic internet connectivity.
- Queues citizen/field reports locally as JSON.
- Queues alerts locally when network delivery is unavailable.
- Reads pending queue items.
- Marks queued items as synced after successful delivery.

This provides store-and-forward support. It is not a full browser PWA cache.
"""

import json
import os
import socket
import uuid
from datetime import datetime

QUEUE_DIR = "data/offline_queue"
REPORT_QUEUE_FILE = os.path.join(QUEUE_DIR, "pending_reports.json")
ALERT_QUEUE_FILE = os.path.join(QUEUE_DIR, "pending_alerts.json")

os.makedirs(QUEUE_DIR, exist_ok=True)


def is_online(timeout=2):
    """Return True when basic internet connectivity appears available."""
    try:
        connection = socket.create_connection(
            ("1.1.1.1", 53),
            timeout=timeout
        )
        connection.close()
        return True
    except OSError:
        return False


def _read_queue(file_path):
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_queue(file_path, items):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(
            items,
            file,
            indent=2,
            ensure_ascii=False,
            default=str
        )


def queue_report(report_data):
    """Store a field/citizen report locally for later synchronization."""
    items = _read_queue(REPORT_QUEUE_FILE)

    item = {
        "queue_id": str(uuid.uuid4()),
        "queued_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PENDING",
        "data": report_data
    }

    items.append(item)
    _write_queue(REPORT_QUEUE_FILE, items)
    return item


def queue_alert(alert_data):
    """Store an alert locally for later synchronization/delivery."""
    items = _read_queue(ALERT_QUEUE_FILE)

    item = {
        "queue_id": str(uuid.uuid4()),
        "queued_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PENDING",
        "data": alert_data
    }

    items.append(item)
    _write_queue(ALERT_QUEUE_FILE, items)
    return item


def get_pending_reports():
    return [
        item for item in _read_queue(REPORT_QUEUE_FILE)
        if item.get("status") == "PENDING"
    ]


def get_pending_alerts():
    return [
        item for item in _read_queue(ALERT_QUEUE_FILE)
        if item.get("status") == "PENDING"
    ]


def get_offline_status():
    """Return connectivity and pending queue counts."""
    return {
        "online": is_online(),
        "pending_reports": len(get_pending_reports()),
        "pending_alerts": len(get_pending_alerts())
    }


def mark_report_synced(queue_id):
    items = _read_queue(REPORT_QUEUE_FILE)

    for item in items:
        if item.get("queue_id") == queue_id:
            item["status"] = "SYNCED"
            item["synced_at"] = datetime.now().isoformat(timespec="seconds")
            break

    _write_queue(REPORT_QUEUE_FILE, items)


def mark_alert_synced(queue_id):
    items = _read_queue(ALERT_QUEUE_FILE)

    for item in items:
        if item.get("queue_id") == queue_id:
            item["status"] = "SYNCED"
            item["synced_at"] = datetime.now().isoformat(timespec="seconds")
            break

    _write_queue(ALERT_QUEUE_FILE, items)


if __name__ == "__main__":
    status = get_offline_status()

    print("\nOFFLINE / LOW-NETWORK STATUS")
    print("============================")
    print("Online:", status["online"])
    print("Pending Reports:", status["pending_reports"])
    print("Pending Alerts:", status["pending_alerts"])