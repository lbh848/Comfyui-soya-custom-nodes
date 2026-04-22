"""Global store for Ray futures keyed by task_id."""

import threading

_store = {}
_lock = threading.Lock()


def put(task_id, entry):
    with _lock:
        _store[task_id] = entry


def get(task_id):
    with _lock:
        return _store.get(task_id)


def pop(task_id):
    with _lock:
        return _store.pop(task_id, None)


def keys():
    with _lock:
        return list(_store.keys())
