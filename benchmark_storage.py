"""
Blob-storage persistence for benchmark runs.

Each run gets two blobs in the configured container:
  <run_id>/manifest.json   – metadata + last-known state (UI source of truth)
  <run_id>/logs.txt        – captured pod logs (written when run terminates)

The Container App's system-assigned managed identity authenticates via
DefaultAzureCredential. The MI must have role
"Storage Blob Data Contributor" on the storage account.

If the azure-storage-blob package is missing or the account env var is
unset, all functions become no-ops so the app keeps working without
persistence (useful for local dev).
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import threading
from typing import Any

log = logging.getLogger(__name__)

ACCOUNT = os.getenv("BENCHMARK_STORAGE_ACCOUNT", "benchmarkdatangjason")
CONTAINER = os.getenv("BENCHMARK_STORAGE_CONTAINER", "runs")

try:
    from azure.identity import DefaultAzureCredential  # type: ignore
    from azure.storage.blob import BlobServiceClient, ContentSettings  # type: ignore
    from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError  # type: ignore
    _SDK = True
except Exception:  # pragma: no cover
    _SDK = False
    DefaultAzureCredential = None  # type: ignore
    BlobServiceClient = None  # type: ignore
    ContentSettings = None  # type: ignore
    ResourceNotFoundError = Exception  # type: ignore
    ResourceExistsError = Exception  # type: ignore

_lock = threading.Lock()
_container_client = None
_init_attempted = False


def is_enabled() -> bool:
    return bool(_SDK and ACCOUNT)


def _client():
    """Return a (cached) ContainerClient, or None if storage is unavailable."""
    global _container_client, _init_attempted
    if not is_enabled():
        return None
    if _container_client is not None:
        return _container_client
    with _lock:
        if _container_client is not None:
            return _container_client
        if _init_attempted:
            return _container_client
        _init_attempted = True
        try:
            cred = DefaultAzureCredential()
            svc = BlobServiceClient(
                account_url=f"https://{ACCOUNT}.blob.core.windows.net",
                credential=cred,
            )
            cc = svc.get_container_client(CONTAINER)
            try:
                cc.create_container()
            except ResourceExistsError:
                pass
            except Exception as e:  # noqa: BLE001
                # Container probably already exists or MI lacks create rights;
                # we'll still try to use it.
                log.info("create_container skipped: %s", e)
            _container_client = cc
        except Exception as e:  # noqa: BLE001
            log.warning("Blob storage disabled (init failed): %s", e)
            _container_client = None
    return _container_client


# ---------- key helpers ----------
def _manifest_key(run_id: str) -> str:
    return f"{run_id}/manifest.json"


def _logs_key(run_id: str) -> str:
    return f"{run_id}/logs.txt"


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------- manifest ops ----------
def save_manifest(run_id: str, data: dict[str, Any]) -> None:
    cc = _client()
    if cc is None:
        return
    body = json.dumps(data, default=str, indent=2).encode("utf-8")
    try:
        cc.upload_blob(
            _manifest_key(run_id),
            body,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json")
            if ContentSettings else None,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("save_manifest(%s) failed: %s", run_id, e)


def load_manifest(run_id: str) -> dict[str, Any] | None:
    cc = _client()
    if cc is None:
        return None
    try:
        raw = cc.download_blob(_manifest_key(run_id)).readall()
        return json.loads(raw)
    except ResourceNotFoundError:
        return None
    except Exception as e:  # noqa: BLE001
        log.warning("load_manifest(%s) failed: %s", run_id, e)
        return None


def update_manifest(run_id: str, **fields: Any) -> dict[str, Any]:
    cur = load_manifest(run_id) or {"run_id": run_id}
    cur.update(fields)
    cur["updated_at"] = _now()
    save_manifest(run_id, cur)
    return cur


def list_manifests() -> list[dict[str, Any]]:
    cc = _client()
    if cc is None:
        return []
    out: list[dict[str, Any]] = []
    try:
        for blob in cc.list_blobs(name_starts_with=""):
            if not blob.name.endswith("/manifest.json"):
                continue
            try:
                raw = cc.download_blob(blob.name).readall()
                out.append(json.loads(raw))
            except Exception as e:  # noqa: BLE001
                log.warning("skipping blob %s: %s", blob.name, e)
    except Exception as e:  # noqa: BLE001
        log.warning("list_manifests failed: %s", e)
    return out


# ---------- log ops ----------
def save_logs(run_id: str, logs: str | None) -> None:
    if not logs:
        return
    cc = _client()
    if cc is None:
        return
    try:
        cc.upload_blob(
            _logs_key(run_id),
            logs.encode("utf-8"),
            overwrite=True,
            content_settings=ContentSettings(content_type="text/plain; charset=utf-8")
            if ContentSettings else None,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("save_logs(%s) failed: %s", run_id, e)


def load_logs(run_id: str) -> str | None:
    cc = _client()
    if cc is None:
        return None
    try:
        return cc.download_blob(_logs_key(run_id)).readall().decode("utf-8", errors="replace")
    except ResourceNotFoundError:
        return None
    except Exception as e:  # noqa: BLE001
        log.warning("load_logs(%s) failed: %s", run_id, e)
        return None


# ---------- delete ----------
def delete_run(run_id: str) -> None:
    cc = _client()
    if cc is None:
        return
    for key in (_manifest_key(run_id), _logs_key(run_id)):
        try:
            cc.delete_blob(key)
        except ResourceNotFoundError:
            pass
        except Exception as e:  # noqa: BLE001
            log.warning("delete_blob(%s) failed: %s", key, e)
