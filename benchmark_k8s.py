"""
Kubernetes orchestration for vLLM benchmark runs.

Workflow per request:
  1. Create a Deployment that hosts the chosen model with `vllm serve`
     on the requested GPU SKU (tensor-parallel = number of GPUs).
  2. Create a ClusterIP Service in front of the Deployment.
  3. Create a Job that runs `vllm bench serve` against that Service.
  4. Job has an initContainer that waits for /health before benchmarking.

All resources for a single run share the same `run-id` label so they can
be cleaned up together.
"""
from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Any

import benchmark_storage

log = logging.getLogger(__name__)

# kubernetes is optional at import time so the calculator still runs locally
try:
    from kubernetes import client, config  # type: ignore
    from kubernetes.client.rest import ApiException  # type: ignore
    _K8S_AVAILABLE = True
except Exception:  # pragma: no cover
    client = None  # type: ignore
    config = None  # type: ignore
    ApiException = Exception  # type: ignore
    _K8S_AVAILABLE = False


# ---------- configuration ----------
NAMESPACE = os.getenv("BENCHMARK_NAMESPACE", "default")
GPU_RESOURCE = os.getenv("BENCHMARK_GPU_RESOURCE", "nvidia.com/gpu")
VLLM_IMAGE = os.getenv("BENCHMARK_VLLM_IMAGE", "vllm/vllm-openai:latest")
HF_SECRET_NAME = os.getenv("BENCHMARK_HF_SECRET", "hf-secret")
HF_SECRET_KEY = os.getenv("BENCHMARK_HF_SECRET_KEY", "hf_api_token")
DATASET_PVC = os.getenv("BENCHMARK_DATASET_PVC", "")  # optional: PVC with ShareGPT json

# Wall-clock deadline for a benchmark run (covers GPU node provisioning,
# model download, and the actual benchmark). After this elapses the Job
# is failed by Kubernetes with reason=DeadlineExceeded.
JOB_ACTIVE_DEADLINE_SECONDS = int(os.getenv("BENCHMARK_JOB_DEADLINE", "3600"))

# Max times the vLLM serving container may restart before the init container
# gives up and fails the benchmark Job (with the vLLM error in the
# termination message).
MAX_VLLM_RESTARTS = int(os.getenv("BENCHMARK_MAX_VLLM_RESTARTS", "3"))

# ServiceAccount the benchmark Job pod runs as. Needs `get,list pods` in
# the namespace so the init container can poll vLLM pod restart count via
# the in-cluster K8s API. Provided by k8s/rbac.yaml.
BENCHMARK_SERVICE_ACCOUNT = os.getenv("BENCHMARK_SERVICE_ACCOUNT", "llm-calculator")

# Map our GPU DB names -> Azure VM instance type label values.
# AKS labels nodes with `node.kubernetes.io/instance-type=Standard_<sku>`.
GPU_SKU_INSTANCE_TYPE = {
    "NC40ads H100 v5":  "Standard_NC40ads_H100_v5",
    "NC80adis H100 v5": "Standard_NC80adis_H100_v5",
}

GPU_NUM_GPUS = {
    "NC40ads H100 v5":  1,
    "NC80adis H100 v5": 2,
}


# ---------- helpers ----------
def _slug(s: str) -> str:
    """RFC1123-friendly slug."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:40] or "x"


def _load_kube():
    if not _K8S_AVAILABLE:
        raise RuntimeError(
            "kubernetes package is not installed. "
            "Add `kubernetes` to requirements and rebuild."
        )
    # 1. In-cluster (when running in AKS itself)
    try:
        config.load_incluster_config()
        return
    except Exception:
        pass

    # 2. Base64-encoded kubeconfig in env (Container Apps / Functions / etc.)
    kubeconfig_b64 = os.getenv("KUBECONFIG_B64")
    if kubeconfig_b64:
        import base64
        import tempfile
        # Strip whitespace/newlines that secret stores sometimes inject and
        # re-pad so callers can paste either standard or raw base64.
        cleaned = "".join(kubeconfig_b64.split())
        cleaned += "=" * (-len(cleaned) % 4)
        try:
            decoded = base64.b64decode(cleaned, validate=False)
        except Exception as e:
            raise RuntimeError(
                f"KUBECONFIG_B64 is not valid base64 ({e}). "
                "Re-create the secret with `base64 -w 0 < kubeconfig`."
            ) from e
        path = os.path.join(tempfile.gettempdir(), "kubeconfig")
        with open(path, "wb") as f:
            f.write(decoded)
        config.load_kube_config(config_file=path)
        return

    # 3. Plain KUBECONFIG path or default ~/.kube/config (local dev)
    config.load_kube_config()


def _hf_env():
    """Return env list pulling HF_TOKEN from the configured Secret (if it exists)."""
    if not HF_SECRET_NAME:
        return []
    return [
        client.V1EnvVar(
            name="HF_TOKEN",
            value_from=client.V1EnvVarSource(
                secret_key_ref=client.V1SecretKeySelector(
                    name=HF_SECRET_NAME,
                    key=HF_SECRET_KEY,
                    optional=True,
                )
            ),
        ),
        client.V1EnvVar(
            name="HUGGING_FACE_HUB_TOKEN",
            value_from=client.V1EnvVarSource(
                secret_key_ref=client.V1SecretKeySelector(
                    name=HF_SECRET_NAME,
                    key=HF_SECRET_KEY,
                    optional=True,
                )
            ),
        ),
    ]


def _node_selector(gpu_sku: str) -> dict[str, str] | None:
    inst = GPU_SKU_INSTANCE_TYPE.get(gpu_sku)
    if not inst:
        return None
    return {"node.kubernetes.io/instance-type": inst}


def _build_serve_args(payload: dict[str, Any], svc_url: str) -> list[str]:
    """Translate the modal payload into `vllm bench serve` args."""
    model = payload["model"]
    tokenizer = payload.get("tokenizer") or model
    dataset = payload["dataset_name"]
    args = [
        "vllm", "bench", "serve",
        "--base-url", svc_url,
        "--model", model,
        "--trust-remote-code",
        "--tokenizer", tokenizer,
        "--dataset-name", dataset,
        "--num-prompts", str(int(payload.get("num_prompts", 1000))),
        "--max-concurrency", str(int(payload.get("max_concurrency", 32))),
        "--seed", str(int(payload.get("seed", 42))),
    ]
    if dataset == "random":
        args += [
            "--random-input-len", str(int(payload.get("random_input_len", 1024))),
            "--random-output-len", str(int(payload.get("random_output_len", 128))),
            "--random-prefix-len", str(int(payload.get("random_prefix_len", 0))),
            "--random-range-ratio", str(float(payload.get("random_range_ratio", 0.0))),
        ]
    elif dataset == "sharegpt":
        # Always use the in-pod path; the initContainer downloads the file here.
        args += [
            "--dataset-path",
            "/data/ShareGPT_V3_unfiltered_cleaned_split.json",
        ]
    return args


# ---------- resource builders ----------
def _build_deployment(run_id: str, model: str, n_gpus: int,
                      gpu_sku: str, labels: dict):
    name = f"vllm-{run_id}"
    container = client.V1Container(
        name="vllm",
        image=VLLM_IMAGE,
        command=["/bin/sh", "-c"],
        args=[
            f"vllm serve {model} "
            # f"--host 0.0.0.0 --port 8000 "
            f"--tensor-parallel-size {n_gpus} "
            f"--trust-remote-code "
            f"--enable-prefix-caching "
            f"--gpu-memory-utilization=0.95 "
            f"--enable-chunked-prefill "
            f"--enforce-eager"
        ],
        ports=[client.V1ContainerPort(container_port=8000)],
        env=_hf_env(),
        resources=client.V1ResourceRequirements(
            limits={GPU_RESOURCE: str(n_gpus)},
            requests={GPU_RESOURCE: str(n_gpus)},
        ),
        readiness_probe=client.V1Probe(
            http_get=client.V1HTTPGetAction(path="/health", port=8000),
            initial_delay_seconds=60,
            period_seconds=10,
            timeout_seconds=5,
            failure_threshold=60,
        ),
        volume_mounts=[
            client.V1VolumeMount(name="dshm", mount_path="/dev/shm"),
        ],
    )
    pod_spec = client.V1PodSpec(
        containers=[container],
        node_selector=_node_selector(gpu_sku),
        restart_policy="Always",
        volumes=[
            client.V1Volume(
                name="dshm",
                empty_dir=client.V1EmptyDirVolumeSource(medium="Memory"),
            )
        ],
    )
    return client.V1Deployment(
        metadata=client.V1ObjectMeta(name=name, labels=labels, namespace=NAMESPACE),
        spec=client.V1DeploymentSpec(
            replicas=1,
            selector=client.V1LabelSelector(match_labels={"app": name}),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={**labels, "app": name}),
                spec=pod_spec,
            ),
        ),
    )


def _build_service(run_id: str, labels: dict):
    name = f"vllm-{run_id}"
    return client.V1Service(
        metadata=client.V1ObjectMeta(name=name, labels=labels, namespace=NAMESPACE),
        spec=client.V1ServiceSpec(
            selector={"app": name},
            ports=[client.V1ServicePort(port=8000, target_port=8000, name="http")],
            type="ClusterIP",
        ),
    )


def _build_benchmark_job(run_id: str, payload: dict, labels: dict):
    name = f"bench-{run_id}"
    svc_url = f"http://vllm-{run_id}.{NAMESPACE}.svc.cluster.local:8000"
    bench_args = _build_serve_args(payload, svc_url)

    is_sharegpt = (payload.get("dataset_name") or "").lower() == "sharegpt"
    dataset_dir = "/data"
    sharegpt_file = "ShareGPT_V3_unfiltered_cleaned_split.json"

    # initContainer waits for /health AND watches the vLLM Deployment's pod
    # for crash-loops. If the vLLM container restarts more than
    # MAX_VLLM_RESTARTS times, the init container writes vLLM's last
    # terminated message to /dev/termination-log and exits 1, which fails
    # the Job (backoff_limit=0). Total /health wait: 354 * 10s = 3540s.
    #
    # Dynamic values are injected as shell variables at the top so the
    # rest of the script is plain sh/jq with no Python f-string escaping.
    deploy_label = f"app=vllm-{run_id}"
    wait_cmd = (
        f'SVC_URL="{svc_url}"; '
        f'LABEL="{deploy_label}"; '
        f'MAX_RESTARTS={MAX_VLLM_RESTARTS}; '
    ) + r"""
apk add --no-cache curl jq >/dev/null 2>&1 || true
set -u
API=https://kubernetes.default.svc
TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
NS=$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace)
CACERT=/var/run/secrets/kubernetes.io/serviceaccount/ca.crt
echo "Waiting for vLLM at $SVC_URL/health (max $MAX_RESTARTS vllm restarts)"
for i in $(seq 1 354); do
  if curl -fsS "$SVC_URL/health" >/dev/null 2>&1; then
    echo 'vLLM is ready'
    if [ -n "${DATASET_URL:-}" ]; then
      echo "Downloading dataset from $DATASET_URL"
      curl -fL --retry 5 --retry-delay 5 -o "$DATASET_PATH" "$DATASET_URL" || {
        MSG="Failed to download dataset from $DATASET_URL"
        echo "$MSG"
        echo "$MSG" > /dev/termination-log
        exit 1
      }
      ls -lh "$DATASET_PATH"
    fi
    exit 0
  fi
  POD_JSON=$(curl -fsS --cacert "$CACERT" -H "Authorization: Bearer $TOKEN" \
    "$API/api/v1/namespaces/$NS/pods?labelSelector=$LABEL" 2>/dev/null || echo '')
  RESTARTS=0
  if [ -n "$POD_JSON" ]; then
    RESTARTS=$(echo "$POD_JSON" | jq -r '[.items[].status.containerStatuses[]?|select(.name=="vllm")|.restartCount] | max // 0')
  fi
  if [ "$RESTARTS" -ge "$MAX_RESTARTS" ]; then
    LAST=$(echo "$POD_JSON" | jq -r '[.items[].status.containerStatuses[]?|select(.name=="vllm")|.lastState.terminated]|map(select(.))|.[0]|if . then "exit \(.exitCode) (\(.reason // "Error")): \(.message // "")" else "no termination details available" end')
    MSG="vLLM serving pod failed to start after $MAX_RESTARTS restarts. Last container error: $LAST"
    echo "$MSG"
    echo "$MSG" > /dev/termination-log
    exit 1
  fi
  echo "attempt $i... (vllm restarts=$RESTARTS)"
  sleep 10
done
MSG="Timed out waiting for vLLM to become ready. This is usually caused by GPU capacity constraints for the requested SKU (no node could be provisioned in time, or the model took too long to download / load on a freshly provisioned node). Please try again later or pick a different GPU SKU."
echo "$MSG"
echo "$MSG" > /dev/termination-log
exit 1
"""

    init_env: list = []
    init_volume_mounts = []
    bench_volume_mounts = []
    pod_volumes = []
    if is_sharegpt:
        init_env.append(
            client.V1EnvVar(
                name="DATASET_URL",
                value=os.getenv(
                    "BENCHMARK_SHAREGPT_URL",
                    "https://huggingface.co/datasets/anon8231489123/"
                    "ShareGPT_Vicuna_unfiltered/resolve/main/"
                    "ShareGPT_V3_unfiltered_cleaned_split.json",
                ),
            )
        )
        init_env.append(
            client.V1EnvVar(
                name="DATASET_PATH",
                value=f"{dataset_dir}/{sharegpt_file}",
            )
        )
        init_volume_mounts.append(
            client.V1VolumeMount(name="dataset", mount_path=dataset_dir)
        )
        bench_volume_mounts.append(
            client.V1VolumeMount(name="dataset", mount_path=dataset_dir)
        )
        pod_volumes.append(
            client.V1Volume(
                name="dataset",
                empty_dir=client.V1EmptyDirVolumeSource(size_limit="4Gi"),
            )
        )

    init_container = client.V1Container(
        name="wait-for-vllm",
        # alpine (not curlimages/curl) so we can `apk add jq` at runtime.
        # curlimages/curl runs as non-root and can't install packages.
        image="alpine:3.19",
        command=["/bin/sh", "-c", wait_cmd],
        env=init_env or None,
        volume_mounts=init_volume_mounts or None,
    )

    bench_container = client.V1Container(
        name="benchmark",
        image=VLLM_IMAGE,
        command=bench_args[:1],   # "vllm"
        args=bench_args[1:],
        env=_hf_env(),
        volume_mounts=bench_volume_mounts or None,
    )

    annotations = {
        "llm-calculator/requestor": str(payload.get("requestor") or "unknown"),
        "llm-calculator/model": str(payload.get("model") or ""),
        "llm-calculator/gpu-sku": str(payload.get("gpu_sku") or ""),
        "llm-calculator/dataset": str(payload.get("dataset_name") or ""),
    }
    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name=name, labels=labels, annotations=annotations, namespace=NAMESPACE
        ),
        spec=client.V1JobSpec(
            backoff_limit=0,
            # Keep the Job around well after completion so the status
            # poller has time to observe the terminal state and persist
            # it to blob storage before Kubernetes garbage-collects it.
            ttl_seconds_after_finished=300,
            active_deadline_seconds=JOB_ACTIVE_DEADLINE_SECONDS,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels=labels),
                spec=client.V1PodSpec(
                    init_containers=[init_container],
                    containers=[bench_container],
                    restart_policy="Never",
                    # Needed so the init container can poll vLLM pod
                    # restart count via the in-cluster K8s API.
                    service_account_name=BENCHMARK_SERVICE_ACCOUNT,
                    volumes=pod_volumes or None,
                ),
            ),
        ),
    )
    return job


# ---------- public API ----------
def submit_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Validate the payload, then create Deployment + Service + Job.
    Returns the names of created resources.
    """
    # ---- validate ----
    required = ("model", "dataset_name", "gpu_sku")
    missing = [k for k in required if not payload.get(k)]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    gpu_sku = payload["gpu_sku"]
    if gpu_sku not in GPU_NUM_GPUS:
        raise ValueError(
            f"Benchmarks are only supported on H100 SKUs. Got '{gpu_sku}'."
        )
    dataset = payload["dataset_name"]
    if dataset not in ("random", "sharegpt"):
        raise ValueError(f"dataset_name must be 'random' or 'sharegpt', got '{dataset}'")

    _load_kube()

    n_gpus = GPU_NUM_GPUS[gpu_sku]
    run_id = f"{_slug(payload['model'].split('/')[-1])}-{uuid.uuid4().hex[:6]}"
    labels = {
        "app.kubernetes.io/managed-by": "llm-calculator",
        "llm-calculator/run-id": run_id,
        "llm-calculator/component": "benchmark",
    }

    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()
    batch_v1 = client.BatchV1Api()

    # Create the Job first so we can use its UID as an OwnerReference on
    # the vLLM Deployment + Service. When the Job is TTL-deleted (10 min
    # after completion), Kubernetes garbage-collects the Deployment + Service
    # too, so the vLLM pod (and its GPU) is freed automatically once no
    # benchmark is running against it.
    job = _build_benchmark_job(run_id, payload, labels)
    created_job = batch_v1.create_namespaced_job(NAMESPACE, job)

    owner = client.V1OwnerReference(
        api_version="batch/v1",
        kind="Job",
        name=created_job.metadata.name,
        uid=created_job.metadata.uid,
        block_owner_deletion=False,
        controller=False,
    )

    deploy = _build_deployment(run_id, payload["model"], n_gpus, gpu_sku, labels)
    deploy.metadata.owner_references = [owner]
    svc = _build_service(run_id, labels)
    svc.metadata.owner_references = [owner]

    apps_v1.create_namespaced_deployment(NAMESPACE, deploy)
    core_v1.create_namespaced_service(NAMESPACE, svc)

    submitted_at = (
        created_job.metadata.creation_timestamp.isoformat()
        if created_job.metadata.creation_timestamp else None
    )

    # Persist initial manifest so the run survives Job TTL deletion.
    benchmark_storage.save_manifest(run_id, {
        "run_id": run_id,
        "namespace": NAMESPACE,
        "job_name": created_job.metadata.name,
        "deployment": deploy.metadata.name,
        "service": svc.metadata.name,
        "requestor": str(payload.get("requestor") or "unknown"),
        "model": str(payload.get("model") or ""),
        "gpu_sku": str(payload.get("gpu_sku") or ""),
        "dataset": str(payload.get("dataset_name") or ""),
        "submitted_at": submitted_at,
        "state": "provisioning",
        "payload": payload,
        "updated_at": benchmark_storage._now(),  # type: ignore[attr-defined]
    })

    return {
        "run_id": run_id,
        "namespace": NAMESPACE,
        "deployment": deploy.metadata.name,
        "service": svc.metadata.name,
        "job": created_job.metadata.name,
        "command": " ".join(_build_serve_args(payload,
                            f"http://{svc.metadata.name}:8000")),
    }


def _derive_state(job_status, deploy_ready: bool) -> str:
    """Derive a UI-friendly state from Job + Deployment status.

    Possible values: provisioning | running | success | failed | unknown.
    Runs are only considered failed when the Job itself has failed (e.g.
    after ``activeDeadlineSeconds`` elapses). Pods that are merely waiting
    for a GPU node to come up are still ``provisioning``.
    """
    if job_status is None:
        return "unknown"
    if (job_status.succeeded or 0) > 0:
        return "success"
    if (job_status.failed or 0) > 0:
        return "failed"
    # Job exists, not terminated yet.
    if (job_status.active or 0) > 0 and deploy_ready:
        return "running"
    return "provisioning"


def _job_metadata(job) -> dict[str, Any]:
    md = job.metadata
    ann = md.annotations or {}
    return {
        "run_id": (md.labels or {}).get("llm-calculator/run-id") or md.name.replace("bench-", "", 1),
        "job_name": md.name,
        "namespace": md.namespace,
        "requestor": ann.get("llm-calculator/requestor", "unknown"),
        "model": ann.get("llm-calculator/model", ""),
        "gpu_sku": ann.get("llm-calculator/gpu-sku", ""),
        "dataset": ann.get("llm-calculator/dataset", ""),
        "submitted_at": md.creation_timestamp.isoformat() if md.creation_timestamp else None,
    }


def _pod_for_job(core_v1, job_name: str):
    """Return the most recent Pod created by the named Job, or None."""
    pods = core_v1.list_namespaced_pod(
        NAMESPACE, label_selector=f"job-name={job_name}"
    )
    if not pods.items:
        return None
    return sorted(
        pods.items,
        key=lambda p: p.metadata.creation_timestamp or 0,
        reverse=True,
    )[0]


def list_runs() -> list[dict[str, Any]]:
    """List all benchmark runs (most recent first).

    Merges live Kubernetes Jobs with persisted blob manifests so that runs
    whose k8s resources were already TTL-deleted still appear in the UI.
    """
    runs_by_id: dict[str, dict[str, Any]] = {}

    # 1. Persisted manifests (the durable history).
    for m in benchmark_storage.list_manifests():
        rid = m.get("run_id")
        if not rid:
            continue
        runs_by_id[rid] = {
            "run_id": rid,
            "job_name": m.get("job_name"),
            "namespace": m.get("namespace") or NAMESPACE,
            "requestor": m.get("requestor", "unknown"),
            "model": m.get("model", ""),
            "gpu_sku": m.get("gpu_sku", ""),
            "dataset": m.get("dataset", ""),
            "submitted_at": m.get("submitted_at"),
            "state": m.get("state", "unknown"),
        }

    # 2. Overlay live state from the cluster (authoritative while it exists).
    try:
        _load_kube()
        batch_v1 = client.BatchV1Api()
        apps_v1 = client.AppsV1Api()

        label_selector = (
            "app.kubernetes.io/managed-by=llm-calculator,"
            "llm-calculator/component=benchmark"
        )
        jobs = batch_v1.list_namespaced_job(NAMESPACE, label_selector=label_selector)
        deploys = apps_v1.list_namespaced_deployment(
            NAMESPACE, label_selector=label_selector
        )
        deploy_ready: dict[str, bool] = {}
        for d in deploys.items:
            rid = (d.metadata.labels or {}).get("llm-calculator/run-id")
            if rid:
                deploy_ready[rid] = (d.status.ready_replicas or 0) >= (
                    d.spec.replicas or 1
                )

        for j in jobs.items:
            meta = _job_metadata(j)
            rid = meta["run_id"]
            meta["state"] = _derive_state(j.status, deploy_ready.get(rid, False))
            prev = runs_by_id.get(rid, {})
            runs_by_id[rid] = {**prev, **meta}

            # Persist terminal states immediately so the run keeps its
            # final state after the Job is TTL-deleted, even if the user
            # never opens the per-run detail page (which calls get_status).
            if (
                meta["state"] in ("success", "failed")
                and prev.get("state") not in ("success", "failed")
            ):
                try:
                    update_fields: dict[str, Any] = {
                        "state": meta["state"],
                        "completed_at": (
                            benchmark_storage._now()  # type: ignore[attr-defined]
                        ),
                    }
                    # Extract a meaningful error message from the Job's
                    # Failed condition (DeadlineExceeded, BackoffLimitExceeded,
                    # init container exit 1 from the /health wait loop, ...).
                    if meta["state"] == "failed":
                        for cond in (j.status.conditions or []):
                            if cond.type == "Failed" and cond.status == "True":
                                reason = cond.reason or "Failed"
                                msg = cond.message or ""
                                if reason == "DeadlineExceeded":
                                    update_fields["error"] = (
                                        f"Benchmark exceeded the "
                                        f"{JOB_ACTIVE_DEADLINE_SECONDS}s "
                                        f"deadline and was terminated by "
                                        f"Kubernetes. {msg}"
                                    ).strip()
                                else:
                                    update_fields["error"] = (
                                        f"{reason}: {msg}".strip(": ").strip()
                                    )
                                break
                    benchmark_storage.update_manifest(rid, **update_fields)
                except Exception as e:  # noqa: BLE001
                    log.warning("failed to persist terminal state for %s: %s", rid, e)
    except Exception:  # noqa: BLE001
        # If the k8s API is unreachable we still return the persisted view.
        pass

    runs = list(runs_by_id.values())
    runs.sort(key=lambda r: r.get("submitted_at") or "", reverse=True)
    return runs


def get_status(run_id: str) -> dict[str, Any]:
    """Detailed status for a previously submitted run, including logs/error.

    Live data from Kubernetes is preferred while it exists. Once the Job has
    been TTL-deleted, the persisted blob manifest + logs are returned instead
    so the UI keeps showing the run's final state and output.
    """
    job_name = f"bench-{run_id}"
    deploy_name = f"vllm-{run_id}"
    persisted = benchmark_storage.load_manifest(run_id) or {}

    out: dict[str, Any] = {"run_id": run_id, "namespace": NAMESPACE}
    # Seed with persisted metadata so callers always have model/sku/etc.
    for key in ("requestor", "model", "gpu_sku", "dataset", "submitted_at",
                "job_name", "deployment", "service"):
        if persisted.get(key):
            out[key] = persisted[key]

    # Try to load kubeconfig. If it fails (e.g. running outside the cluster
    # without a valid kubeconfig), skip all live-cluster lookups and serve
    # the persisted view from blob storage instead.
    batch_v1 = None
    apps_v1 = None
    core_v1 = None
    try:
        _load_kube()
        batch_v1 = client.BatchV1Api()
        apps_v1 = client.AppsV1Api()
        core_v1 = client.CoreV1Api()
    except Exception as e:  # noqa: BLE001
        log.warning("k8s config unavailable, using persisted state only: %s", e)

    job = None
    job_missing = False  # True only when the API confirmed the Job is gone (404).
    if batch_v1 is not None:
        try:
            job = batch_v1.read_namespaced_job(job_name, NAMESPACE)
            out.update(_job_metadata(job))
            out["job"] = {
                "name": job_name,
                "active": job.status.active or 0,
                "succeeded": job.status.succeeded or 0,
                "failed": job.status.failed or 0,
            }
        except ApiException as e:
            out["job"] = {"name": job_name, "error": e.reason}
            if getattr(e, "status", None) == 404:
                job_missing = True
        except Exception as e:  # noqa: BLE001 – DNS/network failure, etc.
            out["job"] = {"name": job_name, "error": "k8s api unreachable"}
            log.warning("k8s read_namespaced_job failed: %s", e)

    deploy_ready = False
    if apps_v1 is not None:
        try:
            dep = apps_v1.read_namespaced_deployment_status(deploy_name, NAMESPACE)
            out["deployment"] = {
                "name": deploy_name,
                "ready_replicas": dep.status.ready_replicas or 0,
                "replicas": dep.status.replicas or 0,
            }
            deploy_ready = (dep.status.ready_replicas or 0) >= (dep.spec.replicas or 1)
        except ApiException as e:
            out["deployment"] = {"name": deploy_name, "error": e.reason}
        except Exception as e:  # noqa: BLE001
            out["deployment"] = {"name": deploy_name, "error": "k8s api unreachable"}
            log.warning("k8s read_namespaced_deployment_status failed: %s", e)

    if job is not None:
        out["state"] = _derive_state(job.status, deploy_ready)
        if out["state"] == "failed" and not out.get("error"):
            # Prefer the Job-level failure condition (set by Kubernetes
            # itself) over the dead pod's exit code — it explains *why*
            # the Job failed (DeadlineExceeded, BackoffLimitExceeded, ...).
            for cond in (job.status.conditions or []):
                if cond.type == "Failed" and cond.status == "True":
                    reason = cond.reason or "Failed"
                    msg = cond.message or ""
                    if reason == "DeadlineExceeded":
                        out["error"] = (
                            f"Benchmark exceeded the {JOB_ACTIVE_DEADLINE_SECONDS}s "
                            f"deadline and was terminated by Kubernetes. {msg}"
                        ).strip()
                    else:
                        out["error"] = f"{reason}: {msg}".strip(": ").strip()
                    break
    else:
        prev_state = persisted.get("state", "unknown")
        # If the Job has been confirmed gone (404) but the persisted record
        # still says the run was in progress, mark it failed — it cannot
        # progress without a Job. This covers manual `kubectl delete` and
        # the case where TTL cleanup ran before we observed completion.
        if job_missing and prev_state in ("provisioning", "running"):
            out["state"] = "failed"
            if not out.get("error"):
                out["error"] = "Benchmark Job no longer exists in the cluster."
        else:
            out["state"] = prev_state

    # Pull benchmark pod logs / error reason from the live pod where useful.
    pod = None
    if core_v1 is not None:
        try:
            pod = _pod_for_job(core_v1, job_name)
        except ApiException:
            pod = None
        except Exception as e:  # noqa: BLE001
            log.warning("k8s _pod_for_job failed: %s", e)
            pod = None

    if pod is not None and core_v1 is not None:
        out["pod"] = {
            "name": pod.metadata.name,
            "phase": pod.status.phase,
        }
        try:
            # Fetch raw bytes; tail_lines can split a multi-byte UTF-8 char
            # mid-sequence, which makes the default strict-UTF-8 decode raise.
            resp = core_v1.read_namespaced_pod_log(
                pod.metadata.name, NAMESPACE,
                container="benchmark", tail_lines=400,
                _preload_content=False,
            )
            out["logs"] = resp.data.decode("utf-8", errors="replace")
        except ApiException as e:
            out["logs"] = None
            out["logs_error"] = e.reason
        except UnicodeDecodeError:
            out["logs"] = None
            out["logs_error"] = "log decode error"

        if out["state"] == "failed":
            err_lines: list[str] = []
            for cs in (pod.status.container_statuses or []):
                term = cs.state and cs.state.terminated
                if term and (term.exit_code or 0) != 0:
                    err_lines.append(
                        f"container '{cs.name}' exited {term.exit_code}: "
                        f"{term.reason or ''} {term.message or ''}".strip()
                    )
            for cs in (pod.status.init_container_statuses or []):
                term = cs.state and cs.state.terminated
                if term and (term.exit_code or 0) != 0:
                    # The wait-for-vllm initContainer writes a meaningful
                    # message to /dev/termination-log before exiting (vLLM
                    # restart-limit hit with the real error, or the GPU
                    # capacity timeout). Surface that directly instead of
                    # the generic exit-code line.
                    msg = (term.message or "").strip()
                    if cs.name == "wait-for-vllm" and msg:
                        err_lines.append(msg)
                    else:
                        err_lines.append(
                            f"init '{cs.name}' exited {term.exit_code}: "
                            f"{term.reason or ''} {term.message or ''}".strip()
                        )
            if err_lines:
                out["error"] = "\n".join(err_lines)

    # Fall back to persisted logs/error if the live pod is gone.
    if out.get("logs") is None:
        cached = benchmark_storage.load_logs(run_id)
        if cached:
            out["logs"] = cached
    if not out.get("error") and persisted.get("error"):
        out["error"] = persisted["error"]

    # Persist any new state/logs/error so future reads survive Job TTL.
    try:
        update_fields: dict[str, Any] = {"state": out["state"]}
        if out["state"] in ("success", "failed"):
            if out.get("logs"):
                benchmark_storage.save_logs(run_id, out["logs"])
            if out.get("error"):
                update_fields["error"] = out["error"]
            if persisted.get("state") not in ("success", "failed"):
                update_fields["completed_at"] = (
                    benchmark_storage._now()  # type: ignore[attr-defined]
                )
        if update_fields.get("state") != persisted.get("state") or "error" in update_fields:
            benchmark_storage.update_manifest(run_id, **update_fields)
    except Exception:  # noqa: BLE001
        pass

    return out


def cleanup(run_id: str) -> dict[str, Any]:
    """Tear down Deployment + Service + Job for a run."""
    _load_kube()
    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()
    batch_v1 = client.BatchV1Api()

    deleted = {}
    propagation = client.V1DeleteOptions(propagation_policy="Background")
    for kind, fn, name in (
        ("job", lambda n: batch_v1.delete_namespaced_job(n, NAMESPACE, body=propagation),
         f"bench-{run_id}"),
        ("service", lambda n: core_v1.delete_namespaced_service(n, NAMESPACE),
         f"vllm-{run_id}"),
        ("deployment", lambda n: apps_v1.delete_namespaced_deployment(n, NAMESPACE, body=propagation),
         f"vllm-{run_id}"),
    ):
        try:
            fn(name)
            deleted[kind] = name
        except ApiException as e:
            deleted[kind] = f"error: {e.reason}"
    # Remove persisted manifest + logs so the run no longer shows up in the UI.
    try:
        benchmark_storage.delete_run(run_id)
        deleted["blob"] = "deleted"
    except Exception as e:  # noqa: BLE001
        deleted["blob"] = f"error: {e}"
    return {"run_id": run_id, "deleted": deleted}
