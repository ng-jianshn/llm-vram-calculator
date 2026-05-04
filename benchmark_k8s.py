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

import os
import re
import uuid
from typing import Any

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
        path = os.path.join(tempfile.gettempdir(), "kubeconfig")
        with open(path, "wb") as f:
            f.write(base64.b64decode(kubeconfig_b64))
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
        args += [
            "--dataset-path",
            payload.get("dataset_path", "./ShareGPT_V3_unfiltered_cleaned_split.json"),
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

    # initContainer waits for /health
    wait_cmd = (
        f"echo 'Waiting for vLLM at {svc_url}/health';"
        f"for i in $(seq 1 180); do "
        f"  if wget -q -O- {svc_url}/health >/dev/null 2>&1; then "
        f"    echo 'vLLM is ready'; exit 0; "
        f"  fi; "
        f"  echo \"attempt $i...\"; sleep 10; "
        f"done; echo 'vLLM never became ready'; exit 1"
    )
    init_container = client.V1Container(
        name="wait-for-vllm",
        image="busybox:1.36",
        command=["/bin/sh", "-c", wait_cmd],
    )

    bench_container = client.V1Container(
        name="benchmark",
        image=VLLM_IMAGE,
        command=bench_args[:1],   # "vllm"
        args=bench_args[1:],
        env=_hf_env(),
    )

    job = client.V1Job(
        metadata=client.V1ObjectMeta(name=name, labels=labels, namespace=NAMESPACE),
        spec=client.V1JobSpec(
            backoff_limit=0,
            ttl_seconds_after_finished=180,  # auto-clean Job 3min after completion
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels=labels),
                spec=client.V1PodSpec(
                    init_containers=[init_container],
                    containers=[bench_container],
                    restart_policy="Never",
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

    return {
        "run_id": run_id,
        "namespace": NAMESPACE,
        "deployment": deploy.metadata.name,
        "service": svc.metadata.name,
        "job": created_job.metadata.name,
        "command": " ".join(_build_serve_args(payload,
                            f"http://{svc.metadata.name}:8000")),
    }


def get_status(run_id: str) -> dict[str, Any]:
    """Lightweight status check for a previously submitted run."""
    _load_kube()
    batch_v1 = client.BatchV1Api()
    apps_v1 = client.AppsV1Api()

    job_name = f"bench-{run_id}"
    deploy_name = f"vllm-{run_id}"

    out: dict[str, Any] = {"run_id": run_id, "namespace": NAMESPACE}
    try:
        job = batch_v1.read_namespaced_job_status(job_name, NAMESPACE)
        out["job"] = {
            "name": job_name,
            "active": job.status.active or 0,
            "succeeded": job.status.succeeded or 0,
            "failed": job.status.failed or 0,
        }
    except ApiException as e:
        out["job"] = {"name": job_name, "error": e.reason}

    try:
        dep = apps_v1.read_namespaced_deployment_status(deploy_name, NAMESPACE)
        out["deployment"] = {
            "name": deploy_name,
            "ready_replicas": dep.status.ready_replicas or 0,
            "replicas": dep.status.replicas or 0,
        }
    except ApiException as e:
        out["deployment"] = {"name": deploy_name, "error": e.reason}

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
    return {"run_id": run_id, "deleted": deleted}
