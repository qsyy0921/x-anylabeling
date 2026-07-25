import fcntl
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/data/mfl/autolabel")
RUN_DIR = ROOT / "run"
LOCK_PATH = RUN_DIR / "gpu-worker-assignments.lock"
STATE_PATH = RUN_DIR / "gpu-worker-assignments.json"

bind = "127.0.0.1:18618"
workers = int(os.getenv("AUTOLABEL_EFFECTIVE_WORKERS", "1"))
worker_class = "uvicorn.workers.UvicornWorker"
preload_app = False
timeout = 330
graceful_timeout = 45
keepalive = 5
accesslog = "-"
errorlog = "-"
capture_output = True


def _pid_is_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _query_gpus():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    gpus = []
    for line in output.splitlines():
        index, total_mb, free_mb, utilization = [
            value.strip() for value in line.split(",")
        ]
        gpus.append(
            {
                "index": int(index),
                "total_mb": int(total_mb),
                "free_mb": int(free_mb),
                "utilization": int(utilization),
            }
        )
    return gpus


def _read_state():
    if not STATE_PATH.exists():
        return {}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        pid: record
        for pid, record in data.items()
        if isinstance(record, dict) and _pid_is_alive(pid)
    }


def _write_state(state):
    temp_path = STATE_PATH.with_name(
        f"{STATE_PATH.name}.tmp.{os.getpid()}"
    )
    temp_path.write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp_path, STATE_PATH)


def _allowed_gpu_ids():
    value = os.getenv("AUTOLABEL_GPU_ID", "").strip()
    if not value:
        return None
    return {
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    }


def _select_gpu(worker_pid):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.touch(exist_ok=True)

    with LOCK_PATH.open("r+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        state = _read_state()
        gpus = _query_gpus()
        allowed = _allowed_gpu_ids()
        if allowed is not None:
            gpus = [gpu for gpu in gpus if gpu["index"] in allowed]
        if not gpus:
            raise RuntimeError("No allowed NVIDIA GPU is available")

        min_free_mb = int(os.getenv("AUTOLABEL_MIN_FREE_MB", "8192"))
        max_util = int(os.getenv("AUTOLABEL_MAX_UTIL", "50"))
        preferred = [
            gpu
            for gpu in gpus
            if gpu["free_mb"] >= min_free_mb
            and gpu["utilization"] < max_util
        ]
        candidates = preferred or gpus

        assigned_counts = {}
        for record in state.values():
            if "gpu" not in record:
                continue
            gpu_id = int(record["gpu"])
            assigned_counts[gpu_id] = assigned_counts.get(gpu_id, 0) + 1

        utilization_penalty = int(
            os.getenv("AUTOLABEL_WORKER_UTIL_PENALTY", "10")
        )
        memory_reservation_mb = int(
            os.getenv("AUTOLABEL_WORKER_MEMORY_MB", "8192")
        )

        def load_score(gpu):
            assigned = assigned_counts.get(gpu["index"], 0)
            effective_utilization = min(
                100,
                gpu["utilization"] + assigned * utilization_penalty,
            )
            effective_free_mb = max(
                0,
                gpu["free_mb"] - assigned * memory_reservation_mb,
            )
            memory_pressure = (
                100.0 * (gpu["total_mb"] - effective_free_mb)
                / gpu["total_mb"]
            )
            # Compute activity is slightly more volatile, so give it the
            # larger share while keeping memory pressure a first-class input.
            return 0.60 * effective_utilization + 0.40 * memory_pressure

        # Existing assignments are a soft load, not an exclusive reservation.
        # A sufficiently idle GPU may therefore receive multiple workers.
        chosen = min(
            candidates,
            key=lambda gpu: (
                load_score(gpu),
                -gpu["free_mb"],
                gpu["index"],
            ),
        )

        assigned_before = assigned_counts.get(chosen["index"], 0)
        effective_free_mb = max(
            0,
            chosen["free_mb"]
            - assigned_before * memory_reservation_mb,
        )
        chosen["workers_already_assigned"] = assigned_before
        chosen["combined_load_score"] = round(load_score(chosen), 3)
        state[str(worker_pid)] = {
            "gpu": chosen["index"],
            "utilization_at_start": chosen["utilization"],
            "total_mb": chosen["total_mb"],
            "free_mb_at_start": chosen["free_mb"],
            "workers_already_assigned": assigned_before,
            "effective_utilization_score": (
                chosen["utilization"]
                + assigned_before * utilization_penalty
            ),
            "effective_free_mb": effective_free_mb,
            "combined_load_score": chosen["combined_load_score"],
            "started_at": time.time(),
        }
        _write_state(state)
        return chosen


def _release_gpu(worker_pid):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.touch(exist_ok=True)
    with LOCK_PATH.open("r+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        state = _read_state()
        state.pop(str(worker_pid), None)
        _write_state(state)


def post_fork(server, worker):
    chosen = _select_gpu(worker.pid)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen["index"])
    os.environ["AUTOLABEL_PHYSICAL_GPU_ID"] = str(chosen["index"])
    worker.log.info(
        "Worker %s assigned to physical GPU %s "
        "(utilization=%s%%, free_memory=%s MiB, "
        "workers_already_assigned=%s, combined_score=%s)",
        worker.pid,
        chosen["index"],
        chosen["utilization"],
        chosen["free_mb"],
        chosen.get("workers_already_assigned", 0),
        chosen.get("combined_load_score", "n/a"),
    )


def worker_exit(server, worker):
    _release_gpu(worker.pid)
