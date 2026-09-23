"""Initialize all four TPU hosts before executing a TP32 experiment."""

import os
import runpy
import socket
import sys

rank = int(os.environ["SLURM_PROCID"])
os.environ["TPU_WORKER_ID"] = str(rank)
os.environ["CLOUD_TPU_TASK_ID"] = str(rank)
llo_dump_root = os.environ.get("TPU_LLO_DUMP_ROOT")
if llo_dump_root and rank == 0:
    llo_flags = (
        f"--xla_jf_dump_to={llo_dump_root} "
        "--xla_jf_dump_hlo_text=true "
        "--xla_jf_dump_llo_text=true "
        "--xla_jf_dump_llo_html=false "
        "--xla_jf_dump_llo_static_gaps=true "
        "--xla_jf_emit_annotations=true "
        "--xla_mosaic_enable_llo_source_annotations=true "
        "--xla_jf_debug_level=2"
    )
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(
        part for part in (os.environ.get("LIBTPU_INIT_ARGS"), llo_flags) if part
    )
for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(name, None)
# The TPU runtime reopens fd 0 as /dev/null during initialization; keep a
# duplicate of the launcher's stdin (srun forwards the terminal to task 0) so
# interactive scripts can still read it (see demo_kimi_dspark.py).
try:
    os.environ["K3_STDIN_FD"] = str(os.dup(0))
except OSError:
    pass
import jax

print(f"initializing rank={rank} host={socket.gethostname()}", flush=True)
jax.distributed.initialize(
    coordinator_address=os.environ["TP32_COORDINATOR"],
    num_processes=4,
    process_id=rank,
    initialization_timeout=180,
)
print(f"ready rank={rank} local={jax.local_device_count()} global={jax.device_count()}", flush=True)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
