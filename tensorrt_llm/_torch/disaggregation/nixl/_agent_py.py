import os
import shutil
import subprocess
import time
from enum import Enum

from nixl import nixl_agent, nixl_agent_config, nixl_xfer_handle

from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger

# Import base classes for type compatibility
from ..base.agent import BaseTransferAgent, RegMemoryDescs, TransferRequest, TransferStatus

# Emit the TCP-fallback warning at most once per process (one NIXL agent is
# created per rank, and each rank may build both a sender and a receiver agent).
_UCX_TCP_FALLBACK_WARNED = False
# Cached {transport_name: type} from ``ucx_info -d`` (a sentinel = not probed;
# None = probe failed/unavailable).
_UNPROBED = object()
_UCX_TRANSPORTS = _UNPROBED

# UCX transports that give a fast CROSS-NODE path for KV cache transfer:
#   * RDMA/IB verbs (rc*/dc*/ud*) — type "network".
#   * cuda/NVLink transports reported as inter-node — i.e. multi-node NVLink
#     (MNNVL, e.g. GB200/GB300 NVL72). Note: a plain ``cuda_ipc`` reported as
#     ``intra-node`` is NOT cross-node and does not count.
_UCX_NVLINK_PREFIXES = ("cuda", "gdr", "nvlink", "gga")


def _token_is_rdma(tok: str) -> bool:
    """Whether a single UCX_TLS token denotes RDMA / InfiniBand verbs."""
    # rc*/dc*/ud* are the IB verbs/mlx5 transports; "ib" is the device alias
    # that expands to them; "rdma" is accepted as a catch-all.
    return tok == "rdma" or tok.startswith(("rc", "dc", "ud", "ib"))


def _ucx_probe_transports():
    """Probe UCX at runtime (``ucx_info -d``) for the transports actually
    available on this node and their type (intra-node / inter-node / network).
    Reflects real hardware + UCX build (IB fabric, MNNVL, libuct_ib), not
    configuration. Returns a ``{name: type}`` dict, or None if unavailable."""
    global _UCX_TRANSPORTS
    if _UCX_TRANSPORTS is not _UNPROBED:
        return _UCX_TRANSPORTS
    ucx_info = shutil.which("ucx_info") or "/usr/local/ucx/bin/ucx_info"
    try:
        out = subprocess.run([ucx_info, "-d"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None  # leave unprobed so a later call may retry
    transports, current = {}, None
    for line in out.splitlines():
        if "Transport:" in line:
            current = line.split("Transport:", 1)[1].strip()
            transports.setdefault(current, "")
        elif "Type:" in line and current is not None:
            transports[current] = line.split("Type:", 1)[1].strip()
    _UCX_TRANSPORTS = transports
    return _UCX_TRANSPORTS


def _ucx_fast_xnode_hw(transports):
    """From a probed ``{name: type}`` map, return (has_ib, has_mnnvl):
    whether IB verbs and/or an inter-node cuda/NVLink (MNNVL) transport
    are available for fast cross-node KV transfer."""
    has_ib = any(n.startswith(("rc", "dc", "ud")) for n in transports)
    has_mnnvl = any(
        n.startswith(_UCX_NVLINK_PREFIXES) and t == "inter-node" for n, t in transports.items()
    )
    return has_ib, has_mnnvl


def _ucx_tls_disables_rdma(ucx_tls: str) -> bool:
    """Return True when UCX_TLS filters out RDMA/IB verbs. Returns False when
    unset or ``all``:
      * exclude-list, e.g. ``^ib``     -> True iff an RDMA token is excluded.
      * include-list, e.g. ``tcp,sm``  -> True iff no RDMA token is included.
    """
    val = ucx_tls.strip().lower()
    if not val or val == "all":
        return False
    is_exclude = val.startswith("^")
    tokens = [t.strip() for t in val.lstrip("^").split(",") if t.strip()]
    if not tokens:
        return False
    if is_exclude:
        return any(_token_is_rdma(t) for t in tokens)
    return not any(_token_is_rdma(t) for t in tokens)


def _ucx_tls_allows_cuda(ucx_tls: str) -> bool:
    """Return True when UCX_TLS permits the cuda/NVLink transports (used for
    MNNVL). True when unset/``all``; ``^ib`` does NOT exclude cuda."""
    val = ucx_tls.strip().lower()
    if not val or val == "all":
        return True
    is_exclude = val.startswith("^")
    tokens = [t.strip() for t in val.lstrip("^").split(",") if t.strip()]
    if is_exclude:
        return not any(t.startswith(_UCX_NVLINK_PREFIXES) for t in tokens)
    return any(t.startswith(_UCX_NVLINK_PREFIXES) for t in tokens)


def _warn_if_ucx_tcp_fallback() -> None:
    """Warn once if cross-node KV cache transfer will fall back to TCP instead
    of a fast fabric. Detection is an actual UCX runtime probe (``ucx_info
    -d``, including transport type for MNNVL/NVLink) combined with the UCX_TLS
    filter — so it catches a node with no IB/NVLink transport AND a node where
    UCX_TLS excludes an available one, without false-warning when multi-node
    NVLink is the (fast) path."""
    global _UCX_TCP_FALLBACK_WARNED
    if _UCX_TCP_FALLBACK_WARNED:
        return

    ucx_tls = os.environ.get("UCX_TLS")
    transports = _ucx_probe_transports()  # {name: type} or None

    if transports is None:
        # Could not probe — fall back to the UCX_TLS-only heuristic.
        if ucx_tls is None or not _ucx_tls_disables_rdma(ucx_tls):
            return
        reason = (
            f"UCX_TLS={ucx_tls!r} excludes the IB transports "
            "(could not run ucx_info to confirm fabric availability)"
        )
    else:
        has_ib, has_mnnvl = _ucx_fast_xnode_hw(transports)
        mnnvl_usable = has_mnnvl and (ucx_tls is None or _ucx_tls_allows_cuda(ucx_tls))
        ib_usable = has_ib and not (ucx_tls is not None and _ucx_tls_disables_rdma(ucx_tls))
        if mnnvl_usable or ib_usable:
            return  # a fast cross-node path (NVLink or IB) is available
        if not has_ib and not has_mnnvl:
            reason = (
                "UCX on this node exposes no RDMA/IB and no inter-node "
                "NVLink transport (only TCP) — check the IB fabric / "
                "MNNVL fabric-manager / rdma-core / UCX build (libuct_ib) "
                "in the container"
            )
        else:
            avail = (
                ("RDMA/IB" if has_ib else "")
                + (" and " if has_ib and has_mnnvl else "")
                + ("inter-node NVLink" if has_mnnvl else "")
            )
            reason = f"{avail} is available but UCX_TLS={ucx_tls!r} excludes it"

    _UCX_TCP_FALLBACK_WARNED = True
    rank = (
        os.environ.get("SLURM_PROCID")
        or os.environ.get("OMPI_COMM_WORLD_RANK")
        or os.environ.get("RANK")
        or "?"
    )
    logger.warning(
        f"[RANK {rank}] NIXL KV cache transfer will use TCP, not RDMA/NVLink: "
        f"{reason}. Cross-node KV transfer over TCP is far slower than RDMA/NVLink "
        f"and is a common cause of 'KV cache transfer timeout' errors at high "
        f"concurrency. To use the fast fabric, ensure IB/MNNVL is reachable and "
        f"either unset UCX_TLS (let UCX auto-select) or include a fast transport "
        f"(e.g. UCX_TLS=rc,cuda_ipc,sm). If TCP is intended, raise "
        f"cache_transceiver_config.kv_transfer_timeout_ms accordingly."
    )


class TransferState(Enum):
    PENDING = "PENDING"
    PROCESSING = "PROC"
    DONE = "DONE"
    ERROR = "ERROR"


class NixlTransferStatus(TransferStatus):
    def __init__(self, agent: nixl_agent, handle: nixl_xfer_handle):
        self.agent = agent
        self.handle = handle

    def is_completed(self):
        status = TransferState(self.agent.check_xfer_state(self.handle))
        return status == TransferState.DONE

    def wait(self, timeout_ms=None):
        start_time = time.time()
        status = TransferState.PENDING
        sleep_time = 0.0001  # 0.1ms in seconds
        max_sleep_time = 0.01  # 10ms in seconds

        timeout = timeout_ms / 1000 if timeout_ms is not None else None

        while status in (TransferState.PENDING, TransferState.PROCESSING):
            status = TransferState(self.agent.check_xfer_state(self.handle))
            if status == TransferState.ERROR:
                logger.error("NIXL transfer entered ERROR state (agent=%s).", self.agent.name)
                return False
            if timeout is not None and (time.time() - start_time > timeout):
                logger.warning("NIXL transfer wait timed out after %s ms.", timeout_ms)
                return False
            time.sleep(sleep_time)
            sleep_time = min(sleep_time * 2, max_sleep_time)
        return status == TransferState.DONE


class NixlTransferAgent(BaseTransferAgent):
    """NixlTransferAgent using Python nixl library."""

    def __init__(self, name: str, use_prog_thread: bool = True, num_threads: int = 1, **kwargs):
        """
        Initialize NixlTransferAgent.
        :param name: Name of the agent.
        :param use_prog_thread: Whether to enable the progress thread, if available.
        :param num_workers: Specify number of threads for the supported multi-threaded backends.
        """
        self.name = name
        self.backends = ["UCX"]
        # The UCX backend picks TCP vs RDMA from UCX_TLS; warn if RDMA is disabled.
        _warn_if_ucx_tcp_fallback()
        agent_config = nixl_agent_config(
            enable_prog_thread=use_prog_thread, backends=self.backends, num_threads=num_threads
        )
        self.agent = nixl_agent(name, agent_config)

    def shutdown(self):
        if getattr(self, "agent", None) is None:
            return
        self.agent = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.shutdown()

    def _get_validated_reg_descs(self, descs: RegMemoryDescs):
        if not descs.descs:
            raise ValueError("descs.descs must not be empty")
        if isinstance(descs.descs[0], tuple) and len(descs.descs[0]) != 4:
            raise ValueError(
                f"Expected 4 elements per desc, got {len(descs.descs[0])}: {descs.descs[0]}"
            )
        reg_descs = self.agent.get_reg_descs(descs.descs, descs.type)
        if reg_descs is None:
            raise RuntimeError(
                f"nixl get_reg_descs returned None for type={descs.type}, count={len(descs.descs)}"
            )
        return reg_descs

    def register_memory(self, descs: RegMemoryDescs):
        self.agent.register_memory(self._get_validated_reg_descs(descs))

    def deregister_memory(self, descs: RegMemoryDescs):
        self.agent.deregister_memory(self._get_validated_reg_descs(descs))

    def load_remote_agent(self, name: str, agent_desc: bytes):
        self.agent.add_remote_agent(agent_desc)

    def get_local_agent_desc(self):
        return self.agent.get_agent_metadata()

    def invalidate_remote_agent(self, name: str):
        self.agent.remove_remote_agent(name)

    def check_remote_descs(self, name: str, memory_descs: list[int]) -> bool:
        raise NotImplementedError

    def notify_sync_message(self, name: str, sync_message: str):
        raise NotImplementedError

    @nvtx_range("NixlTransferAgent.submit_transfer_requests")
    def submit_transfer_requests(self, request: TransferRequest) -> TransferStatus:
        src_xfer_descs = self.agent.get_xfer_descs(request.src_descs.descs, request.src_descs.type)
        if src_xfer_descs is None:
            raise RuntimeError(
                f"nixl get_xfer_descs returned None for src type={request.src_descs.type}"
            )
        dst_xfer_descs = self.agent.get_xfer_descs(request.dst_descs.descs, request.dst_descs.type)
        if dst_xfer_descs is None:
            raise RuntimeError(
                f"nixl get_xfer_descs returned None for dst type={request.dst_descs.type}"
            )
        sync_message = "" if request.sync_message is None else request.sync_message
        handle = self.agent.initialize_xfer(
            request.op,
            src_xfer_descs,
            dst_xfer_descs,
            request.remote_name,
            sync_message,
        )
        status = self.agent.transfer(handle)
        if status == "ERROR":
            raise RuntimeError(
                f"NIXL transfer failed: op={request.op}, remote={request.remote_name}"
            )
        return NixlTransferStatus(self.agent, handle)
