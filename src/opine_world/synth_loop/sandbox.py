"""Bubblewrap-based filesystem sandbox for analyzer/synthesis subprocesses.

Prevents subprocesses from reading game source outside their workspace. Network is
not isolated. Falls back to unsandboxed execution with a warning when bwrap is absent.
"""
from __future__ import annotations

import logging
import os
import resource
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def bwrap_available() -> bool:
    return shutil.which("bwrap") is not None


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_bytes(raw: str | None, default: str) -> int:
    text = (raw or default).strip().lower()
    if not text:
        text = default
    units = {
        "k": 1024,
        "kb": 1024,
        "m": 1024 ** 2,
        "mb": 1024 ** 2,
        "g": 1024 ** 3,
        "gb": 1024 ** 3,
    }
    for suffix, mult in sorted(units.items(), key=lambda kv: -len(kv[0])):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)].strip()) * mult)
    return int(float(text))


def _claude_limit_values() -> dict[str, int]:
    """Hard caps inherited by local Claude Code and all Bash/Python children.

    Codex runs are already container-capped. These rlimits cover the local
    Claude path, including tool-launched planner probes that otherwise outlive
    the parent prompt and can exhaust WSL.
    """
    if not _env_flag("ARC3_CLAUDE_RLIMITS", True):
        return {}
    memory_raw = (
        os.environ.get("ARC3_CLAUDE_SUBPROCESS_MEMORY")
        or os.environ.get("ARC3_SUBPROCESS_MEMORY")
        or "24g"
    )
    rss_raw = (
        os.environ.get("ARC3_CLAUDE_SUBPROCESS_RSS")
        or os.environ.get("ARC3_SUBPROCESS_RSS")
        or "12g"
    )
    file_raw = (
        os.environ.get("ARC3_CLAUDE_SUBPROCESS_FSIZE")
        or os.environ.get("ARC3_SUBPROCESS_FSIZE")
        or "1g"
    )
    return {
        "as_bytes": _parse_bytes(memory_raw, "24g"),
        "rss_bytes": _parse_bytes(rss_raw, "12g"),
        "cpu_seconds": _env_int(
            "ARC3_CLAUDE_SUBPROCESS_CPU_SECONDS",
            _env_int("ARC3_SUBPROCESS_CPU_SECONDS", 240),
        ),
        "fsize_bytes": _parse_bytes(file_raw, "1g"),
        "nofile": _env_int(
            "ARC3_CLAUDE_SUBPROCESS_NOFILE",
            _env_int("ARC3_SUBPROCESS_NOFILE", 1024),
        ),
    }


def describe_claude_resource_limits() -> str:
    limits = _claude_limit_values()
    if not limits:
        return "disabled"
    vmem_gb = limits["as_bytes"] / (1024 ** 3)
    rss_gb = limits.get("rss_bytes", limits["as_bytes"]) / (1024 ** 3)
    f_gb = limits["fsize_bytes"] / (1024 ** 3)
    return (
        f"vmem(virtual)={vmem_gb:.1f}GB, rss(kill)={rss_gb:.1f}GB, "
        f"cpu={limits['cpu_seconds']}s/process, "
        f"fsize={f_gb:.1f}GB, nofile={limits['nofile']}"
    )


def claude_resource_preexec():
    """Return a preexec_fn that applies inherited local-Claude rlimits."""
    limits = _claude_limit_values()
    if not limits:
        return None

    def _apply_limits() -> None:
        as_bytes = int(limits["as_bytes"])
        resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))
        if hasattr(resource, "RLIMIT_DATA"):
            try:
                resource.setrlimit(resource.RLIMIT_DATA, (as_bytes, as_bytes))
            except Exception:
                pass

        cpu = int(limits["cpu_seconds"])
        if cpu > 0:
            hard = max(cpu + 5, cpu)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, hard))

        fsize = int(limits["fsize_bytes"])
        if fsize > 0:
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))

        nofile = int(limits["nofile"])
        if nofile > 0:
            resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))

    return _apply_limits


def claude_popen_kwargs() -> dict[str, Any]:
    """Popen kwargs for local Claude so its tool subprocesses are bounded."""
    kwargs: dict[str, Any] = {"start_new_session": True}
    preexec = claude_resource_preexec()
    if preexec is not None:
        kwargs["preexec_fn"] = preexec
    return kwargs


def terminate_process_group(
    proc: subprocess.Popen,
    *,
    grace_s: float = 3.0,
) -> None:
    """Terminate a Popen process and any children in its process group."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=grace_s)
        return
    except Exception:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=grace_s)
    except Exception:
        pass


def _children_by_parent() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    proc_dir = Path("/proc")
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat = (entry / "stat").read_text()
            after_comm = stat.rsplit(")", 1)[1].strip().split()
            ppid = int(after_comm[1])
        except Exception:
            continue
        children.setdefault(ppid, []).append(pid)
    return children


def process_tree_pids(root_pid: int) -> set[int]:
    children = _children_by_parent()
    seen: set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, []))
    return seen


def _pid_rss_bytes(pid: int) -> int:
    try:
        statm = Path(f"/proc/{pid}/statm").read_text().split()
        if len(statm) < 2:
            return 0
        return int(statm[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


def process_tree_rss_bytes(root_pid: int) -> int:
    return sum(_pid_rss_bytes(pid) for pid in process_tree_pids(root_pid))


def wait_with_resource_monitor(
    proc: subprocess.Popen,
    *,
    timeout_s: int | float | None = None,
    memory_bytes: int | None = None,
    poll_s: float = 1.0,
    log_fn: Any | None = None,
) -> int:
    """Wait for proc while killing the whole tree on aggregate RSS overflow."""
    limits = _claude_limit_values()
    rss_limit = memory_bytes
    if rss_limit is None and limits:
        rss_limit = int(limits.get("rss_bytes") or limits["as_bytes"])
    start = time.monotonic()
    peak_rss = 0
    while True:
        rc = proc.poll()
        if rc is not None:
            return int(rc)

        if rss_limit and rss_limit > 0:
            rss = process_tree_rss_bytes(proc.pid)
            peak_rss = max(peak_rss, rss)
            if rss > rss_limit:
                if log_fn is not None:
                    log_fn(
                        "RESOURCE",
                        "process tree RSS limit exceeded: "
                        f"rss={rss / (1024 ** 3):.2f}GB "
                        f"limit={rss_limit / (1024 ** 3):.2f}GB "
                        f"root_pid={proc.pid}",
                    )
                terminate_process_group(proc)
                return int(proc.returncode if proc.returncode is not None else -9)

        if timeout_s is not None and (time.monotonic() - start) >= timeout_s:
            raise subprocess.TimeoutExpired(proc.args, timeout_s)

        time.sleep(poll_s)


def docker_available() -> bool:
    return shutil.which("docker") is not None


def _live_claude_gateway_ip() -> str | None:
    """The claude-gateway container's IP on the claude-filtered network."""
    try:
        res = subprocess.run(
            [
                "docker", "inspect", "-f",
                '{{(index .NetworkSettings.Networks "claude-filtered").IPAddress}}',
                "claude-gateway",
            ],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except Exception:
        return None
    if res.returncode != 0:
        return None
    ip = res.stdout.strip()
    return ip if ip and ip != "<no value>" else None


def claude_gateway_ip(explicit: str | None = None) -> str | None:
    """Resolve the claude egress gateway's agent-side IP (default route + DNS).

    Prefers the explicit value, then the live claude-gateway container, then the
    cached file. A file-only IP is treated as stale after a Docker restart (same
    reasoning as the codex backend): force a gateway_up.sh rather than launch a
    container pointed at a dead gateway.
    """
    if explicit:
        return explicit
    live = _live_claude_gateway_ip()
    f = Path(__file__).resolve().parents[3] / "docker" / "gateway_internal_ip.txt"
    if live:
        try:
            f.write_text(live)
        except Exception:
            pass
        return live
    return None


def wrap_for_docker(
    cmd: list[str],
    *,
    workspace_dir: Path,
    engine_output_dir: Path,
    home_dir: Path | None = None,
    image: str = "claude-agent",
    network: str = "claude-filtered",
    gateway: str | None = None,
    container_name: str | None = None,
    memory: str = "12g",
    cpus: str = "2.0",
    pids_limit: str = "512",
    extra_ro_binds: list[Path] | None = None,
) -> list[str]:
    """Wrap `cmd` (a full `claude ...` argv) in a locked-down Docker run.

    This is the Docker analogue of ``wrap_for_sandbox`` and is intentionally
    bind-for-bind identical to it, mounted at the SAME absolute host paths inside
    the container. Same paths => the prompt's embedded absolute paths, the cwd,
    and the ~/.claude project-session hashing (which keys on cwd) all match the
    bwrap path exactly, so claude behaves identically, --continue/--resume keep
    working, and the files written into the run dir are byte-equivalent. The only
    differences from bwrap are (a) network egress is filtered through the
    allowlisting gateway instead of open, and (b) the image ships a Python
    toolbox. Raises RuntimeError if the gateway is not up (fail closed -- never
    silently run without isolation).
    """
    home = Path(home_dir or os.environ.get("HOME", "/root"))
    workspace_dir = workspace_dir.resolve()
    engine_output_dir = engine_output_dir.resolve()

    gw = claude_gateway_ip(gateway)
    if not gw:
        raise RuntimeError(
            "claude egress gateway unavailable; run docker/gateway_up.sh first"
        )
    uid, gid = os.getuid(), os.getgid()

    docker_cmd: list[str] = ["docker", "run", "--rm", "-i", "--init"]
    if container_name:
        docker_cmd += ["--name", container_name]
    docker_cmd += [
        "--network", network,
        "--cap-add", "NET_ADMIN",
        "--dns", gw,
        "--user", "0",
        "--workdir", str(workspace_dir),
    ]
    if memory and str(memory).lower() not in {"0", "none", "off"}:
        docker_cmd += ["--memory", str(memory), "--memory-swap", str(memory)]
    if cpus and str(cpus).lower() not in {"0", "none", "off"}:
        docker_cmd += ["--cpus", str(cpus)]
    if pids_limit and str(pids_limit).lower() not in {"0", "none", "off"}:
        docker_cmd += ["--pids-limit", str(pids_limit)]

    if engine_output_dir.exists():
        docker_cmd += ["-v", f"{engine_output_dir}:{engine_output_dir}:ro"]
    docker_cmd += ["-v", f"{workspace_dir}:{workspace_dir}:rw"]

    claude_dir = home / ".claude"
    if claude_dir.exists():
        docker_cmd += ["-v", f"{claude_dir}:{claude_dir}:rw"]
    local_dir = home / ".local"
    if local_dir.exists():
        docker_cmd += ["-v", f"{local_dir}:{local_dir}:ro"]

    claude_json = home / ".claude.json"
    cfg_dir_env = os.environ.get("CLAUDE_CONFIG_DIR")
    claude_json_src: Path | None = claude_json if claude_json.exists() else None
    extra_env: list[str] = []
    if cfg_dir_env:
        cfg_dir = Path(cfg_dir_env).resolve()
        if cfg_dir.exists():
            docker_cmd += ["-v", f"{cfg_dir}:{cfg_dir}:rw"]
            extra_env += ["--env", f"CLAUDE_CONFIG_DIR={cfg_dir}"]
            alt_json = cfg_dir / ".claude.json"
            if alt_json.exists():
                claude_json_src = alt_json
    if claude_json_src is not None:
        docker_cmd += ["-v", f"{claude_json_src}:{claude_json}:rw"]

    for extra in (extra_ro_binds or []):
        p = Path(extra)
        if p.exists():
            rp = p.resolve()
            docker_cmd += ["-v", f"{rp}:{rp}:ro"]

    for k, v in os.environ.items():
        if k == "CLAUDE_CONFIG_DIR":
            continue
        if k.startswith("ANTHROPIC_") or k.startswith("CLAUDE_"):
            extra_env += ["--env", f"{k}={v}"]
    docker_cmd += extra_env
    docker_cmd += [image]

    cfg_export = f"CLAUDE_CONFIG_DIR={cfg_dir_env} " if cfg_dir_env else ""
    wrapper = (
        f"mkdir -p {home} 2>/dev/null; chown {uid}:{gid} {home} 2>/dev/null; "
        f"ip route replace default via {gw} >/dev/null 2>&1; "
        f"exec setpriv --reuid={uid} --regid={gid} --init-groups "
        f"env HOME={home} {cfg_export}\"$@\""
    )
    return docker_cmd + ["bash", "-lc", wrapper, "_", *cmd]


def wrap_for_sandbox(
    cmd: list[str],
    *,
    workspace_dir: Path,
    engine_output_dir: Path,
    home_dir: Path | None = None,
    extra_ro_binds: list[Path] | None = None,
) -> list[str]:
    """Wrap cmd in a bwrap sandbox. Returns cmd unchanged if bwrap is unavailable.

    workspace_dir is bound read-write. engine_output_dir is bound read-only so symlinks resolve.
    """
    if not bwrap_available():
        log.warning(
            "bwrap not found; running '%s' without filesystem sandbox. "
            "Install with `apt install bubblewrap` to enable.", cmd[0],
        )
        return cmd

    home = Path(home_dir or os.environ.get("HOME", "/root"))
    workspace_dir = workspace_dir.resolve()
    engine_output_dir = engine_output_dir.resolve()

    agent_pkgs_raw = os.environ.get(
        "ARC3_AGENT_PYTHONPATH", str(home / ".arc_agent_pkgs" / "py312")
    )
    agent_pkgs: Path | None = None
    if agent_pkgs_raw.strip():
        candidate = Path(agent_pkgs_raw).expanduser().resolve()
        if candidate.is_dir():
            agent_pkgs = candidate

    ro_binds: list[Path] = []
    for p in ("/usr", "/lib", "/lib32", "/lib64", "/bin", "/sbin", "/etc"):
        if Path(p).exists():
            ro_binds.append(Path(p))
    for extra in (extra_ro_binds or []):
        if Path(extra).exists():
            ro_binds.append(Path(extra).resolve())

    claude_dir = home / ".claude"
    local_dir = home / ".local"

    bwrap_cmd: list[str] = ["bwrap"]
    bwrap_cmd += [
        "--unshare-pid",
        "--unshare-uts",
        "--unshare-ipc",
        "--die-with-parent",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
    ]
    for p in ro_binds:
        bwrap_cmd += ["--ro-bind", str(p), str(p)]

    bwrap_cmd += ["--tmpfs", str(home)]
    if agent_pkgs is not None:
        bwrap_cmd += ["--ro-bind", str(agent_pkgs), str(agent_pkgs)]
    if claude_dir.exists():
        bwrap_cmd += ["--bind", str(claude_dir), str(claude_dir)]
    if local_dir.exists():
        bwrap_cmd += ["--ro-bind", str(local_dir), str(local_dir)]
    claude_json = home / ".claude.json"
    if claude_json.exists():
        bwrap_cmd += ["--bind", str(claude_json), str(claude_json)]

    cfg_dir_env = os.environ.get("CLAUDE_CONFIG_DIR")
    if cfg_dir_env:
        cfg_dir = Path(cfg_dir_env).resolve()
        if cfg_dir.exists():
            bwrap_cmd += ["--bind", str(cfg_dir), str(cfg_dir)]
            alt_json = cfg_dir / ".claude.json"
            if alt_json.exists():
                bwrap_cmd += ["--bind", str(alt_json), str(claude_json)]
            bwrap_cmd += ["--setenv", "CLAUDE_CONFIG_DIR", str(cfg_dir)]

    resolv = Path("/etc/resolv.conf")
    if resolv.is_symlink():
        try:
            target = resolv.resolve()
            if target.exists() and not str(target).startswith("/etc"):
                bwrap_cmd += ["--ro-bind", str(target), str(target)]
        except Exception:
            pass

    if engine_output_dir.exists():
        bwrap_cmd += ["--ro-bind", str(engine_output_dir),
                      str(engine_output_dir)]
    bwrap_cmd += ["--bind", str(workspace_dir), str(workspace_dir)]

    bwrap_cmd += [
        "--setenv", "HOME", str(home),
        "--chdir", str(workspace_dir),
    ]
    if agent_pkgs is not None:
        bwrap_cmd += ["--setenv", "PYTHONPATH", str(agent_pkgs)]
    bwrap_cmd += ["--"]

    return bwrap_cmd + cmd
