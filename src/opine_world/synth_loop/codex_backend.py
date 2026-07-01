"""OpenAI Codex backend: run an analyzer / synth turn inside the locked-down
codex-agent Docker container instead of the local `claude` CLI.

Network isolation is enforced at the network
layer, not by trusting the model: the container runs on `codex-filtered`, a
Docker bridge with host NAT disabled. At startup it briefly runs as root only
to point its default route and DNS at `codex-gateway`, then drops to a
non-sudo user before Codex starts. The gateway NATs/forwards only allowlisted
OpenAI/ChatGPT IPs learned through dnsmasq+nftset and drops everything else.
See codex_backend/ for the images, gateway, and egress_test.sh proof.

Filesystem isolation: only the run dir (the model's own output) and CODEX_HOME
(auth) are mounted. The game source, repo, and the rest of $HOME are never
mounted, so they are invisible -- the same posture as the bwrap claude path.

The contract is file-based and identical to the claude backend: the analyzer
writes next_actions.json into its workspace, and the synth edits game_engine.py.
This module only has to (a) deliver the prompt + any images, (b) run the turn
to completion through the gateway, and (c) report success/usage + the codex
session id for `resume` continuity (the analogue of `claude --continue`).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

DEFAULT_IMAGE = "codex-agent"
DEFAULT_NETWORK = "codex-filtered"
DEFAULT_CODEX_HOME = "~/.codex-arc"
DEFAULT_DOCKER_CPUS = "2.0"
DEFAULT_DOCKER_MEMORY = "4g"
DEFAULT_DOCKER_PIDS_LIMIT = "512"
_PROVIDER_FLAGS = [
    "-c", 'model_provider="openai_https"',
    "-c", 'model_providers.openai_https.name="OpenAI"',
    "-c", 'model_providers.openai_https.wire_api="responses"',
    "-c", "model_providers.openai_https.requires_openai_auth=true",
    "-c", "model_providers.openai_https.supports_websockets=false",
    "-c", 'web_search="disabled"',
]

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                      r"[0-9a-f]{4}-[0-9a-f]{12}$")


def codex_home_path(codex_home: str | None) -> Path:
    return Path(codex_home or DEFAULT_CODEX_HOME).expanduser().resolve()


def _live_gateway_ip() -> str | None:
    try:
        res = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                '{{(index .NetworkSettings.Networks "codex-filtered").IPAddress}}',
                "codex-gateway",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if res.returncode != 0:
        return None
    ip = res.stdout.strip()
    return ip if ip and ip != "<no value>" else None


def gateway_ip(explicit: str | None = None) -> str | None:
    """The transparent gateway's internal IP (agent default route + DNS).
    Uses the explicit value, else the live codex-gateway container.

    Do not blindly trust gateway_internal_ip.txt after WSL/Docker restarts: a
    stale route produces Codex reconnect storms that look like API flakiness.
    """
    if explicit:
        return explicit
    f = Path(__file__).resolve().parents[3] / "codex_backend" / "gateway_internal_ip.txt"
    live = _live_gateway_ip()
    if live:
        try:
            f.write_text(live)
        except Exception:
            pass
        return live
    try:
        ip = f.read_text().strip()
    except Exception:
        return None
    if ip and ip != "invalid IP":
        return None
    return None


def _extract_session_id(events: list[dict]) -> str | None:
    """Pull the codex session/thread id out of the --json event stream."""
    for ev in events:
        for k in ("session_id", "thread_id", "conversation_id"):
            v = ev.get(k)
            if isinstance(v, str) and _UUID_RE.match(v):
                return v
        for v in ev.values():
            if isinstance(v, dict):
                for kk in ("id", "session_id", "thread_id"):
                    vv = v.get(kk)
                    if isinstance(vv, str) and _UUID_RE.match(vv):
                        return vv
    return None


def _scan_events(out_text: str) -> tuple[list[dict], bool, dict[str, int]]:
    """Parse --json lines. Returns (events, turn_failed, usage)."""
    events: list[dict] = []
    turn_failed = False
    turn_completed = False
    usage = {"input_tokens": 0, "output_tokens": 0}
    for line in out_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        events.append(ev)
        t = ev.get("type")
        if t == "turn.completed":
            turn_completed = True
        if t == "turn.failed" or t == "error":
            turn_failed = True
        u = ev.get("usage") or (ev.get("turn") or {}).get("usage") \
            or (ev.get("response") or {}).get("usage")
        if isinstance(u, dict):
            for src, dst in (("input_tokens", "input_tokens"),
                             ("output_tokens", "output_tokens"),
                             ("prompt_tokens", "input_tokens"),
                             ("completion_tokens", "output_tokens")):
                if isinstance(u.get(src), int):
                    usage[dst] = u[src]
    return events, (turn_failed and not turn_completed), usage


def _is_quota_limited(out_text: str, err_text: str) -> bool:
    """Detect hard Codex account quota exhaustion.

    This is different from a transient transport failure: retrying immediately
    just creates new failed turns until the account reset time.
    """
    text = f"{out_text}\n{err_text}".lower()
    quota_markers = (
        "you've hit your usage limit",
        "you have hit your usage limit",
        "usage limit",
        "purchase more credits",
        "try again at",
        "codex/settings/usage",
    )
    return any(marker in text for marker in quota_markers)


def _is_remote_compact_failure(out_text: str, err_text: str) -> bool:
    text = f"{out_text}\n{err_text}".lower()
    return (
        "remote compact task" in text
        or "compact_remote" in text
        or "remote compaction failed" in text
    )


def _docker_resource_flags(
    *,
    cpus: str | None = None,
    memory: str | None = None,
    memory_swap: str | None = None,
    pids_limit: str | None = None,
) -> list[str]:
    """Default Codex containers to bounded local resource use.

    Any limit may be disabled with the corresponding env var set to "0" or
    "none". These are safety rails for local sweeps, not security boundaries.
    """
    cpus = cpus if cpus is not None else os.environ.get(
        "CODEX_DOCKER_CPUS", DEFAULT_DOCKER_CPUS
    )
    memory = memory if memory is not None else os.environ.get(
        "CODEX_DOCKER_MEMORY", DEFAULT_DOCKER_MEMORY
    )
    memory_swap = memory_swap if memory_swap is not None else os.environ.get(
        "CODEX_DOCKER_MEMORY_SWAP", memory
    )
    pids_limit = pids_limit if pids_limit is not None else os.environ.get(
        "CODEX_DOCKER_PIDS_LIMIT", DEFAULT_DOCKER_PIDS_LIMIT
    )

    flags: list[str] = []
    if cpus and cpus.lower() not in {"0", "none", "off"}:
        flags += ["--cpus", cpus]
    if memory and memory.lower() not in {"0", "none", "off"}:
        flags += ["--memory", memory]
        if memory_swap and memory_swap.lower() not in {"0", "none", "off"}:
            flags += ["--memory-swap", memory_swap]
    if pids_limit and pids_limit.lower() not in {"0", "none", "off"}:
        flags += ["--pids-limit", pids_limit]
    return flags


def build_codex_cmd(
    *,
    workspace_dir: Path,
    run_dir: Path,
    container_cd: str,
    model: str,
    effort: str,
    codex_home: Path,
    images: list[str] | None = None,
    session_id: str | None = None,
    image_name: str = DEFAULT_IMAGE,
    network: str = DEFAULT_NETWORK,
    gateway: str | None = None,
    container_name: str | None = None,
    docker_cpus: str | None = None,
    docker_memory: str | None = None,
    docker_memory_swap: str | None = None,
    docker_pids_limit: str | None = None,
) -> list[str]:
    """Build the `docker run ... codex exec` command for the airtight gateway.

    The container starts as root ONLY to point its default route at the egress
    gateway, then drops to the unprivileged image user (via setpriv) -- which has
    no sudo and cannot re-route -- to run codex. So the model is confined to the
    gateway's domain allowlist with no bypass. supports_websockets=false forces
    the HTTPS/SSE transport (codex's ChatGPT websocket won't traverse the gateway
    and codex ignores HTTP_PROXY). The run dir is mounted RO with
    the workspace overlaid RW. CODEX_HOME is mounted RW for auth and session.
    `images` are in-container /run/... paths passed to codex via -i.
    """
    run_dir = run_dir.resolve()
    workspace_dir = workspace_dir.resolve()
    rel = workspace_dir.relative_to(run_dir)
    gw = gateway_ip(gateway)
    if not gw:
        raise RuntimeError(
            "codex gateway unavailable; run codex_backend/gateway_up.sh first"
        )
    uid, gid = os.getuid(), os.getgid()
    wrapper = (
        f"ip route replace default via {gw} >/dev/null 2>&1; "
        f"exec setpriv --reuid={uid} --regid={gid} --init-groups "
        f"env HOME=/home/user CODEX_HOME=/home/user/.codex \"$@\""
    )
    codex_argv: list[str] = [
        "codex", "-m", model, "-c", f"model_reasoning_effort={effort}",
        *_PROVIDER_FLAGS,
        "--disable", "plugins",
        "--dangerously-bypass-approvals-and-sandbox",
        "exec", "--skip-git-repo-check", "--ignore-user-config",
        "--ignore-rules", "--cd", container_cd, "--json",
    ]
    for im in (images or []):
        codex_argv += ["-i", im]
    if session_id:
        codex_argv += ["resume", session_id, "-"]
    resource_flags = _docker_resource_flags(
        cpus=docker_cpus,
        memory=docker_memory,
        memory_swap=docker_memory_swap,
        pids_limit=docker_pids_limit,
    )
    cmd = [
        "docker", "run", "--rm", "-i",
        *resource_flags,
        "--network", network,
        "--cap-add", "NET_ADMIN",
        "--dns", gw,
        "--user", "0",
        "-v", f"{run_dir}:/run:ro",
        "-v", f"{run_dir / rel}:/run/{rel}:rw",
        "-v", f"{codex_home}:/home/user/.codex",
        image_name,
        "bash", "-lc", wrapper, "_", *codex_argv,
    ]
    if container_name:
        cmd[3:3] = ["--name", container_name]
    return cmd


def run_codex_turn(
    *,
    prompt: str,
    workspace_dir: Path,
    run_dir: Path,
    container_cd: str,
    model: str,
    effort: str,
    codex_home: Path,
    images: list[str] | None = None,
    session_id: str | None = None,
    timeout_s: int | None = None,
    image_name: str = DEFAULT_IMAGE,
    network: str = DEFAULT_NETWORK,
    gateway: str | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    docker_cpus: str | None = None,
    docker_memory: str | None = None,
    docker_memory_swap: str | None = None,
    docker_pids_limit: str | None = None,
) -> dict[str, Any]:
    """Run one codex turn to completion. Prompt is delivered on stdin.

    Returns {session_id, returncode, turn_failed, usage, duration_s, reason}.
    The model's actual output is the files it wrote into the workspace. This
    only reports whether the turn ran cleanly, plus the session id for resume.
    """
    container_name = f"arc-codex-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    cmd = build_codex_cmd(
        workspace_dir=workspace_dir, run_dir=run_dir, container_cd=container_cd,
        model=model, effort=effort, codex_home=codex_home, images=images,
        session_id=session_id, image_name=image_name, network=network,
        gateway=gateway, container_name=container_name,
        docker_cpus=docker_cpus,
        docker_memory=docker_memory,
        docker_memory_swap=docker_memory_swap,
        docker_pids_limit=docker_pids_limit,
    )
    t0 = time.time()
    out_text = ""
    err_text = ""
    timed_out = False
    rc = -1
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        try:
            out_text, err_text = proc.communicate(input=prompt, timeout=timeout_s)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            try:
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                )
            except Exception:
                pass
            out_text, err_text = proc.communicate()
            rc = -1
    except Exception as e:
        return {"session_id": session_id, "returncode": -1, "turn_failed": True,
                "usage": {}, "duration_s": round(time.time() - t0, 1),
                "reason": f"docker-spawn: {type(e).__name__}: {e}"}

    if stdout_path is not None:
        try:
            stdout_path.write_text(out_text)
        except Exception:
            pass
    if stderr_path is not None:
        try:
            stderr_path.write_text(err_text)
        except Exception:
            pass

    events, turn_failed, usage = _scan_events(out_text)
    new_sid = _extract_session_id(events) or session_id
    remote_compact_failed = _is_remote_compact_failure(out_text, err_text)
    quota_limited = _is_quota_limited(out_text, err_text)
    retryable_infra_failure = timed_out or turn_failed or rc != 0
    reason = ("timed_out" if timed_out
              else "remote_compact_failed" if remote_compact_failed
              else "quota_limited" if quota_limited
              else "retryable_infra_failure" if retryable_infra_failure
              else "turn_failed" if (turn_failed or rc != 0)
              else "ok")
    return {
        "session_id": new_sid,
        "returncode": rc,
        "turn_failed": turn_failed or rc != 0 or timed_out,
        "remote_compact_failed": remote_compact_failed,
        "quota_limited": quota_limited,
        "retryable_infra_failure": retryable_infra_failure,
        "usage": usage,
        "duration_s": round(time.time() - t0, 1),
        "reason": reason,
    }
