#!/usr/bin/env bash
#   - api.anthropic.com         MUST be reachable (allowlisted)
#   - example.com / 1.1.1.1     MUST be blocked (not allowlisted)
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

GW=$(cat "$here/gateway_internal_ip.txt" 2>/dev/null || true)
if [ -z "${GW}" ]; then
  echo "FAIL: no gateway IP; run gateway_up.sh first"; exit 1
fi
uid=$(id -u); gid=$(id -g)

run_probe() {
  # $1 = shell snippet run as the unprivileged user after route is set
  docker run --rm \
    --network claude-filtered --cap-add NET_ADMIN --dns "$GW" --user 0 \
    claude-agent bash -lc \
    "ip route replace default via $GW >/dev/null 2>&1; \
     exec setpriv --reuid=$uid --regid=$gid --init-groups bash -lc '$1'"
}

echo "[egress] allowlisted (api.anthropic.com:443) -- expect REACHABLE:"
# /dev/tcp connect test; 0 = connected. DNS via gateway resolves + nftset-allows.
if run_probe 'timeout 8 bash -c "exec 3<>/dev/tcp/api.anthropic.com/443" && echo ALLOW_OK || echo ALLOW_FAIL'; then :; fi

echo "[egress] non-allowlisted (example.com:443) -- expect BLOCKED:"
run_probe 'timeout 8 bash -c "exec 3<>/dev/tcp/example.com/443" && echo BLOCK_FAIL || echo BLOCK_OK'

echo "[egress] non-allowlisted raw IP (1.1.1.1:443) -- expect BLOCKED:"
run_probe 'timeout 8 bash -c "exec 3<>/dev/tcp/1.1.1.1/443" && echo BLOCK_FAIL || echo BLOCK_OK'

echo "[egress] done. Want: ALLOW_OK, BLOCK_OK, BLOCK_OK"
