#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

docker build -q -t claude-gateway -f "$here/Dockerfile.gateway" "$here" >/dev/null
echo "[gateway] image built"

docker network inspect claude-filtered >/dev/null 2>&1 \
  || docker network create --opt com.docker.network.bridge.enable_ip_masquerade=false claude-filtered
docker network inspect claude-egress >/dev/null 2>&1 \
  || docker network create claude-egress

docker rm -f claude-gateway >/dev/null 2>&1 || true
docker run -d --name claude-gateway \
  --restart unless-stopped \
  --cap-add NET_ADMIN \
  --sysctl net.ipv4.ip_forward=1 \
  --network claude-egress claude-gateway >/dev/null
docker network connect claude-filtered claude-gateway

sleep 2
GWIP=$(docker inspect -f '{{(index .NetworkSettings.Networks "claude-filtered").IPAddress}}' claude-gateway)
echo "$GWIP" > "$here/gateway_internal_ip.txt"
echo "[gateway] running; claude-filtered IP = $GWIP (written to gateway_internal_ip.txt)"
docker ps --format '{{.Names}} {{.Status}}' | grep claude-gateway
