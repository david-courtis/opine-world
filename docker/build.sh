#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

docker build \
  --build-arg USER_ID="$(id -u)" \
  --build-arg GROUP_ID="$(id -g)" \
  -t claude-agent \
  -f "$here/Dockerfile.agent" "$here"
echo "[build] claude-agent image built (uid=$(id -u) gid=$(id -g))"
