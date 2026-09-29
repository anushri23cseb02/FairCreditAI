#!/usr/bin/env sh
# Verify a running deployment. Uses the host's Python if present (standard library only),
# otherwise runs the same checks inside the backend container.
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
  python3 scripts/verify_deployment.py "$@"
elif command -v python >/dev/null 2>&1; then
  python scripts/verify_deployment.py "$@"
else
  echo "No Python on this computer - running the checks inside the backend container."
  docker compose exec backend python scripts/verify_deployment.py --inside-container "$@"
fi
