#!/usr/bin/env sh
# Start FairCredit AI (Linux / macOS). Safe to re-run; never deletes data.
cd "$(dirname "$0")" || exit 1
if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running (or not installed). Start Docker and try again." >&2
  exit 1
fi
sh scripts/init_env.sh || exit $?
docker compose up -d --build || exit $?
docker compose ps
echo
echo "Frontend : http://localhost:8501"
echo "API docs : http://localhost:8000/docs"
echo "Other computers on your network: http://<this-computer-IP>:8501  (see DEPLOYMENT.md)"
echo "Containers can take up to a minute to become healthy. Check with: ./verify.sh"
