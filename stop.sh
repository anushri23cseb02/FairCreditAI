#!/usr/bin/env sh
# Stop FairCredit AI. Data (MySQL volume, models, data/) is kept.
cd "$(dirname "$0")" || exit 1
docker compose down
