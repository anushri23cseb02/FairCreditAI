#!/usr/bin/env sh
# Creates .env from .env.example with freshly generated database passwords.
# Never overwrites an existing .env. Run from the project root (start.sh does this).
set -e

if [ -f .env ]; then
  echo ".env already exists - leaving it untouched."
  exit 0
fi
if [ ! -f .env.example ]; then
  echo "ERROR: .env.example not found. Run this from the project root." >&2
  exit 1
fi

# A MySQL data volume keeps the passwords it was FIRST created with. If one already
# exists, generating new passwords would lock the backend out of the database.
if command -v docker >/dev/null 2>&1 && docker volume ls -q 2>/dev/null | grep -q "mysql_data"; then
  echo "WARNING: an existing MySQL data volume was found, but there is no .env file." >&2
  echo "A new random password would not match the password stored in that volume." >&2
  echo "Choose ONE of these:" >&2
  echo "  1) Copy your previous .env into this folder, then run this script again." >&2
  echo "  2) Reset the database (DELETES stored prediction history):" >&2
  echo "       docker compose down -v" >&2
  echo "     then run this script again." >&2
  exit 2
fi

gen() { (LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 24) 2>/dev/null; }
ROOT_PW="$(gen)"; USER_PW="$(gen)"
# Replace the longer placeholder first so 'change_me' does not clobber 'change_me_root'.
sed -e "s/change_me_root/${ROOT_PW}/g" -e "s/change_me/${USER_PW}/g" .env.example > .env
echo "Created .env with generated database passwords (not shown, not committed)."
