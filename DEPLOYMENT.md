# FairCredit AI — Deployment Guide

This guide only covers **running the existing application on another computer**. It does not change how the application behaves.

```
 User's browser
      │  http://localhost:8501   (or http://HOST-IP:8501 from another computer)
      ▼
 ┌──────────────────┐   HTTP (Docker network)   ┌──────────────────┐   SQL    ┌──────────────┐
 │ Streamlit        │ ────────────────────────► │ FastAPI          │ ───────► │ MySQL 8.0    │
 │ container        │      http://backend:8000  │ container        │  mysql:  │ container    │
 │ faircredit_      │                           │ faircredit_      │  3306    │ faircredit_  │
 │ frontend :8501   │                           │ backend :8000    │          │ mysql        │
 └──────────────────┘                           └────────┬─────────┘          └──────┬───────┘
                                                         │ loads at startup          │
                                                 ./models  (trained models)     volume: mysql_data
                                                 ./data    (datasets, reports)
```

## 0. Current production deployment (live)

The application is deployed and publicly reachable right now:

| What | URL |
|---|---|
| **App (Streamlit)** | https://faircredit-frontend.onrender.com |
| **API** | https://faircredit-backend.onrender.com |
| **API docs (Swagger)** | https://faircredit-backend.onrender.com/docs |

**Stack:** [Render](https://render.com) hosts both `Dockerfile.backend` and `Dockerfile.frontend` as free-tier web services; [Aiven](https://aiven.io) hosts the free-tier MySQL database (TLS-only, see `DB_SSL_CA_PATH` in section L). No laptop, Docker Desktop, or local process is involved in serving this — it runs entirely on those two providers.

**Access:** the whole app is behind HTTP Basic Auth (`APP_BASIC_AUTH_USER` / `APP_BASIC_AUTH_PASSWORD`, set as Render env vars — see section Q3 and the security note in section R). Credentials aren't in this file; ask whoever deployed it, or read them from the Render dashboard's Environment tab.

**Known free-tier trade-offs:**
- Render's free web services sleep after 15 minutes of inactivity; the next request wakes them in 30-60 seconds. This is normal, not a bug.
- Aiven's free MySQL can be powered off after prolonged inactivity and needs a **manual** restart from the Aiven console (unlike Render, it does not auto-wake). If `/health/detailed` reports `"database": "unreachable"` after the app has been idle a long time, this is almost certainly why.
- **Auto-deploy on `git push` is not yet wired up** — these two services were created via Render's API rather than through the dashboard's GitHub connection flow, so Render's GitHub App isn't authorized for this repo yet. Until that's done (Account Settings → GitHub → Configure → add this repo), redeploys after a push must be triggered manually (dashboard "Manual Deploy", or `POST /v1/services/{id}/deploys` with a Render API key).

## 0b. ML/data audit findings (2026-09-29)

A full technical audit of the data pipeline, training code, fairness math, explainability, and tests was performed against the actual code (not a superficial review — metrics below were independently recomputed from the persisted confusion matrices to confirm they aren't fabricated). Summary:

**Verified sound, no changes needed:**
- Leakage-prone raw columns (`denial_reason_name_*`, `rate_spread`, `purchaser_type_name`, etc.) are deliberately excluded from HMDA model features, with reasons documented in `backend/ml/feature_config.py`.
- Both tracks fit all preprocessing (imputation, scaling, one-hot encoding) only inside `Pipeline.fit(X_train, ...)` — no train/val/test contamination.
- HMDA explanations are exact linear-model coefficient decomposition, not an approximation — verified by hand against `backend/ml/explain.py`.
- Fairness metrics (`backend/ml/fairness.py`) were checked against `tests/test_fairness.py`'s known-disparity synthetic cases and match Fairlearn's standard definitions exactly.
- Recomputing accuracy/precision/recall/F1 from the stored confusion matrices in `models/metadata/model_v1.json` and `models/metadata/credit_card_model_comparison.json` reproduces the stored values exactly — the reported metrics are real, not invented.

**Known, disclosed limitations (not fixed — see reasons):**
1. **Credit-card model selection uses held-out test-set metrics rather than validation-set metrics** (`backend/ml/credit_card/trainer.py`), which is a methodological gap — test data should only be used for final reporting, not for picking the winning model. Recomputing the same selection formula on validation-set metrics instead of test-set metrics gives Logistic Regression 0.5998 vs XGBoost's 0.6025 — meaning the methodologically correct approach would select XGBoost instead, by a razor-thin margin. **Decision: left as-is** (2026-09-29) — the currently-deployed Logistic Regression model stays selected, since the margin is negligible either way and switching would trade exact coefficient-based explanations for approximate SHAP-based ones. Revisit if more candidate models or more training data are added later.
2. **HMDA outlier capping** (`loan_amount_000s`, `applicant_income_000s` clipped at the 99.5th percentile in `scripts/prepare_data.py`) is computed on the full dataset before the train/val/test split — a minor, low-impact pre-split statistic leak (only affects clipping bounds, not fitted model parameters). **Not fixed**: the raw HMDA source file (`data/raw/Washington_State_HDMA-2016.csv`, ~260MB, intentionally not committed) is not present in any environment this audit had access to, so this cannot be verified further or corrected without that file.

**Dashboard design:** already uses a restrained, professional pastel fintech theme (`frontend/assets/style.css`) with no clutter, gradients, or decorative-only elements — a redesign was considered but not warranted; no concrete defects were found in navigation, forms, or error handling (manually tested invalid input, missing fields, and malformed JSON against the live API — all return clean structured 422 errors, never a raw stack trace).

## Q3. How this was actually set up (Render + Aiven)

Reference for reproducing or modifying this deployment:

**Aiven MySQL:**
```
avn service create faircredit-mysql --project <project> --service-type mysql --plan free-1-1gb --cloud do-sgp
avn project ca-get --project <project> --target-filepath backend/aiven-ca.pem   # committed to the repo -- public CA cert, not a secret
avn service user-password-reset faircredit-mysql --project <project> --username avnadmin --new-password '<generated>'
```
Free MySQL plans are only available on DigitalOcean/UpCloud regions, not AWS/GCP/Azure — Aiven's API returns a clear 403 with the allowed cloud list if you pick an unsupported one.

**Render (backend + frontend):** created directly via Render's REST API (`POST /v1/services`) rather than `render.yaml`, because the Render CLI's `services create` has no flag for a custom Dockerfile path — only the Blueprint YAML and the raw API support `serviceDetails.envSpecificDetails.dockerfilePath`. Each service points at a different Dockerfile in the same repo:
```json
{"type": "web_service", "name": "faircredit-backend", "repo": "https://github.com/<org>/FairCreditAI",
 "serviceDetails": {"runtime": "docker", "plan": "free", "region": "singapore",
   "envSpecificDetails": {"dockerfilePath": "./Dockerfile.backend", "dockerContext": "."},
   "healthCheckPath": "/health"}}
```
Env vars were set with `PUT /v1/services/{id}/env-vars`, then a deploy was explicitly triggered with `POST /v1/services/{id}/deploys` -- setting env vars alone does **not** restart the already-building/running instance.

`render.yaml` (committed at the repo root) documents the same two services declaratively and is the easier path if creating this fresh through the Render dashboard's Blueprint flow instead of the API.

## A. Prerequisites

Only one thing needs installing:

- **Docker Desktop** (Windows / macOS) or **Docker Engine + Compose plugin** (Linux).

You do **not** need Python, pip, MySQL, or Git on the computer that runs the application.

Check it works:

```
docker --version
docker compose version
```

Docker must actually be **running** (on Windows/macOS: Docker Desktop open, tray icon says it is running).

## B. Windows 10 / 11 (Docker Desktop, PowerShell or Command Prompt)

1. Install Docker Desktop, start it, and wait until it says it is running.
2. Extract/copy the project folder anywhere, for example `C:\FairCreditAI`.
3. Open that folder, then either **double-click `start.bat`**, or in PowerShell:

```powershell
cd "C:\path\to\FairCreditAI"
docker compose up -d --build
```

If you use the manual command and there is no `.env` file yet, first run `copy .env.example .env` and edit the two passwords (see section L). `start.bat` does this for you and generates random passwords.

## C. Linux

```bash
cd /path/to/FairCreditAI
./start.sh                     # or: cp .env.example .env  (edit passwords)  &&  docker compose up -d --build
```

If you get "permission denied" on the script: `chmod +x start.sh stop.sh verify.sh scripts/init_env.sh`. If Docker needs sudo on your system, prefix the docker commands with `sudo` (or add your user to the `docker` group).

## D. macOS

Same as Linux, with Docker Desktop for Mac running:

```bash
cd /path/to/FairCreditAI
./start.sh
```

## E. First startup

```
docker compose up -d --build
```

The first build downloads base images and Python packages, so it can take **several minutes** (about 10 on a normal connection). Later starts take under a minute. Containers start in order — MySQL → backend → frontend — each waiting for the previous one's health check, not for a fixed delay.

```
docker compose ps        # all three should say "healthy"
```

## F. Accessing the application

| What | Same computer | Another computer on the same network |
|---|---|---|
| **Application (Streamlit)** | http://localhost:8501 | http://HOST-IP:8501 |
| **API** | http://localhost:8000 | http://HOST-IP:8000 |
| **API documentation (Swagger)** | http://localhost:8000/docs | http://HOST-IP:8000/docs |

**Finding HOST-IP** (the IP of the computer that is running Docker):

- Windows: `ipconfig` → the *IPv4 Address* of your active Wi-Fi/Ethernet adapter (usually `192.168.x.x` or `10.x.x.x`).
- Linux: `hostname -I`
- macOS: `ipconfig getifaddr en0` (Wi-Fi; try `en1` if empty)

Other computers can only connect while the host computer is on, connected to the same network, and Docker is running.

**Windows Firewall.** If another computer cannot connect but `localhost` works on the host, Windows Firewall is probably blocking the port. Do not disable the firewall. Allow just the application port, from an **Administrator** PowerShell:

```powershell
New-NetFirewallRule -DisplayName "FairCredit AI (Streamlit)" -Direction Inbound -Protocol TCP -LocalPort 8501 -Action Allow -Profile Private
```

Users only need port **8501**. Open 8000 the same way only if other people must call the API directly.

**Optional desktop shortcut:** create a browser shortcut to `http://localhost:8501` (Chrome/Edge: ⋮ → *Save and share* → *Create shortcut*). This only opens the same web page.

## G. Ports used

| Port | Service | Reachable from |
|---|---|---|
| 8501 | Streamlit frontend | all network interfaces |
| 8000 | FastAPI backend | all network interfaces |
| 3307 → 3306 | MySQL | **this computer only** (`127.0.0.1:3307`) |

MySQL is deliberately published on the loopback interface only, so it cannot be reached from the network. Containers talk to it internally as `mysql:3306`. To connect a desktop MySQL client on the same computer, use host `127.0.0.1`, port `3307`. (Host port 3307 is used instead of 3306 so it does not clash with a MySQL already installed on the host.)

## H. Stopping

```
docker compose down          # or stop.bat / ./stop.sh
```

Containers are removed, **data is kept** (see section K). Never add `-v` unless you intend to delete the database.

## I. Restarting

```
docker compose restart
```

The containers use `restart: unless-stopped`, so after a reboot they come back automatically once Docker is running (on Windows/macOS enable *Start Docker Desktop when you sign in* in Docker Desktop's settings).

## J. Updating

1. Stop nothing yet. Copy the new project files over the old ones, **keeping** your `.env` and the `data/`, `models/`, `reports/` folders.
2. Rebuild and restart:
   ```
   docker compose up -d --build
   ```
The MySQL volume, models and data are untouched. If you deploy with git: `git pull` then the same command.

## K. What persists and what does not

| Item | Where it lives | Survives `down`/`up` | Survives `down -v` |
|---|---|---|---|
| Prediction audit history (both tables) | Docker volume `mysql_data` | yes | **NO — deleted** |
| Trained models, registry, metadata | `./models` (host folder) | yes | yes |
| Datasets, processed splits, reports | `./data` (host folder) | yes | yes |
| Generated reports | `./reports` (host folder) | yes | yes |
| Application code | inside the images | rebuilt by `--build` | rebuilt |
| Container logs | Docker log store (rotated: 3 × 10 MB per service) | no (removed with the container) | no |
| Uploaded batch CSVs | not stored — scored in memory | n/a | n/a |

Nothing in normal startup drops or deletes tables. The backend only creates missing tables (`CREATE TABLE IF NOT EXISTS` behaviour).

## L. Configuration (`.env`)

`.env` is never committed and is not baked into images. Create it with `start.bat` / `./start.sh` (random passwords), or copy `.env.example` and change:

```
MYSQL_PASSWORD=...        # use a long random value
MYSQL_ROOT_PASSWORD=...   # use a different long random value
```

All variables are documented in `.env.example`. Inside Docker the backend reaches the database at `mysql:3306` (set by `docker-compose.yml`), never `localhost`.

> **Important:** MySQL stores its passwords inside the `mysql_data` volume the *first* time it starts. Changing `MYSQL_PASSWORD` in `.env` later does **not** change the stored password and the backend will be refused ("Access denied"). Either restore the old `.env`, or reset the database with `docker compose down -v` (this deletes prediction history).

## M. Verifying a deployment

```
verify.bat            (Windows)         ./verify.sh          (Linux/macOS)
```
or `python scripts/verify_deployment.py`. It checks Docker, container health, environment variables, model files, database connectivity, backend, frontend, and sends one real test prediction to each model. It ends with `DEPLOYMENT VERIFICATION PASSED` or `FAILED` and states the reason for every failure. Options: `--no-predict` (the two test predictions write two rows to the audit tables; use this flag to avoid that), `--skip-docker`, `--inside-container`.

If the computer has no Python, `verify.bat` / `verify.sh` run the same checks inside the backend container automatically.

`scripts/verify_project.py` and `pytest` are **development** tools: they need the project's `tests/` folder and Python packages, which are intentionally not inside the runtime images. Run them from a development checkout (`pip install -r requirements-dev.txt`, then `pytest`), not with `docker compose exec`.

## N. Logs

```
docker compose logs                 # everything
docker compose logs backend         # API, model loading, database
docker compose logs mysql
docker compose logs frontend
docker compose logs -f backend      # follow live
```

Passwords are not written to the logs. With `APP_DEBUG=false` (the default in `.env.example`) the logs are short and readable and contain no applicant input values. If your logs are full of SQL statements, your `.env` has `APP_DEBUG=true` (older copies did): change it to `false` and run `docker compose up -d --force-recreate backend`. Model loading appears as `Loaded model version ...` / `Loaded credit-card models ...`; a missing model appears as a clear "not trained yet" warning.

## O. Backup and restore of the database

**Backup** (creates `backup.sql` in the current folder). The file is produced inside the container and then copied out, which avoids the UTF-16 encoding corruption that PowerShell's `>` redirection causes:

```
docker compose exec mysql sh -c 'mysqldump -uroot -p"$MYSQL_ROOT_PASSWORD" --single-transaction "$MYSQL_DATABASE" > /tmp/backup.sql'
docker compose cp mysql:/tmp/backup.sql ./backup.sql
```

**Restore** (into a running, healthy stack):

```
docker compose cp ./backup.sql mysql:/tmp/backup.sql
docker compose exec mysql sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" "$MYSQL_DATABASE" < /tmp/backup.sql'
```

**Models and data** are ordinary folders (`models/`, `data/`); back them up by copying them.

## P. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `failed to connect to the docker API ... dockerDesktopLinuxEngine` | Docker Desktop is not running. Start it, wait until it says it is running, retry. |
| `docker compose ps` prints nothing | Containers were never started. Run `docker compose up -d`. |
| `port is already allocated` on 8501 / 8000 / 3307 | Another program uses that port. Stop it, or change the left-hand number in `docker-compose.yml` (e.g. `"8502:8501"`). |
| `mysql` not healthy | `docker compose logs mysql`. First start needs ~20–60 s. |
| Backend logs `Access denied for user` | `.env` passwords do not match the passwords stored in the existing `mysql_data` volume — see the note in section L. |
| Backend unhealthy / restarting | `docker compose logs backend`. Look for a missing model or database error. |
| Prediction returns **503** "not trained yet" | A model file is missing from `./models`. Restore the `models/` folder (registry + artifacts). Retraining needs the raw datasets: see README "Training". No substitute model is ever created automatically. |
| Credit-card fairness / SHAP pages say "unavailable" | `data/processed/credit_card_*.csv` is missing. Restore it, or run `docker compose exec backend python scripts/prepare_credit_card_data.py`. |
| Retrain button says training data missing (HMDA) | `data/processed/train.csv` etc. are not shipped (large). Put `Washington_State_HDMA-2016.csv` in `data/raw/` and run `docker compose exec backend python scripts/prepare_data.py`. |
| Frontend loads but shows "Cannot reach backend" | Backend not healthy yet or crashed: `docker compose ps`, `docker compose logs backend`. |
| Works on `localhost`, not from another computer | Firewall — see section F. Also confirm both computers are on the same network and you used the host's IP, not `localhost`. |
| `start.bat` says an existing MySQL volume was found | You have no `.env` but an old database exists. Copy your old `.env` back, or `docker compose down -v` to reset. |
| First build is very slow | Normal (downloads packages). Later builds use the cache. |

## Q. Deploying to a server (VPS / cloud VM)

The stack runs anywhere Docker and Compose run (any Linux VM). No cloud-specific service is required.

1. Install Docker Engine + Compose plugin, copy the project, run `./start.sh`.
2. Keep MySQL private (already loopback-only). Expose only what you need — ideally only the reverse proxy (443), not 8501/8000 directly (remove those `ports:` lines or firewall them).
3. Use strong passwords in `.env`; keep `APP_DEBUG=false` (the default in `.env.example`); restrict `CORS_ORIGINS` to your real site URL.
4. This application has **no user login by default**. Do not put it on the public internet without authentication in front of it — either set `APP_BASIC_AUTH_USER` / `APP_BASIC_AUTH_PASSWORD` (built-in, see section Q2) or add basic auth / SSO at the reverse proxy.

### HTTPS behind Nginx (optional, not enabled by default)

Streamlit needs WebSocket upgrade headers to work through a proxy:

```nginx
server {
    listen 443 ssl;
    server_name credit.example.com;
    # ssl_certificate / ssl_certificate_key ... (e.g. from Let's Encrypt / certbot)

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
    }
}
```

## Q2. Deploying to Railway (public HTTPS, no laptop/Docker Desktop dependency)

This uses the same `Dockerfile.backend` / `Dockerfile.frontend` as sections A-P — Railway
just builds and runs them on Railway's infrastructure instead of your machine. Repo:
`https://github.com/<your-account>/FairCreditAI`.

**1. Create the project and three services**, all from the same GitHub repo:

- New Project → **Deploy from GitHub repo** → select `FairCreditAI`.
- Railway creates one service from the repo root. Rename it `backend`, then in its
  **Settings → Build**, set **Dockerfile Path** to `Dockerfile.backend`.
- **+ New → GitHub Repo** → same repo again → rename it `frontend` → set **Dockerfile Path**
  to `Dockerfile.frontend`.
- **+ New → Database → Add MySQL.**

Each service redeploys automatically on every `git push` to `main`, once connected this way —
that is the whole CI/CD loop (GitHub Actions in `.github/workflows/ci.yml` runs the tests;
Railway does the deploy).

**2. Backend service → Variables:**

```
DATABASE_URL=mysql+pymysql://${{MySQL.MYSQLUSER}}:${{MySQL.MYSQLPASSWORD}}@${{MySQL.MYSQLHOST}}:${{MySQL.MYSQLPORT}}/${{MySQL.MYSQLDATABASE}}
APP_ENV=production
APP_DEBUG=false
CORS_ORIGINS=https://<frontend-service>.up.railway.app
APP_BASIC_AUTH_USER=<choose a username>
APP_BASIC_AUTH_PASSWORD=<choose a strong password>
```

`${{MySQL.*}}` are Railway's reference variables — they pull the actual values from the MySQL
service automatically; don't hand-type database credentials. Do **not** set `PORT` yourself —
Railway injects it, and the Dockerfile already reads `$PORT` (see the Docker fix below).

**3. Frontend service → Variables:**

```
BACKEND_URL=https://<backend-service>.up.railway.app
APP_BASIC_AUTH_USER=<same username as backend>
APP_BASIC_AUTH_PASSWORD=<same password as backend>
```

**4. Generate public domains:** in each of `backend` and `frontend` → **Settings → Networking
→ Generate Domain**. These are the real `https://*.up.railway.app` URLs — HTTPS is automatic,
no certificate setup needed. The frontend's is the one you share with users.

**5. Health checks:** set the backend service's health check path to `/health` (Settings →
Deploy) so Railway restarts it automatically if it ever stops responding.

**6. Why this works without your laptop:** `Dockerfile.backend` now `COPY`s `models/` and
`data/` into the image at build time (previously they only existed via a docker-compose bind
mount, which has nothing to bind to on Railway) — so the deployed backend loads the same
trained models this repo ships, with no volume or manual upload step.

**7. Verify the live deployment** from any machine:

```
python scripts/verify_deployment.py --skip-docker \
  --backend-url https://<backend-service>.up.railway.app \
  --frontend-url https://<frontend-service>.up.railway.app \
  --basic-auth <username>:<password>
```

**8. Auth note:** setting `APP_BASIC_AUTH_USER`/`APP_BASIC_AUTH_PASSWORD` is what turns on the
login gate described in section R below. Leaving them unset on Railway would put this
no-login app on the open internet — not recommended, since `/train` and
`/credit-card/train` would be publicly callable by anyone with the URL.

**9. Cost:** Railway is not free for something running 24/7 — three always-on services
(backend + frontend + MySQL) exceed the free trial credit; check Railway's current pricing
before leaving this running long-term.

## R. Security summary

- Secrets live only in `.env` (git-ignored, excluded from Docker images by `.dockerignore`) locally, or in the host platform's environment variables (e.g. Railway's Variables tab) in the cloud — never in source code.
- MySQL is not reachable from the network locally (loopback publish only); on Railway, don't expose the MySQL service's public networking unless you genuinely need external DB access.
- Passwords are not printed by any script or log. Keep `APP_DEBUG=false`: `true` logs SQL statements including applicant input values.
- Containers currently run as root inside the container (the images were not changed for this deployment). See the limitations in the delivery report.
- Optional built-in HTTP Basic Auth (`APP_BASIC_AUTH_USER` / `APP_BASIC_AUTH_PASSWORD`, both blank by default): when set, gates the whole API (except `/health`) and every Streamlit page. See section Q2.
