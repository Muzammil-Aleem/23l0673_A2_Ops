# Task 6.3 / 6.4 / 6.5 — operating the Compose stack. Run block by block from the repo root.

# ---------- 6.3 start + status ----------
docker compose up -d --build
docker compose ps                         # both services "healthy"
curl.exe -s http://localhost:8000/health

# proof the DB port is closed from outside: no host port is published ...
docker compose port db 5432               # prints nothing / error: no public port
Test-NetConnection localhost -Port 5432   # TcpTestSucceeded : False
# ... but the web container reaches it by service name on the internal network
docker compose exec web python -c "import socket; s=socket.create_connection(('db', 5432), 3); print('db:5432 reachable from web')"

# user inside the container (6.2): must NOT be root
docker compose exec web whoami
docker compose exec web id

# ---------- 6.4 image analysis ----------
docker images visionguard-web
docker history visionguard-web:latest --format "table {{.Size}}\t{{.CreatedBy}}" | Select-Object -First 15
docker build -f Dockerfile.unoptimized -t visionguard-web:before .
docker images visionguard-web             # compare :before vs :latest

# ---------- 6.5 operations ----------
docker compose stop web
docker compose start web
docker compose restart web
docker compose logs --tail 50 web
docker compose logs --tail 50 db
docker compose exec web sh                # shell in the web service (exit to leave)
docker compose exec db psql -U visionguard -d visionguard -c "SELECT id, created_at, label, round(confidence::numeric,3), model_version FROM predictions ORDER BY id DESC LIMIT 10;"

# ---------- 6.5 break it on purpose: missing model file ----------
Rename-Item models\champion.keras champion.keras.bak
docker compose restart web
docker compose logs --tail 30 web         # -> "MODEL NOT LOADED: FileNotFoundError ... champion.keras"
curl.exe -s http://localhost:8000/health  # -> 503, "status": "degraded"
Rename-Item models\champion.keras.bak champion.keras
docker compose restart web
docker compose logs --tail 10 web         # -> "predictor ready"

# ---------- 6.5 break it on purpose (alternative): wrong DB password ----------
# edit .env: POSTGRES_PASSWORD=wrong   (the DB volume still has the old password)
docker compose up -d web
docker compose logs --tail 30 web         # -> 'password authentication failed for user "visionguard"'
# fix: restore the correct password in .env, then
docker compose up -d web

# ---------- stop everything (named volume keeps the history) ----------
docker compose down                        # add -v ONLY if you want to delete the database
