.PHONY: build up down down-v logs ps backend frontend test clean

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

down-v:
	@echo "WARNING: this deletes the mysql_data volume (all database data)."
	docker compose down -v

logs:
	docker compose logs -f

ps:
	docker compose ps

backend:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	streamlit run frontend/app.py --server.address=0.0.0.0 --server.port=8501

test:
	pytest -q

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
