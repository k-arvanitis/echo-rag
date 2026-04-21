setup:
	uv sync

dev:
	uv run streamlit run app.py

test:
	uv run pytest tests/

services:
	docker compose up -d

services-down:
	docker compose down
