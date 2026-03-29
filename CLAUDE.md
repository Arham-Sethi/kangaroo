# Kangaroo Shift — Claude Code Instructions

## Project Overview
Kangaroo Shift is a middleware context-management platform for seamless LLM switching.

## First Step Every Session
Read `PROGRESS.md` to determine current phase and task.

## Key Commands
- Backend: `cd backend && uvicorn app.main:create_app --factory --reload`
- Frontend: `cd frontend && npm run dev`
- Tests: `cd backend && pytest --cov=app`
- Full stack: `docker compose up`

## Conventions
- Python 3.12+, type hints everywhere, docstrings on all functions
- Pydantic v2 for all schemas
- FastAPI with async endpoints
- Conventional commits: feat:, fix:, docs:, test:, chore:, refactor:
- 80%+ test coverage per module
- All config via environment variables
