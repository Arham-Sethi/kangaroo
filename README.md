# Kangaroo Shift

Middleware context-management platform that captures, compresses, structures, encrypts, and reconstructs conversational context so users can move seamlessly between LLMs or sessions without losing accumulated knowledge.

## Quick Start

```bash
cp .env.example .env
docker compose up
```

- Backend API: http://localhost:8000/docs
- Frontend: http://localhost:3000

## Development

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest --cov=app
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### CLI
```bash
cd cli
pip install -e ".[dev]"
kangaroo --help
```

## Architecture

- **Backend:** Python 3.12+ (FastAPI)
- **Context Engine:** spaCy, sentence-transformers, LangChain
- **Database:** PostgreSQL 16 + pgvector
- **Cache:** Redis 7+
- **Search:** Meilisearch
- **Frontend:** React 18 + Next.js 14 + TailwindCSS
- **Encryption:** AES-256-GCM via Argon2id key derivation

## License

Proprietary — All rights reserved.
