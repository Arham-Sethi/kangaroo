# Kangaroo Shift Build Progress

## Product Vision: Your AI Second Brain
Not just a context shifter — a persistent, intelligent memory that follows you everywhere.
Memory Engine + Multi-Model Cockpit + Universal Presence + Privacy-First Architecture.

## Current Phase: 4 (Security & Storage) — COMPLETE
## Current Task: Phase 4 Task 4 — COMPLETE
## Last Updated: 2026-03-31
## Last Updated By: Claude Code

## Completed:
- [x] Phase 1 Task 1: Repo scaffolding (full monorepo, all placeholders)
- [x] Phase 1 Task 2: Docker Compose (multi-stage Dockerfiles, health checks, .dockerignore, non-root users)
- [x] Phase 1 Task 3: Database schema & migrations (9 tables, hand-crafted Alembic migration, async session layer)
- [x] Phase 1 Task 4: UCS Pydantic models (full schema with 15+ sub-models, cross-reference validation, 34 tests)
- [x] Phase 1 Task 5: FastAPI application shell (CORS, request ID, timing middleware, structured logging, JWT auth with register/login/refresh/me)
- [x] Phase 1 Task 6: Configuration system (Pydantic Settings, env validation, production secret enforcement)
- [x] Phase 2 Task 1: Conversation Parser (4 formats, CCR models, auto-detection registry, 119 tests)
- [x] Phase 2 Task 2: Entity Extraction Pipeline (spaCy + technical NER + knowledge graph, 71 tests)
- [x] Phase 2 Task 3: Hierarchical Summarization (3-tier TF-IDF extractive, 45 tests)
- [x] Phase 2 Task 4: Context Compression (priority queue adaptive, 35 tests)
- [x] Phase 2 Task 5: UCS Generator Pipeline (full orchestrator, 24 tests)
- [x] Phase 2 Task 6: Local Processing Engine (privacy-first offline mode, 10 tests)
- [x] Phase 3 Task 1: Output Adapters (OpenAI, Claude, Gemini — UCS -> LLM format, 40 tests)
- [x] Phase 3 Task 2: Manual Import Connector (text/JSON/file with auto-detection, 19 tests)
- [x] Phase 3 Task 3: API Proxy Capture (OpenAI/Claude/Gemini transparent capture, 12 tests)
- [x] Phase 3 Task 4: API Endpoints (contexts, shifts, import — wired to engine, 0 extra tests — tested via integration)
- [x] Phase 4 Task 1: Security & Storage (encryption, sanitizer, policy engine, vault, versioning — 176 tests)
- [x] Phase 4 Task 2: Session CRUD + Audit Logging (AuditLogger, full CRUD API, ownership auth — 34 tests)
- [x] Phase 4 Task 3: API Key Management (create/list/revoke, bcrypt hashing, scope validation, X-API-Key auth dependency — 17 tests)
- [x] Phase 4 Task 4: Webhook System (CRUD, HMAC-SHA256 dispatcher, retry/backoff, auto-disable, event validation — 21 tests)

## Phase 4 Task 4 (COMPLETE — 21 new tests, 664 total):
- [x] Webhook Dispatcher (`webhook_dispatcher.py`) — HMAC-SHA256 signing, exponential backoff retry, auto-disable at 10 failures
- [x] Webhook CRUD (`webhooks.py`) — create/list/update/delete/test endpoints
- [x] Event validation — 8 valid event types, duplicate URL rejection, per-user limit (10)
- [x] Ownership enforcement — users can only manage their own webhooks

## Phase 4 Task 3 (COMPLETE — 17 new tests, 643 total):
- [x] API Key Auth Dependency (`api_key_auth.py`) — X-API-Key header auth, bcrypt verify, expiration check, last_used tracking
- [x] API Key CRUD (`api_keys.py`) — create (full key shown once), list (metadata only), revoke (soft-disable)
- [x] Scope validation — 7 valid scopes, invalid rejected with clear message
- [x] Per-user key limit (25 max), expiration validation, ownership enforcement

## Phase 4 Task 2 (COMPLETE — 34 new tests, 626 total):
- [x] AuditLogger — immutable append-only audit log with action/resource constants, query methods (14 tests)
- [x] Session CRUD API — list/get/create/update/archive/unarchive/delete with pagination, filtering, soft-delete (20 tests)
- [x] Audit integration — all session mutations logged with user_id, resource_id, IP, metadata
- [x] Authorization — session ownership enforced, other users get 404

## Phase 4 Task 1 (COMPLETE — 176 new tests, 592 total):
- [x] Encryption Engine — AES-256-GCM with PBKDF2 key derivation (31 tests)
- [x] Sanitizer — 3-stage PII + injection + policy pipeline (42 tests)
- [x] Policy Engine — DEFAULT/STRICT/ENTERPRISE tiers with custom overrides (30 tests)
- [x] Vault — Encrypted blob storage with compression and session binding (28 tests)
- [x] Versioning — Git-like DAG with branching, diffing, history traversal (45 tests)

## Phase 3 (COMPLETE — 67 new tests, 416 total):
- [x] Phase 3 Task 1: Output Adapters — COMPLETE (40 tests)
- [x] Phase 3 Task 2: Manual Import Connector — COMPLETE (19 tests)
- [x] Phase 3 Task 3: API Proxy Capture — COMPLETE (12 tests)
- [x] Phase 3 Task 4: API Endpoints — COMPLETE (wired contexts/shifts/import routes)

## Phase 2 (COMPLETE — 349 tests):
- [x] Phase 2 Task 1: Conversation Parser — COMPLETE (119 tests)
- [x] Phase 2 Task 2: Entity Extraction — COMPLETE (71 tests)
- [x] Phase 2 Task 3: Hierarchical Summarization — COMPLETE (45 tests)
- [x] Phase 2 Task 4: Context Compression — COMPLETE (35 tests)
- [x] Phase 2 Task 5: UCS Generator Pipeline — COMPLETE (24 tests)
- [x] Phase 2 Task 6: Local Processing Engine — COMPLETE (10 tests)

## Phase 2 Task 1 Summary: Conversation Parser

### Architecture
- `ccr.py` — Canonical Conversation Representation (CCR): the brain's internal language
  - ContentBlock (text/code/image/file/tool_call/tool_result/thinking/error)
  - Message (role, content blocks, metadata, timestamps)
  - Conversation (messages, system instruction, source tracking, aggregates)
  - All models immutable (frozen=True)

### Parsers (4 formats, auto-detected)
- `parsers/openai.py` — OpenAI API messages + ChatGPT export JSON (tree linearization)
  - Multi-modal (text + image_url), tool_calls (modern + legacy), code blocks
  - ChatGPT export: tree structure → linearized main branch
  - API response wrapper (choices + usage)
- `parsers/claude.py` — Anthropic Claude API format
  - Typed content blocks (text, tool_use, tool_result, image, thinking)
  - System prompt as top-level field (string or block list)
  - Base64 and URL image sources, error tool results
- `parsers/gemini.py` — Google Gemini API format
  - "model" role → ASSISTANT, "parts" → content blocks
  - functionCall/functionResponse, executableCode, codeExecutionResult
  - systemInstruction (camelCase and snake_case), inline_data
  - generateContent response (candidates + usageMetadata)
- `parsers/generic.py` — Markdown/text fallback
  - Turn detection: User:/Human:/Q: → user, Assistant:/AI:/Claude:/GPT: → assistant
  - Flexible separators (: - | >), section separators (--- ===) ignored
  - No markers → single user message, empty → empty conversation

### Parser Registry
- `ParserRegistry` with priority-based auto-detection
- JSON string → auto-parse to dict/list → detect format → route to parser
- Priorities: OpenAI(100) > Claude(90) > Gemini(80) > Generic(0)

### Test Results
- 349 tests total, 0 failures, ~26s
- Cross-format consistency tests verify identical CCR output

## Phase 3 Architecture Summary: Connectors & Adapters

### Output Adapters (UCS -> LLM format)
- `adapters/base.py` — BaseAdapter ABC + AdapterRegistry + context summary builder
  - `_build_context_summary()`: generates structured markdown from UCS (entities, decisions, tasks, artifacts, preferences)
  - `AdapterRegistry`: routes UCS to correct adapter by format name
  - `AdaptedOutput`: frozen result with messages, system_prompt, metadata, token_estimate
- `adapters/openai_adapter.py` — UCS -> OpenAI Chat Completions format
  - System message with full context + assistant continuation + artifact messages
  - Model suggestion based on token count (gpt-4o / gpt-4o-mini)
- `adapters/claude_adapter.py` — UCS -> Anthropic Claude Messages format
  - System prompt as TOP-LEVEL field (not a message), typed content blocks
  - Strict user/assistant alternation, priming message for context establishment
- `adapters/gemini_adapter.py` — UCS -> Google Gemini generateContent format
  - Uses "model" role (not "assistant"), "parts" format (not "content")
  - systemInstruction in metadata for API consumption

### Ingestion Connectors
- `connectors/manual_import.py` — Text/JSON/file import with auto-detection
  - `import_text()`: paste raw text (JSON auto-parsed, markdown, plain text)
  - `import_json()`: parsed JSON data (any supported format)
  - `import_file()`: upload .json/.txt/.md files, handles encoding (utf-8/utf-8-sig/latin-1)
- `connectors/api_proxy.py` — Transparent API request/response capture
  - `capture_openai()`: captures Chat Completions exchanges
  - `capture_claude()`: captures Messages API exchanges
  - `capture_gemini()`: captures generateContent exchanges
  - `capture_auto()`: auto-detects provider from request/response structure
  - `CapturedExchange`: frozen result with exchange_id, provider, tokens, UCS

### API Endpoints (wired to engine)
- `POST /api/v1/contexts/generate` — Full UCS generation from raw data
- `POST /api/v1/contexts/generate/local` — Local-only (privacy mode)
- `POST /api/v1/contexts/entities` — Entity extraction only
- `POST /api/v1/contexts/summarize` — Summarization only
- `POST /api/v1/shifts/execute` — Full context shift (raw -> target LLM format)
- `POST /api/v1/shifts/preview` — Preview shift stats before executing
- `GET  /api/v1/shifts/formats` — List available input/output formats
- `POST /api/v1/import/text` — Import from pasted text
- `POST /api/v1/import/json` — Import from JSON data
- `POST /api/v1/import/file` — Import from uploaded file

## Phase 2 Task 2 Summary: Entity Extraction Pipeline
- `entities.py` — 600+ lines, 5-stage pipeline
  - `SpaCyExtractor`: NER for people, orgs, locations (lazy-loads model, graceful fallback)
  - `TechnicalExtractor`: regex for file paths, URLs, APIs, imports, env vars, code symbols
  - Technology Gazetteer: 200+ terms (languages, frameworks, tools, databases, cloud, AI models)
  - `RelationshipExtractor`: co-occurrence + verb pattern matching (uses, depends_on, part_of, etc.)
  - `EntityDeduplicator`: exact match + substring merging with alias tracking
  - `KnowledgeGraphBuilder`: importance scoring (0.3*recency + 0.3*frequency + 0.2*relationships + 0.2*type_weight)
  - `EntityPipeline`: orchestrator combining all stages -> EntityResult
- 71 tests covering all extractors, deduplication, relationships, graph building, and integration

## Phase 2 Task 3 Summary: Hierarchical Summarization
- `summarizer.py` — 400+ lines, LOCAL-FIRST extractive summarization (no LLM API calls)
  - `TFIDFScorer`: Term Frequency - Inverse Document Frequency for sentence importance
  - `TopicDetector`: keyword overlap clustering with Jaccard similarity + sliding window
  - `MessageSummarizer`: Level 1 — one sentence per message (TF-IDF top sentence)
  - `TopicSummarizer`: Level 2 — one paragraph per topic cluster (top 5 sentences)
  - `GlobalSummarizer`: Level 3 — ~200 word budget covering entire conversation
  - `SummarizationPipeline`: orchestrator -> SummaryResult with all 3 tiers
- 45 tests covering TF-IDF, topic detection, all 3 summary levels, and pipeline integration

## Phase 2 Task 4 Summary: Context Compression
- `compressor.py` — 300+ lines, adaptive priority-based compression
  - Priority queue: lower priority items dropped first
  - Base weights: MESSAGE_SUMMARY(1) < ENTITY(2) < ARTIFACT(2.5) < TOPIC_SUMMARY(3) < GLOBAL/DECISION/TASK(10)
  - Entities weighted by importance score (low-importance dropped before high)
  - Artifacts truncated (not dropped) — keeps imports and signatures
  - Decisions, tasks, and global summary are NEVER dropped (protected categories)
  - Token estimation: ~4 chars/token heuristic
  - `CompressionPipeline`: configurable target_tokens budget -> CompressionResult
- 35 tests covering priority ordering, protected content, truncation, and compression strategy

## Phase 2 Task 5 Summary: UCS Generator Pipeline
- `ucs_generator.py` — 250+ lines, the brain's main orchestrator
  - Full pipeline: Raw Input -> Parser -> Entity Extraction -> Summarization -> Compression -> UCS Assembly -> Validation
  - `UCSGeneratorPipeline.generate()`: raw data in, validated UCS out
  - `UCSGeneratorPipeline.generate_from_conversation()`: pre-parsed CCR in
  - `GenerationResult`: contains UCS + Conversation + GenerationStats
  - `GenerationStats`: source info, counts, timing, validation warnings
  - Automatic source format -> SourceLLM mapping (OpenAI/Claude/Gemini/Generic)
- 24 tests covering full pipeline, different input formats, stats, and validation

## Phase 2 Task 6 Summary: Local Processing Engine
- `local_engine.py` — 150+ lines, privacy-first offline mode
  - Wraps UCSGeneratorPipeline with local-only guarantees
  - No network calls, no API keys, no telemetry
  - All processing on-device (spaCy + TF-IDF + regex)
  - Output stamped with ProcessingMode.LOCAL
  - `LocalProcessingConfig`: configurable tokens, spaCy, thresholds
  - `LocalEngine.process()`: raw data -> LOCAL UCS
  - `LocalEngine.process_conversation()`: CCR -> LOCAL UCS
- 10 tests covering local stamp, config, privacy guarantees

## Phase 1 Architecture Summary:

### Database (9 tables)
- `users` — accounts, subscription tiers (free/pro/team/enterprise), Stripe IDs, soft-delete
- `sessions` — captured conversations, LLM source tracking, team scoping
- `contexts` — versioned encrypted blobs (AES-256-GCM), branching support
- `api_keys` — hashed keys with prefix display, scopes, expiration
- `teams` — workspaces with slug, member limits, settings
- `team_members` — RBAC roles (owner/admin/member/viewer)
- `webhooks` — event subscriptions with failure tracking, auto-disable
- `audit_logs` — immutable, append-only security/compliance log
- `shift_records` — billable action tracking, LLM pair analytics

### Auth Flow
- Registration → bcrypt hash (cost 12) → JWT access + refresh tokens
- Login → constant-time comparison → same error for email-not-found and wrong-password
- Refresh → validates refresh token type → returns new pair with current tier
- Protected routes → `get_current_user` dependency → queries real database

### Infrastructure
- Multi-stage Dockerfiles (builder → runtime → development)
- Non-root container users (appuser/nextjs)
- .dockerignore prevents secret/cache leakage
- Health checks on all services
- Connection pooling (20 + 10 overflow)
- Structured JSON logging (structlog)

## Phase 4 Task 1 Summary: Security & Storage

### Encryption (`security/encryption.py`)
- `EncryptionEngine`: AES-256-GCM with unique nonce + salt per encryption
- `derive_key()`: PBKDF2-HMAC-SHA256, 600K iterations (OWASP 2024)
- `EncryptedPayload`: frozen dataclass with to_bytes/from_bytes serialization
- Format: [version:1][salt:32][nonce:12][ciphertext:N]
- AAD (Associated Authenticated Data) for session binding
- String convenience methods, secure key generation

### Sanitizer (`security/sanitizer.py`)
- 3-stage pipeline: PII redaction -> injection detection -> policy check
- PII patterns: email, phone, credit card, SSN, IPv4, API keys (7 providers), bearer tokens
- Injection patterns: system overrides, jailbreaks (DAN), encoding attacks, delimiter injection (ChatML, LLaMA)
- `SecurityViolation` exception with flags when blocking on critical
- `SanitizeResult`: frozen result with cleaned text, flags, counts, safety status
- Batch `sanitize_messages()` for full conversation processing

### Policy Engine (`security/policy_engine.py`)
- `SecurityTier`: DEFAULT (free), STRICT (pro/team), ENTERPRISE
- `SecurityPolicy`: 15-field frozen dataclass (PII, injections, encryption, MFA, DLP, residency, etc.)
- Subscription tier mapping (free/pro/team/enterprise -> security tier)
- `create_custom_policy()`: base tier + arbitrary overrides
- `validate_policy()`: consistency warnings (encryption off, DLP without PII, etc.)

### Vault (`storage/vault.py`)
- `Vault`: encrypted UCS storage with zlib compression
- Store: UCS -> JSON -> compress -> encrypt -> VaultEntry
- Retrieve: VaultEntry -> decrypt -> decompress -> JSON -> UCS
- Session-based AAD binding (tampering session_id breaks decryption)
- `VaultEntry`: frozen metadata (blob size, compression ratio, timestamps, searchable metadata)

### Versioning (`storage/versioning.py`)
- `VersionGraph`: git-like DAG for UCS version history
- `VersionNode`: frozen node with parent_id, branch_name, version_number
- `Branch`: named branch with head/base tracking, status lifecycle
- `compute_diff()`: entity/decision/task diffs, message count delta, summary changes
- Branch management: create, commit, archive (main protected)
- History traversal: walk parent chain with max_depth

## Notes:
- Auth is wired to real PostgreSQL via async SQLAlchemy (not in-memory dict)
- Database models use `extra_data` Python attribute mapped to `metadata` column (SQLAlchemy reserves `metadata`)
- JSONB columns use `with_variant(JSON(), "sqlite")` for test compatibility
- CCR models are separate from UCS models: CCR = raw input, UCS = processed output
- Code block extraction uses markdown fence detection (```language ... ```)
- Parser priority system ensures most-specific parser wins over generic fallback
- JWT InsecureKeyLengthWarning in tests is expected (dev key is intentionally short)
- Production secret validation enforced in config.py (staging/prod fail if using defaults)
