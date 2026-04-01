"""Named entity recognition and knowledge graph construction.

The Entity Extraction Pipeline is the brain's perception layer. It reads a
normalized Conversation (CCR) and identifies every meaningful entity — people,
technologies, code symbols, file paths, URLs, API endpoints — then maps the
relationships between them into a knowledge graph.

Pipeline stages:
    1. SpaCy NER      — people, organizations, locations via trained model
    2. Technical NER   — file paths, URLs, APIs, imports, env vars via regex
                         + 200-term technology gazetteer
    3. Relationship    — co-occurrence + verb pattern extraction
    4. Deduplication   — exact match + substring merging
    5. Knowledge Graph — importance scoring + UCS Entity/KnowledgeGraph building

Design principles:
    - Immutable outputs (frozen Pydantic models in UCS)
    - Mutable intermediates (dataclasses for pipeline efficiency)
    - Graceful degradation (works without spaCy model installed)
    - Deterministic (same input always produces same output)
    - No side effects (pure functions, no I/O)

Usage:
    from app.core.engine.entities import EntityPipeline, EntityResult

    pipeline = EntityPipeline()
    result = pipeline.extract(conversation)
    # result.entities   -> tuple[Entity, ...]
    # result.graph      -> KnowledgeGraph
    # result.scores     -> dict[str, float]  (entity_id -> importance)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from app.core.engine.ccr import ContentType, Conversation, Message, MessageRole
from app.core.models.ucs import (
    Entity,
    EntityRelationship,
    EntityType,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    RelationshipType,
)


# -- Mutable intermediates (pipeline-internal only) --------------------------


@dataclass
class RawEntity:
    """Mutable entity representation used during extraction.

    This is NOT part of the public API. It accumulates mentions across
    messages before being converted to an immutable UCS Entity.
    """

    name: str
    type: EntityType
    aliases: list[str] = field(default_factory=list)
    mention_indices: list[int] = field(default_factory=list)
    mention_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_name(self) -> str:
        """Return the canonical (most common) form of this entity's name."""
        return self.name

    def add_mention(self, message_index: int) -> None:
        """Record a new mention of this entity."""
        if message_index not in self.mention_indices:
            self.mention_indices.append(message_index)
        self.mention_count += 1


@dataclass
class RawRelationship:
    """Mutable relationship between two raw entities."""

    source_name: str
    target_name: str
    type: RelationshipType
    confidence: float = 1.0
    evidence_count: int = 1


# -- Technology Gazetteer ---------------------------------------------------

# 200+ terms organized by category for precise technology detection.
# Each term maps to its canonical display name.

_LANGUAGES: dict[str, str] = {
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "java": "Java",
    "kotlin": "Kotlin",
    "swift": "Swift",
    "rust": "Rust",
    "golang": "Go",
    "ruby": "Ruby",
    "php": "PHP",
    "scala": "Scala",
    "elixir": "Elixir",
    "haskell": "Haskell",
    "clojure": "Clojure",
    "lua": "Lua",
    "perl": "Perl",
    "r lang": "R",
    "dart": "Dart",
    "zig": "Zig",
    "nim": "Nim",
    "julia": "Julia",
    "erlang": "Erlang",
    "fortran": "Fortran",
    "cobol": "COBOL",
    "assembly": "Assembly",
    "solidity": "Solidity",
    "sql": "SQL",
    "graphql": "GraphQL",
    "html": "HTML",
    "css": "CSS",
    "sass": "Sass",
    "less": "LESS",
}

_FRAMEWORKS: dict[str, str] = {
    "react": "React",
    "next.js": "Next.js",
    "nextjs": "Next.js",
    "vue": "Vue.js",
    "nuxt": "Nuxt.js",
    "angular": "Angular",
    "svelte": "Svelte",
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "express": "Express.js",
    "nestjs": "NestJS",
    "spring boot": "Spring Boot",
    "spring": "Spring",
    "rails": "Ruby on Rails",
    "laravel": "Laravel",
    "phoenix": "Phoenix",
    "gin": "Gin",
    "fiber": "Fiber",
    "actix": "Actix",
    "axum": "Axum",
    "rocket": "Rocket",
    "tailwind": "Tailwind CSS",
    "tailwindcss": "Tailwind CSS",
    "bootstrap": "Bootstrap",
    "material ui": "Material UI",
    "chakra ui": "Chakra UI",
    "shadcn": "shadcn/ui",
    "storybook": "Storybook",
    "remix": "Remix",
    "astro": "Astro",
    "gatsby": "Gatsby",
    "vite": "Vite",
    "webpack": "Webpack",
    "esbuild": "esbuild",
    "turbopack": "Turbopack",
    "pydantic": "Pydantic",
    "sqlalchemy": "SQLAlchemy",
    "alembic": "Alembic",
    "celery": "Celery",
    "pytest": "pytest",
    "jest": "Jest",
    "vitest": "Vitest",
    "playwright": "Playwright",
    "cypress": "Cypress",
    "selenium": "Selenium",
}

_TOOLS: dict[str, str] = {
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "terraform": "Terraform",
    "ansible": "Ansible",
    "jenkins": "Jenkins",
    "github actions": "GitHub Actions",
    "gitlab ci": "GitLab CI",
    "circleci": "CircleCI",
    "vercel": "Vercel",
    "netlify": "Netlify",
    "heroku": "Heroku",
    "nginx": "NGINX",
    "apache": "Apache",
    "caddy": "Caddy",
    "git": "Git",
    "npm": "npm",
    "yarn": "Yarn",
    "pnpm": "pnpm",
    "pip": "pip",
    "poetry": "Poetry",
    "cargo": "Cargo",
    "gradle": "Gradle",
    "maven": "Maven",
    "make": "Make",
    "cmake": "CMake",
    "bazel": "Bazel",
    "prometheus": "Prometheus",
    "grafana": "Grafana",
    "datadog": "Datadog",
    "sentry": "Sentry",
    "elasticsearch": "Elasticsearch",
    "kibana": "Kibana",
    "logstash": "Logstash",
    "meilisearch": "Meilisearch",
    "stripe": "Stripe",
    "twilio": "Twilio",
    "sendgrid": "SendGrid",
    "auth0": "Auth0",
}

_DATABASES: dict[str, str] = {
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "mysql": "MySQL",
    "mariadb": "MariaDB",
    "sqlite": "SQLite",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "memcached": "Memcached",
    "cassandra": "Cassandra",
    "dynamodb": "DynamoDB",
    "firestore": "Firestore",
    "supabase": "Supabase",
    "cockroachdb": "CockroachDB",
    "neo4j": "Neo4j",
    "influxdb": "InfluxDB",
    "clickhouse": "ClickHouse",
    "pinecone": "Pinecone",
    "weaviate": "Weaviate",
    "qdrant": "Qdrant",
    "chromadb": "ChromaDB",
    "pgvector": "pgvector",
    "milvus": "Milvus",
}

_CLOUD: dict[str, str] = {
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "GCP",
    "google cloud": "GCP",
    "azure": "Azure",
    "digitalocean": "DigitalOcean",
    "cloudflare": "Cloudflare",
    "fly.io": "Fly.io",
    "railway": "Railway",
    "render": "Render",
    "lambda": "AWS Lambda",
    "s3": "Amazon S3",
    "ec2": "Amazon EC2",
    "ecs": "Amazon ECS",
    "eks": "Amazon EKS",
    "rds": "Amazon RDS",
    "sqs": "Amazon SQS",
    "sns": "Amazon SNS",
    "cloudfront": "CloudFront",
}

_AI_MODELS: dict[str, str] = {
    "gpt-4": "GPT-4",
    "gpt-4o": "GPT-4o",
    "gpt-3.5": "GPT-3.5",
    "chatgpt": "ChatGPT",
    "claude": "Claude",
    "claude-3": "Claude 3",
    "gemini": "Gemini",
    "llama": "LLaMA",
    "mistral": "Mistral",
    "mixtral": "Mixtral",
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "cohere": "Cohere",
    "hugging face": "Hugging Face",
    "huggingface": "Hugging Face",
    "langchain": "LangChain",
    "llamaindex": "LlamaIndex",
    "transformers": "Transformers",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "keras": "Keras",
    "spacy": "spaCy",
    "stable diffusion": "Stable Diffusion",
    "midjourney": "Midjourney",
    "dall-e": "DALL-E",
    "whisper": "Whisper",
    "ollama": "Ollama",
}

# Combined gazetteer: lowercase lookup -> (canonical_name, EntityType)
TECHNOLOGY_GAZETTEER: dict[str, tuple[str, EntityType]] = {}
for _terms, _etype in [
    (_LANGUAGES, EntityType.TECHNOLOGY),
    (_FRAMEWORKS, EntityType.TECHNOLOGY),
    (_TOOLS, EntityType.TECHNOLOGY),
    (_DATABASES, EntityType.TECHNOLOGY),
    (_CLOUD, EntityType.TECHNOLOGY),
    (_AI_MODELS, EntityType.TECHNOLOGY),
]:
    for _key, _canonical in _terms.items():
        TECHNOLOGY_GAZETTEER[_key] = (_canonical, _etype)


# -- Regex Patterns ----------------------------------------------------------

# File paths: Unix and Windows styles
_FILE_PATH_RE = re.compile(
    r"(?<!\w)"
    r"(?:"
    r"(?:[A-Za-z]:)?(?:[/\\])(?:[\w\-. ]+[/\\])*[\w\-. ]+\.\w{1,10}"  # with extension
    r"|"
    r"(?:\./|\.\\|~/|~\\)(?:[\w\-. ]+[/\\])*[\w\-. ]+"  # relative path
    r")"
    r"(?!\w)"
)

# URLs: http(s) and common protocols
_URL_RE = re.compile(
    r"https?://[^\s<>\"'`)\]},;]+[^\s<>\"'`)\]},;.]"
)

# API endpoints: REST-style paths
_API_ENDPOINT_RE = re.compile(
    r"(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+"
    r"(/[\w\-./{}:?&=]+)"
    r"|"
    r"(?<!\w)(/(?:api|v\d)/[\w\-./{}:?&=]+)"
)

# Python/JS/TS imports
_IMPORT_RE = re.compile(
    r"(?:"
    r"(?:from\s+)([\w.]+)\s+import"  # Python from X import
    r"|"
    r"import\s+([\w.]+)"  # Python/JS import X
    r"|"
    r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"  # JS require('X')
    r")"
)

# Environment variables
_ENV_VAR_RE = re.compile(
    r"(?<!\w)([A-Z][A-Z0-9_]{2,})(?!\w)"
)

# Common env var patterns to include (filter noise)
_ENV_VAR_PREFIXES = frozenset({
    "API_", "DB_", "DATABASE_", "REDIS_", "AWS_", "GCP_", "AZURE_",
    "SECRET_", "JWT_", "AUTH_", "SMTP_", "MAIL_", "STRIPE_",
    "SENTRY_", "LOG_", "DEBUG", "NODE_ENV", "PYTHON", "PORT",
    "HOST", "OPENAI_", "ANTHROPIC_", "GOOGLE_", "NEXT_PUBLIC_",
})

# Code symbols: function/class/variable definitions
_CODE_SYMBOL_RE = re.compile(
    r"(?:"
    r"(?:def|function|func|fn)\s+(\w+)"  # function definitions
    r"|"
    r"(?:class|struct|enum|interface|type)\s+(\w+)"  # type definitions
    r"|"
    r"(?:const|let|var|val)\s+(\w+)\s*="  # variable declarations
    r")"
)


# -- Extractors --------------------------------------------------------------


class SpaCyExtractor:
    """Extract named entities using spaCy's pre-trained NER model.

    Lazily loads the spaCy model on first use. Falls back gracefully
    if spaCy or the model is not installed — the pipeline continues
    with technical extraction only.

    Recognized entity types:
        - PERSON -> EntityType.PERSON
        - ORG    -> EntityType.ORGANIZATION
        - GPE    -> EntityType.LOCATION (geo-political entities)
        - LOC    -> EntityType.LOCATION
    """

    _SPACY_TO_ENTITY_TYPE: dict[str, EntityType] = {
        "PERSON": EntityType.PERSON,
        "ORG": EntityType.ORGANIZATION,
        "GPE": EntityType.LOCATION,
        "LOC": EntityType.LOCATION,
    }

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self._model_name = model_name
        self._nlp: Any = None
        self._available: bool | None = None

    def _load_model(self) -> None:
        """Lazy-load spaCy model. Sets _available flag."""
        if self._available is not None:
            return
        try:
            import spacy  # noqa: F811
            self._nlp = spacy.load(self._model_name)
            self._available = True
        except (ImportError, OSError):
            self._available = False

    @property
    def available(self) -> bool:
        """Check if spaCy model is available."""
        self._load_model()
        return self._available is True

    def extract(
        self, messages: tuple[Message, ...],
    ) -> list[RawEntity]:
        """Extract named entities from conversation messages.

        Args:
            messages: Tuple of CCR Messages to process.

        Returns:
            List of RawEntity objects found by spaCy NER.
        """
        self._load_model()
        if not self._available:
            return []

        entities_by_name: dict[str, RawEntity] = {}

        for idx, msg in enumerate(messages):
            text = msg.full_text
            if not text.strip():
                continue

            # Limit text length to avoid memory issues on huge messages
            doc = self._nlp(text[:50000])

            for ent in doc.ents:
                if ent.label_ not in self._SPACY_TO_ENTITY_TYPE:
                    continue

                name = ent.text.strip()
                if len(name) < 2:
                    continue

                entity_type = self._SPACY_TO_ENTITY_TYPE[ent.label_]
                key = name.lower()

                if key in entities_by_name:
                    entities_by_name[key].add_mention(idx)
                else:
                    entities_by_name[key] = RawEntity(
                        name=name,
                        type=entity_type,
                        mention_indices=[idx],
                        mention_count=1,
                    )

        return list(entities_by_name.values())


class TechnicalExtractor:
    """Extract technical entities using regex patterns and a technology gazetteer.

    This catches what NER models miss: file paths, URLs, API endpoints,
    code symbols, import statements, environment variables, and technology
    names from a 200+ term gazetteer.

    The gazetteer uses whole-word matching to avoid false positives
    (e.g., "go" in "going" won't match).
    """

    def extract(
        self, messages: tuple[Message, ...],
    ) -> list[RawEntity]:
        """Extract technical entities from conversation messages.

        Args:
            messages: Tuple of CCR Messages to process.

        Returns:
            List of RawEntity objects found by pattern matching.
        """
        entities_by_key: dict[str, RawEntity] = {}

        for idx, msg in enumerate(messages):
            text = msg.full_text
            if not text.strip():
                continue

            self._extract_file_paths(text, idx, entities_by_key)
            self._extract_urls(text, idx, entities_by_key)
            self._extract_api_endpoints(text, idx, entities_by_key)
            self._extract_imports(text, idx, entities_by_key)
            self._extract_env_vars(text, idx, entities_by_key)
            self._extract_code_symbols(msg, idx, entities_by_key)
            self._extract_technologies(text, idx, entities_by_key)

        return list(entities_by_key.values())

    def _add_entity(
        self,
        entities: dict[str, RawEntity],
        name: str,
        entity_type: EntityType,
        message_index: int,
        key: str | None = None,
    ) -> None:
        """Add or update an entity in the accumulator."""
        if not name or len(name) < 2:
            return
        lookup_key = key or name.lower()
        if lookup_key in entities:
            entities[lookup_key].add_mention(message_index)
        else:
            entities[lookup_key] = RawEntity(
                name=name,
                type=entity_type,
                mention_indices=[message_index],
                mention_count=1,
            )

    def _extract_file_paths(
        self,
        text: str,
        idx: int,
        entities: dict[str, RawEntity],
    ) -> None:
        """Extract file paths from text."""
        for match in _FILE_PATH_RE.finditer(text):
            path = match.group(0).strip()
            self._add_entity(entities, path, EntityType.FILE_PATH, idx)

    def _extract_urls(
        self,
        text: str,
        idx: int,
        entities: dict[str, RawEntity],
    ) -> None:
        """Extract URLs from text."""
        for match in _URL_RE.finditer(text):
            url = match.group(0).strip()
            self._add_entity(entities, url, EntityType.URL, idx)

    def _extract_api_endpoints(
        self,
        text: str,
        idx: int,
        entities: dict[str, RawEntity],
    ) -> None:
        """Extract REST API endpoints from text."""
        for match in _API_ENDPOINT_RE.finditer(text):
            endpoint = (match.group(1) or match.group(2) or "").strip()
            if endpoint:
                self._add_entity(entities, endpoint, EntityType.API, idx)

    def _extract_imports(
        self,
        text: str,
        idx: int,
        entities: dict[str, RawEntity],
    ) -> None:
        """Extract import/require statements from text."""
        for match in _IMPORT_RE.finditer(text):
            module = (match.group(1) or match.group(2) or match.group(3) or "").strip()
            if module and len(module) > 1:
                self._add_entity(
                    entities, module, EntityType.CODE, idx, key=f"import:{module.lower()}",
                )

    def _extract_env_vars(
        self,
        text: str,
        idx: int,
        entities: dict[str, RawEntity],
    ) -> None:
        """Extract environment variable names from text."""
        for match in _ENV_VAR_RE.finditer(text):
            var_name = match.group(1)
            # Filter: must start with a known prefix or be a common env var
            if any(var_name.startswith(prefix) for prefix in _ENV_VAR_PREFIXES):
                self._add_entity(entities, var_name, EntityType.CODE, idx)

    def _extract_code_symbols(
        self,
        msg: Message,
        idx: int,
        entities: dict[str, RawEntity],
    ) -> None:
        """Extract code symbol definitions from code blocks."""
        for block in msg.content:
            if block.type != ContentType.CODE:
                continue
            code_text = block.text
            if not code_text.strip():
                continue
            for match in _CODE_SYMBOL_RE.finditer(code_text):
                symbol = (match.group(1) or match.group(2) or match.group(3) or "").strip()
                if symbol and len(symbol) > 1:
                    self._add_entity(
                        entities, symbol, EntityType.CODE, idx,
                        key=f"symbol:{symbol}",
                    )

    def _extract_technologies(
        self,
        text: str,
        idx: int,
        entities: dict[str, RawEntity],
    ) -> None:
        """Match technology names from the gazetteer using whole-word search."""
        text_lower = text.lower()
        for term, (canonical, etype) in TECHNOLOGY_GAZETTEER.items():
            # Whole-word boundary check
            pattern = r"(?<![a-zA-Z0-9_\-])" + re.escape(term) + r"(?![a-zA-Z0-9_\-])"
            if re.search(pattern, text_lower):
                key = f"tech:{canonical.lower()}"
                if key in entities:
                    entities[key].add_mention(idx)
                else:
                    entities[key] = RawEntity(
                        name=canonical,
                        type=etype,
                        mention_indices=[idx],
                        mention_count=1,
                    )


class RelationshipExtractor:
    """Extract relationships between entities.

    Two strategies:
        1. Co-occurrence: entities mentioned in the same message are related.
        2. Verb patterns: "X uses Y", "X depends on Y", etc.

    Relationship types:
        - USES: "uses", "utilizes", "employs", "leverages"
        - DEPENDS_ON: "depends on", "requires", "needs"
        - PART_OF: "part of", "inside", "within", "belongs to"
        - IMPLEMENTS: "implements", "realizes", "provides"
        - CREATED_BY: "created by", "built by", "made by", "authored by"
    """

    _VERB_PATTERNS: list[tuple[str, RelationshipType]] = [
        (r"\buses\b", RelationshipType.USES),
        (r"\butilizes\b", RelationshipType.USES),
        (r"\bemploys\b", RelationshipType.USES),
        (r"\bleverages\b", RelationshipType.USES),
        (r"\bdepends on\b", RelationshipType.DEPENDS_ON),
        (r"\brequires\b", RelationshipType.DEPENDS_ON),
        (r"\bneeds\b", RelationshipType.DEPENDS_ON),
        (r"\bpart of\b", RelationshipType.PART_OF),
        (r"\binside\b", RelationshipType.PART_OF),
        (r"\bwithin\b", RelationshipType.PART_OF),
        (r"\bbelongs to\b", RelationshipType.PART_OF),
        (r"\bimplements\b", RelationshipType.IMPLEMENTS),
        (r"\brealizes\b", RelationshipType.IMPLEMENTS),
        (r"\bprovides\b", RelationshipType.IMPLEMENTS),
        (r"\bcreated by\b", RelationshipType.CREATED_BY),
        (r"\bbuilt by\b", RelationshipType.CREATED_BY),
        (r"\bmade by\b", RelationshipType.CREATED_BY),
        (r"\bauthored by\b", RelationshipType.CREATED_BY),
    ]

    def extract(
        self,
        entities: list[RawEntity],
        messages: tuple[Message, ...],
    ) -> list[RawRelationship]:
        """Extract relationships between entities.

        Args:
            entities: All extracted entities.
            messages: Original conversation messages.

        Returns:
            List of RawRelationship objects.
        """
        relationships: list[RawRelationship] = []

        # Build message_index -> entities mapping
        msg_entities: dict[int, list[RawEntity]] = {}
        for ent in entities:
            for idx in ent.mention_indices:
                msg_entities.setdefault(idx, []).append(ent)

        # Strategy 1: Co-occurrence (entities in same message)
        seen_pairs: set[tuple[str, str]] = set()
        for idx, ents in msg_entities.items():
            for i, e1 in enumerate(ents):
                for e2 in ents[i + 1:]:
                    pair = (
                        min(e1.name.lower(), e2.name.lower()),
                        max(e1.name.lower(), e2.name.lower()),
                    )
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        relationships.append(RawRelationship(
                            source_name=e1.name,
                            target_name=e2.name,
                            type=RelationshipType.RELATED_TO,
                            confidence=0.5,
                        ))
                    else:
                        # Boost confidence for repeated co-occurrence
                        for rel in relationships:
                            p = (
                                min(rel.source_name.lower(), rel.target_name.lower()),
                                max(rel.source_name.lower(), rel.target_name.lower()),
                            )
                            if p == pair:
                                rel.evidence_count += 1
                                rel.confidence = min(1.0, rel.confidence + 0.1)
                                break

        # Strategy 2: Verb patterns
        entity_names = {e.name.lower(): e for e in entities}
        for idx, msg in enumerate(messages):
            text_lower = msg.full_text.lower()
            for pattern, rel_type in self._VERB_PATTERNS:
                for match in re.finditer(pattern, text_lower):
                    pos = match.start()
                    # Look for entity names near the verb
                    context_before = text_lower[max(0, pos - 100):pos]
                    context_after = text_lower[pos:min(len(text_lower), pos + 100)]

                    source = self._find_nearest_entity(context_before, entity_names, reverse=True)
                    target = self._find_nearest_entity(context_after, entity_names, reverse=False)

                    if source and target and source != target:
                        relationships.append(RawRelationship(
                            source_name=entity_names[source].name,
                            target_name=entity_names[target].name,
                            type=rel_type,
                            confidence=0.7,
                        ))

        return relationships

    @staticmethod
    def _find_nearest_entity(
        text: str,
        entity_names: dict[str, RawEntity],
        reverse: bool = False,
    ) -> str | None:
        """Find the nearest entity name in a text fragment.

        Args:
            text: Text fragment to search.
            entity_names: Lowercase entity name -> entity mapping.
            reverse: If True, find the entity closest to the end.

        Returns:
            Lowercase entity name if found, None otherwise.
        """
        best_name: str | None = None
        best_pos = -1 if reverse else len(text) + 1

        for name in entity_names:
            pos = text.rfind(name) if reverse else text.find(name)
            if pos == -1:
                continue
            if reverse and pos > best_pos:
                best_pos = pos
                best_name = name
            elif not reverse and pos < best_pos:
                best_pos = pos
                best_name = name

        return best_name


class EntityDeduplicator:
    """Merge duplicate entities using exact match and substring rules.

    Deduplication strategies:
        1. Exact match: same canonical name (case-insensitive)
        2. Substring: "React" absorbs "React.js" as an alias
        3. Alias tracking: merged entities keep all name variants

    The winner of a merge is the entity with more mentions.
    """

    def deduplicate(self, entities: list[RawEntity]) -> list[RawEntity]:
        """Merge duplicate entities.

        Args:
            entities: Raw entities potentially containing duplicates.

        Returns:
            Deduplicated entity list with aliases populated.
        """
        if not entities:
            return []

        # Sort by mention count descending (winners first)
        sorted_entities = sorted(entities, key=lambda e: e.mention_count, reverse=True)

        merged: list[RawEntity] = []
        merged_names: dict[str, int] = {}  # lowercase name -> index in merged

        for entity in sorted_entities:
            key = entity.name.lower()

            # Check exact match
            if key in merged_names:
                target = merged[merged_names[key]]
                self._merge_into(target, entity)
                continue

            # Check substring match (e.g., "js" in "javascript")
            found_match = False
            for existing_key, idx in list(merged_names.items()):
                existing = merged[idx]
                if self._is_substring_match(entity.name, existing.name):
                    self._merge_into(existing, entity)
                    found_match = True
                    break
                if self._is_substring_match(existing.name, entity.name):
                    # The new entity is the broader term — it becomes primary
                    self._merge_into(entity, existing)
                    merged[idx] = entity
                    merged_names[entity.name.lower()] = idx
                    found_match = True
                    break

            if not found_match:
                merged_names[key] = len(merged)
                merged.append(entity)

        return merged

    @staticmethod
    def _is_substring_match(shorter: str, longer: str) -> bool:
        """Check if shorter is a meaningful substring of longer.

        Avoids matching single characters or very short substrings
        that would cause false merges.
        """
        s, l_ = shorter.lower(), longer.lower()
        if s == l_:
            return False
        if len(s) < 3:
            return False
        return s in l_

    @staticmethod
    def _merge_into(target: RawEntity, source: RawEntity) -> None:
        """Merge source entity into target, preserving all data."""
        if source.name.lower() != target.name.lower():
            if source.name not in target.aliases:
                target.aliases.append(source.name)
        for idx in source.mention_indices:
            if idx not in target.mention_indices:
                target.mention_indices.append(idx)
        target.mention_count += source.mention_count
        target.metadata.update(source.metadata)


class KnowledgeGraphBuilder:
    """Build UCS-compatible entities and knowledge graph from raw data.

    Importance scoring formula:
        score = 0.3 * recency + 0.3 * frequency + 0.2 * relationships + 0.2 * type_weight

    Where:
        - recency:       How recently the entity was mentioned (0-1)
        - frequency:     Mention count normalized by max mentions (0-1)
        - relationships: Number of relationships normalized (0-1)
        - type_weight:   Base importance by entity type (PERSON=0.8, CODE=0.6, etc.)
    """

    _TYPE_WEIGHTS: dict[EntityType, float] = {
        EntityType.PERSON: 0.8,
        EntityType.ORGANIZATION: 0.7,
        EntityType.TECHNOLOGY: 0.7,
        EntityType.CODE: 0.6,
        EntityType.API: 0.6,
        EntityType.CONCEPT: 0.5,
        EntityType.FILE_PATH: 0.4,
        EntityType.URL: 0.3,
        EntityType.LOCATION: 0.5,
        EntityType.OTHER: 0.3,
    }

    def build(
        self,
        raw_entities: list[RawEntity],
        raw_relationships: list[RawRelationship],
        total_messages: int,
    ) -> tuple[tuple[Entity, ...], KnowledgeGraph, dict[str, float]]:
        """Build UCS entities and knowledge graph.

        Args:
            raw_entities: Deduplicated raw entities.
            raw_relationships: Extracted relationships.
            total_messages: Total message count for recency calculation.

        Returns:
            Tuple of (entities, knowledge_graph, importance_scores).
        """
        if not raw_entities:
            return (), KnowledgeGraph(), {}

        # Assign UUIDs to raw entities
        entity_ids: dict[str, UUID] = {}
        for ent in raw_entities:
            entity_ids[ent.name.lower()] = uuid4()

        # Count relationships per entity
        rel_counts: dict[str, int] = {}
        for rel in raw_relationships:
            key_s = rel.source_name.lower()
            key_t = rel.target_name.lower()
            rel_counts[key_s] = rel_counts.get(key_s, 0) + 1
            rel_counts[key_t] = rel_counts.get(key_t, 0) + 1

        max_mentions = max((e.mention_count for e in raw_entities), default=1)
        max_rels = max(rel_counts.values(), default=1)

        # Build UCS Entity objects with importance scores
        ucs_entities: list[Entity] = []
        importance_scores: dict[str, float] = {}

        for raw in raw_entities:
            eid = entity_ids[raw.name.lower()]

            # Compute importance score
            recency = self._compute_recency(raw.mention_indices, total_messages)
            frequency = raw.mention_count / max_mentions if max_mentions > 0 else 0
            rel_score = rel_counts.get(raw.name.lower(), 0) / max_rels if max_rels > 0 else 0
            type_weight = self._TYPE_WEIGHTS.get(raw.type, 0.3)

            importance = (
                0.3 * recency
                + 0.3 * frequency
                + 0.2 * rel_score
                + 0.2 * type_weight
            )
            importance = round(min(1.0, max(0.0, importance)), 4)

            # Build entity relationships
            entity_rels: list[EntityRelationship] = []
            for rel in raw_relationships:
                if rel.source_name.lower() == raw.name.lower():
                    target_id = entity_ids.get(rel.target_name.lower())
                    if target_id:
                        entity_rels.append(EntityRelationship(
                            target_id=target_id,
                            type=rel.type,
                            confidence=round(rel.confidence, 4),
                        ))

            first_mentioned = min(raw.mention_indices) if raw.mention_indices else 0

            entity = Entity(
                id=eid,
                name=raw.canonical_name,
                type=raw.type,
                aliases=tuple(raw.aliases),
                first_mentioned_at=first_mentioned,
                importance=importance,
                relationships=tuple(entity_rels),
                metadata={k: v for k, v in raw.metadata.items()
                          if isinstance(v, (str, int, float, bool))},
            )
            ucs_entities.append(entity)
            importance_scores[str(eid)] = importance

        # Build knowledge graph
        nodes = tuple(
            KnowledgeGraphNode(
                entity_id=e.id,
                label=e.name,
                group=e.type.value,
            )
            for e in ucs_entities
        )

        node_ids = frozenset(n.entity_id for n in nodes)
        edges: list[KnowledgeGraphEdge] = []
        for rel in raw_relationships:
            source_id = entity_ids.get(rel.source_name.lower())
            target_id = entity_ids.get(rel.target_name.lower())
            if source_id and target_id and source_id in node_ids and target_id in node_ids:
                edges.append(KnowledgeGraphEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relationship=rel.type,
                    weight=round(rel.confidence, 4),
                ))

        graph = KnowledgeGraph(nodes=nodes, edges=tuple(edges))

        return tuple(ucs_entities), graph, importance_scores

    @staticmethod
    def _compute_recency(
        mention_indices: list[int],
        total_messages: int,
    ) -> float:
        """Compute recency score (0-1). Higher = mentioned more recently."""
        if not mention_indices or total_messages <= 0:
            return 0.0
        latest = max(mention_indices)
        return latest / total_messages if total_messages > 0 else 0.0


# -- Pipeline Result ---------------------------------------------------------


@dataclass(frozen=True)
class EntityResult:
    """Immutable result container from the entity extraction pipeline.

    Attributes:
        entities: UCS-compatible Entity objects.
        graph: UCS-compatible KnowledgeGraph.
        scores: Entity UUID string -> importance score mapping.
        raw_entity_count: Number of entities before deduplication.
        deduped_entity_count: Number of entities after deduplication.
        relationship_count: Number of relationships extracted.
        spacy_available: Whether spaCy NER was used.
    """

    entities: tuple[Entity, ...]
    graph: KnowledgeGraph
    scores: dict[str, float]
    raw_entity_count: int
    deduped_entity_count: int
    relationship_count: int
    spacy_available: bool


# -- Main Pipeline -----------------------------------------------------------


class EntityPipeline:
    """Orchestrates the full entity extraction pipeline.

    Usage:
        pipeline = EntityPipeline()
        result = pipeline.extract(conversation)

    The pipeline runs these stages in order:
        1. SpaCy NER (people, orgs, locations)
        2. Technical extraction (paths, URLs, APIs, code, technologies)
        3. Deduplication (merge duplicates)
        4. Relationship extraction (co-occurrence + verb patterns)
        5. Knowledge graph construction (importance scoring + UCS building)
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        enable_spacy: bool = True,
    ) -> None:
        """Initialize the entity extraction pipeline.

        Args:
            spacy_model: Name of spaCy model to use.
            enable_spacy: If False, skip spaCy NER entirely.
        """
        self._spacy_extractor = SpaCyExtractor(model_name=spacy_model) if enable_spacy else None
        self._technical_extractor = TechnicalExtractor()
        self._relationship_extractor = RelationshipExtractor()
        self._deduplicator = EntityDeduplicator()
        self._graph_builder = KnowledgeGraphBuilder()

    def extract(self, conversation: Conversation) -> EntityResult:
        """Run the full entity extraction pipeline on a conversation.

        Args:
            conversation: A normalized CCR Conversation.

        Returns:
            EntityResult with entities, graph, scores, and pipeline stats.
        """
        messages = conversation.messages
        if not messages:
            return EntityResult(
                entities=(),
                graph=KnowledgeGraph(),
                scores={},
                raw_entity_count=0,
                deduped_entity_count=0,
                relationship_count=0,
                spacy_available=False,
            )

        # Stage 1: SpaCy NER
        spacy_entities: list[RawEntity] = []
        spacy_available = False
        if self._spacy_extractor is not None:
            spacy_entities = self._spacy_extractor.extract(messages)
            spacy_available = self._spacy_extractor.available

        # Stage 2: Technical extraction
        technical_entities = self._technical_extractor.extract(messages)

        # Combine all raw entities
        all_raw = spacy_entities + technical_entities
        raw_count = len(all_raw)

        # Stage 3: Deduplication
        deduped = self._deduplicator.deduplicate(all_raw)
        deduped_count = len(deduped)

        # Stage 4: Relationship extraction
        relationships = self._relationship_extractor.extract(deduped, messages)
        rel_count = len(relationships)

        # Stage 5: Knowledge graph construction
        entities, graph, scores = self._graph_builder.build(
            deduped, relationships, len(messages),
        )

        return EntityResult(
            entities=entities,
            graph=graph,
            scores=scores,
            raw_entity_count=raw_count,
            deduped_entity_count=deduped_count,
            relationship_count=rel_count,
            spacy_available=spacy_available,
        )
