"""Hierarchical 3-tier summarization (message, topic, global).

The Summarization Pipeline compresses a conversation into three levels:

    Level 1 — MESSAGE:  One sentence per message (fine-grained)
    Level 2 — TOPIC:    One paragraph per topic cluster (medium)
    Level 3 — GLOBAL:   ~200 words covering the entire conversation (coarse)

During context compression, message-level summaries are dropped first,
then topic-level. The global summary is ALWAYS preserved — it's the
last line of defense against total context loss.

This is a LOCAL-FIRST implementation — no LLM API calls required.
It uses extractive summarization (selecting important sentences) rather
than abstractive (generating new text). This means:
    - Zero cost (no API calls)
    - Deterministic (same input = same output)
    - Fast (~50ms for a 100-message conversation)
    - Works offline

For higher quality summaries, the pipeline can optionally use an LLM
adapter (Phase 3+) to generate abstractive summaries.

Usage:
    from app.core.engine.summarizer import SummarizationPipeline, SummaryResult

    pipeline = SummarizationPipeline()
    result = pipeline.summarize(conversation)
    # result.summaries     -> tuple[Summary, ...]
    # result.topic_clusters -> tuple[TopicCluster, ...]
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from uuid import uuid4

from app.core.engine.ccr import Conversation, Message, MessageRole
from app.core.models.ucs import Summary, SummaryLevel, TopicCluster


# -- Text utilities ----------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens, stripping punctuation."""
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _sentence_split(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


# English stop words (common words that carry little meaning)
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "about", "up", "that",
    "this", "it", "its", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their", "what",
    "which", "who", "whom", "these", "those", "am", "also", "still",
})


class TFIDFScorer:
    """Compute TF-IDF scores for sentences in a document.

    TF-IDF (Term Frequency - Inverse Document Frequency) identifies
    sentences that contain words that are important to THIS specific
    conversation but rare overall. A sentence about "PostgreSQL migration"
    scores higher than one about "please help me" because the technical
    terms carry more information.
    """

    def __init__(self, documents: list[str]) -> None:
        """Initialize with a list of documents (sentences).

        Args:
            documents: List of text strings (sentences) to score.
        """
        self._documents = documents
        self._doc_count = len(documents)
        self._idf: dict[str, float] = {}
        self._build_idf()

    def _build_idf(self) -> None:
        """Compute inverse document frequency for all terms."""
        doc_freq: Counter[str] = Counter()
        for doc in self._documents:
            unique_tokens = set(_tokenize(doc))
            for token in unique_tokens:
                if token not in _STOP_WORDS:
                    doc_freq[token] += 1

        for token, freq in doc_freq.items():
            self._idf[token] = math.log((self._doc_count + 1) / (freq + 1)) + 1

    def score_sentence(self, sentence: str) -> float:
        """Compute TF-IDF score for a single sentence.

        Args:
            sentence: Text to score.

        Returns:
            Float score (higher = more important).
        """
        tokens = _tokenize(sentence)
        if not tokens:
            return 0.0

        # Term frequency within this sentence
        tf: Counter[str] = Counter()
        for token in tokens:
            if token not in _STOP_WORDS:
                tf[token] += 1

        # TF-IDF score = sum of (tf * idf) for each token
        score = 0.0
        for token, count in tf.items():
            tf_val = count / len(tokens)
            idf_val = self._idf.get(token, 1.0)
            score += tf_val * idf_val

        return score


class TopicDetector:
    """Group messages into topic clusters using keyword overlap.

    This is a lightweight alternative to embedding-based clustering.
    It uses a sliding window approach: when the keyword overlap between
    consecutive messages drops below a threshold, a new topic starts.

    Each cluster gets a label derived from its most frequent keywords.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.15,
        min_cluster_size: int = 1,
    ) -> None:
        """Initialize the topic detector.

        Args:
            similarity_threshold: Jaccard similarity below this starts a new topic.
            min_cluster_size: Minimum messages per cluster.
        """
        self._threshold = similarity_threshold
        self._min_size = min_cluster_size

    def detect(
        self, messages: tuple[Message, ...],
    ) -> list[tuple[str, list[int]]]:
        """Detect topic clusters in a conversation.

        Args:
            messages: Conversation messages.

        Returns:
            List of (topic_label, message_indices) tuples.
        """
        if not messages:
            return []

        # Extract keyword sets per message
        keyword_sets: list[set[str]] = []
        for msg in messages:
            tokens = _tokenize(msg.full_text)
            keywords = {t for t in tokens if t not in _STOP_WORDS and len(t) > 2}
            keyword_sets.append(keywords)

        # Cluster by keyword overlap
        clusters: list[list[int]] = [[0]]
        for i in range(1, len(messages)):
            prev_keywords = keyword_sets[i - 1]
            curr_keywords = keyword_sets[i]
            similarity = self._jaccard(prev_keywords, curr_keywords)

            if similarity >= self._threshold:
                clusters[-1].append(i)
            else:
                clusters.append([i])

        # Generate labels and filter by min size
        result: list[tuple[str, list[int]]] = []
        for cluster_indices in clusters:
            if len(cluster_indices) < self._min_size:
                # Merge into previous cluster if possible
                if result:
                    result[-1][1].extend(cluster_indices)
                    continue
            # Label = top 3 keywords from this cluster
            all_keywords: Counter[str] = Counter()
            for idx in cluster_indices:
                all_keywords.update(keyword_sets[idx])
            top_words = [w for w, _ in all_keywords.most_common(3)]
            label = ", ".join(top_words) if top_words else "General"
            result.append((label, cluster_indices))

        return result

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0


class MessageSummarizer:
    """Generate message-level summaries (Level 1).

    Extracts the single most important sentence from each message
    using TF-IDF scoring. Short messages (under 50 chars) are kept
    verbatim — no point summarizing "thanks!" further.
    """

    _SHORT_THRESHOLD = 50  # chars

    def summarize(
        self, messages: tuple[Message, ...],
    ) -> list[Summary]:
        """Generate one summary per message.

        Args:
            messages: Conversation messages.

        Returns:
            List of MESSAGE-level Summary objects.
        """
        if not messages:
            return []

        # Collect all sentences for TF-IDF
        all_sentences: list[str] = []
        for msg in messages:
            text = msg.full_text.strip()
            if text:
                all_sentences.extend(_sentence_split(text))

        scorer = TFIDFScorer(all_sentences) if all_sentences else None

        summaries: list[Summary] = []
        for i, msg in enumerate(messages):
            text = msg.full_text.strip()
            if not text:
                continue

            if len(text) <= self._SHORT_THRESHOLD:
                summary_text = text
            else:
                sentences = _sentence_split(text)
                if not sentences:
                    summary_text = text[:200]
                elif scorer is not None:
                    scored = [(s, scorer.score_sentence(s)) for s in sentences]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    summary_text = scored[0][0]
                else:
                    summary_text = sentences[0]

            token_count = len(_tokenize(summary_text))
            summaries.append(Summary(
                level=SummaryLevel.MESSAGE,
                content=summary_text,
                token_count=token_count,
                covers_messages=(i, i),
            ))

        return summaries


class TopicSummarizer:
    """Generate topic-level summaries (Level 2).

    Combines the top sentences from each topic cluster into a
    coherent paragraph. Uses TF-IDF to select the most important
    sentences from the cluster.
    """

    _MAX_SENTENCES_PER_TOPIC = 5

    def summarize(
        self,
        messages: tuple[Message, ...],
        clusters: list[tuple[str, list[int]]],
    ) -> tuple[list[Summary], list[TopicCluster]]:
        """Generate topic-level summaries.

        Args:
            messages: Conversation messages.
            clusters: Topic clusters from TopicDetector.

        Returns:
            Tuple of (summaries, topic_cluster_objects).
        """
        if not messages or not clusters:
            return [], []

        summaries: list[Summary] = []
        topic_clusters: list[TopicCluster] = []

        for label, indices in clusters:
            # Gather all text from this cluster
            cluster_texts: list[str] = []
            for idx in indices:
                if idx < len(messages):
                    text = messages[idx].full_text.strip()
                    if text:
                        cluster_texts.append(text)

            if not cluster_texts:
                continue

            # Extract sentences and score them
            all_sentences: list[str] = []
            for text in cluster_texts:
                all_sentences.extend(_sentence_split(text))

            if not all_sentences:
                continue

            scorer = TFIDFScorer(all_sentences)
            scored = [(s, scorer.score_sentence(s)) for s in all_sentences]
            scored.sort(key=lambda x: x[1], reverse=True)

            # Take top N sentences
            top_sentences = [s for s, _ in scored[:self._MAX_SENTENCES_PER_TOPIC]]
            summary_text = " ".join(top_sentences)

            start_idx = min(indices)
            end_idx = max(indices)
            token_count = len(_tokenize(summary_text))

            summaries.append(Summary(
                level=SummaryLevel.TOPIC,
                content=summary_text,
                token_count=token_count,
                covers_messages=(start_idx, end_idx),
            ))

            topic_clusters.append(TopicCluster(
                id=uuid4(),
                label=label,
                message_indices=tuple(indices),
            ))

        return summaries, topic_clusters


class GlobalSummarizer:
    """Generate a global summary (Level 3).

    The global summary is ~200 words covering the ENTIRE conversation.
    It's the last thing to be compressed — if everything else is gone,
    this one summary still gives the target LLM enough context to
    continue meaningfully.

    Strategy: select the highest-scoring sentences from the entire
    conversation, respecting a word budget.
    """

    _WORD_BUDGET = 200
    _MAX_SENTENCES = 10

    def summarize(
        self, messages: tuple[Message, ...],
    ) -> Summary | None:
        """Generate a single global summary.

        Args:
            messages: Conversation messages.

        Returns:
            A GLOBAL-level Summary, or None if no content.
        """
        if not messages:
            return None

        # Gather all text
        all_sentences: list[str] = []
        for msg in messages:
            text = msg.full_text.strip()
            if text:
                all_sentences.extend(_sentence_split(text))

        if not all_sentences:
            return None

        scorer = TFIDFScorer(all_sentences)
        scored = [(s, scorer.score_sentence(s)) for s in all_sentences]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select sentences within word budget
        selected: list[str] = []
        word_count = 0
        for sentence, _ in scored:
            words = len(_tokenize(sentence))
            if word_count + words > self._WORD_BUDGET and selected:
                break
            selected.append(sentence)
            word_count += words
            if len(selected) >= self._MAX_SENTENCES:
                break

        if not selected:
            return None

        summary_text = " ".join(selected)
        total_msgs = len(messages)

        return Summary(
            level=SummaryLevel.GLOBAL,
            content=summary_text,
            token_count=len(_tokenize(summary_text)),
            covers_messages=(0, total_msgs - 1),
        )


# -- Pipeline Result ---------------------------------------------------------


@dataclass(frozen=True)
class SummaryResult:
    """Immutable result from the summarization pipeline.

    Attributes:
        summaries: All summaries (message + topic + global levels).
        topic_clusters: Topic cluster objects for the UCS.
        message_summary_count: Number of message-level summaries.
        topic_summary_count: Number of topic-level summaries.
        has_global_summary: Whether a global summary was generated.
        total_token_count: Combined token count across all summaries.
    """

    summaries: tuple[Summary, ...]
    topic_clusters: tuple[TopicCluster, ...]
    message_summary_count: int
    topic_summary_count: int
    has_global_summary: bool
    total_token_count: int


# -- Main Pipeline -----------------------------------------------------------


class SummarizationPipeline:
    """Orchestrates the 3-tier hierarchical summarization pipeline.

    Usage:
        pipeline = SummarizationPipeline()
        result = pipeline.summarize(conversation)

    Stages:
        1. Topic detection (keyword overlap clustering)
        2. Message-level summaries (TF-IDF extractive)
        3. Topic-level summaries (TF-IDF per cluster)
        4. Global summary (TF-IDF across entire conversation)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.15,
        min_cluster_size: int = 1,
    ) -> None:
        """Initialize the summarization pipeline.

        Args:
            similarity_threshold: Jaccard similarity for topic boundaries.
            min_cluster_size: Minimum messages per topic cluster.
        """
        self._topic_detector = TopicDetector(
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
        )
        self._message_summarizer = MessageSummarizer()
        self._topic_summarizer = TopicSummarizer()
        self._global_summarizer = GlobalSummarizer()

    def summarize(self, conversation: Conversation) -> SummaryResult:
        """Run the full summarization pipeline on a conversation.

        Args:
            conversation: A normalized CCR Conversation.

        Returns:
            SummaryResult with summaries, topic clusters, and stats.
        """
        messages = conversation.messages
        if not messages:
            return SummaryResult(
                summaries=(),
                topic_clusters=(),
                message_summary_count=0,
                topic_summary_count=0,
                has_global_summary=False,
                total_token_count=0,
            )

        # Stage 1: Detect topics
        topic_clusters_raw = self._topic_detector.detect(messages)

        # Stage 2: Message-level summaries
        msg_summaries = self._message_summarizer.summarize(messages)

        # Stage 3: Topic-level summaries
        topic_summaries, topic_clusters = self._topic_summarizer.summarize(
            messages, topic_clusters_raw,
        )

        # Stage 4: Global summary
        global_summary = self._global_summarizer.summarize(messages)

        # Combine all summaries
        all_summaries: list[Summary] = []
        all_summaries.extend(msg_summaries)
        all_summaries.extend(topic_summaries)
        if global_summary:
            all_summaries.append(global_summary)

        total_tokens = sum(s.token_count for s in all_summaries)

        return SummaryResult(
            summaries=tuple(all_summaries),
            topic_clusters=tuple(topic_clusters),
            message_summary_count=len(msg_summaries),
            topic_summary_count=len(topic_summaries),
            has_global_summary=global_summary is not None,
            total_token_count=total_tokens,
        )
