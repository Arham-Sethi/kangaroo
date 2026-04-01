"""Comprehensive tests for the Hierarchical Summarization Pipeline.

Tests cover all components:
    1. TF-IDF scoring
    2. Topic detection
    3. Message-level summaries
    4. Topic-level summaries
    5. Global summary
    6. Full pipeline integration

Total: 45+ tests
"""

from __future__ import annotations

import pytest

from app.core.engine.ccr import (
    ContentBlock,
    ContentType,
    Conversation,
    Message,
    MessageRole,
    SourceFormat,
)
from app.core.engine.summarizer import (
    GlobalSummarizer,
    MessageSummarizer,
    SummarizationPipeline,
    SummaryResult,
    TFIDFScorer,
    TopicDetector,
    TopicSummarizer,
    _sentence_split,
    _tokenize,
)
from app.core.models.ucs import SummaryLevel


# -- Helpers -----------------------------------------------------------------


def _msg(text: str, role: MessageRole = MessageRole.USER) -> Message:
    return Message(
        role=role,
        content=(ContentBlock(type=ContentType.TEXT, text=text),),
    )


def _conversation(*messages: Message) -> Conversation:
    return Conversation(
        source_format=SourceFormat.GENERIC,
        messages=messages,
        message_count=len(messages),
    )


# == Utility Tests ===========================================================


class TestTokenize:
    def test_simple_text(self) -> None:
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_with_punctuation(self) -> None:
        tokens = _tokenize("Hello, world! How are you?")
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty_string(self) -> None:
        assert _tokenize("") == []

    def test_numbers_included(self) -> None:
        tokens = _tokenize("Python 3.12 is great")
        assert "3" in tokens
        assert "12" in tokens


class TestSentenceSplit:
    def test_simple_sentences(self) -> None:
        text = "First sentence. Second sentence. Third one!"
        result = _sentence_split(text)
        assert len(result) == 3

    def test_single_sentence(self) -> None:
        result = _sentence_split("Just one sentence.")
        assert len(result) == 1

    def test_empty_string(self) -> None:
        assert _sentence_split("") == []

    def test_no_punctuation(self) -> None:
        result = _sentence_split("No punctuation here")
        assert len(result) == 1


# == TF-IDF Tests ===========================================================


class TestTFIDFScorer:
    def test_score_common_vs_rare(self) -> None:
        docs = [
            "the cat sat on the mat",
            "the dog sat on the mat",
            "a rare postgresql migration error occurred",
        ]
        scorer = TFIDFScorer(docs)
        # Rare technical terms should score higher
        rare_score = scorer.score_sentence("postgresql migration error")
        common_score = scorer.score_sentence("the cat sat")
        assert rare_score > common_score

    def test_empty_sentence(self) -> None:
        scorer = TFIDFScorer(["hello world"])
        assert scorer.score_sentence("") == 0.0

    def test_single_document(self) -> None:
        scorer = TFIDFScorer(["python is great for data science"])
        score = scorer.score_sentence("python data science")
        assert score > 0.0

    def test_stop_words_ignored(self) -> None:
        scorer = TFIDFScorer(["the is a an"])
        assert scorer.score_sentence("the is a") == 0.0


# == Topic Detector Tests ====================================================


class TestTopicDetector:
    def setup_method(self) -> None:
        self.detector = TopicDetector(similarity_threshold=0.15)

    def test_empty_messages(self) -> None:
        assert self.detector.detect(()) == []

    def test_single_message(self) -> None:
        msgs = (_msg("Hello world"),)
        result = self.detector.detect(msgs)
        assert len(result) == 1
        assert result[0][1] == [0]

    def test_similar_messages_same_cluster(self) -> None:
        msgs = (
            _msg("Python is great for data science and machine learning"),
            _msg("Data science uses Python for machine learning models"),
        )
        result = self.detector.detect(msgs)
        # Similar messages should be in the same cluster
        assert len(result) == 1
        assert len(result[0][1]) == 2

    def test_different_topics_separate_clusters(self) -> None:
        msgs = (
            _msg("Python is a programming language for data science"),
            _msg("Machine learning uses Python for training models"),
            _msg("The restaurant serves excellent Italian pasta and pizza"),
            _msg("Italian cuisine includes risotto and tiramisu desserts"),
        )
        result = self.detector.detect(msgs)
        # Should detect at least 2 distinct topics
        assert len(result) >= 2

    def test_cluster_labels_generated(self) -> None:
        msgs = (
            _msg("Python Django FastAPI web development"),
        )
        result = self.detector.detect(msgs)
        assert len(result) == 1
        label = result[0][0]
        assert len(label) > 0

    def test_jaccard_similarity_identical(self) -> None:
        assert TopicDetector._jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_similarity_disjoint(self) -> None:
        assert TopicDetector._jaccard({"a"}, {"b"}) == 0.0

    def test_jaccard_similarity_empty(self) -> None:
        assert TopicDetector._jaccard(set(), set()) == 1.0


# == Message Summarizer Tests ================================================


class TestMessageSummarizer:
    def setup_method(self) -> None:
        self.summarizer = MessageSummarizer()

    def test_empty_messages(self) -> None:
        result = self.summarizer.summarize(())
        assert result == []

    def test_short_message_kept_verbatim(self) -> None:
        msgs = (_msg("thanks!"),)
        result = self.summarizer.summarize(msgs)
        assert len(result) == 1
        assert result[0].content == "thanks!"
        assert result[0].level == SummaryLevel.MESSAGE

    def test_long_message_summarized(self) -> None:
        long_text = (
            "PostgreSQL is an advanced relational database system. "
            "It supports complex queries and transactions. "
            "It also has excellent support for JSON data types. "
            "Many enterprise applications rely on PostgreSQL for critical data storage. "
            "The ecosystem includes tools like pgAdmin and pgvector."
        )
        msgs = (_msg(long_text),)
        result = self.summarizer.summarize(msgs)
        assert len(result) == 1
        # Summary should be shorter than original
        assert len(result[0].content) < len(long_text)

    def test_covers_messages_range(self) -> None:
        msgs = (_msg("First message"), _msg("Second message"))
        result = self.summarizer.summarize(msgs)
        assert result[0].covers_messages == (0, 0)
        assert result[1].covers_messages == (1, 1)

    def test_empty_text_skipped(self) -> None:
        msgs = (_msg(""), _msg("Real content here"))
        result = self.summarizer.summarize(msgs)
        assert len(result) == 1

    def test_token_count_populated(self) -> None:
        msgs = (_msg("Python is a great programming language"),)
        result = self.summarizer.summarize(msgs)
        assert result[0].token_count > 0


# == Topic Summarizer Tests ==================================================


class TestTopicSummarizer:
    def setup_method(self) -> None:
        self.summarizer = TopicSummarizer()

    def test_empty_input(self) -> None:
        summaries, clusters = self.summarizer.summarize((), [])
        assert summaries == []
        assert clusters == []

    def test_single_topic(self) -> None:
        msgs = (
            _msg("Python is great for web development"),
            _msg("Django and Flask are popular Python web frameworks"),
        )
        clusters = [("python, web, development", [0, 1])]
        summaries, topic_clusters = self.summarizer.summarize(msgs, clusters)
        assert len(summaries) == 1
        assert summaries[0].level == SummaryLevel.TOPIC
        assert len(topic_clusters) == 1

    def test_covers_messages_range(self) -> None:
        msgs = (
            _msg("msg 0"),
            _msg("msg 1"),
            _msg("msg 2"),
        )
        clusters = [("cluster1", [0, 1, 2])]
        summaries, _ = self.summarizer.summarize(msgs, clusters)
        assert summaries[0].covers_messages == (0, 2)

    def test_topic_cluster_has_label(self) -> None:
        msgs = (_msg("Python web development"),)
        clusters = [("python, web", [0])]
        _, topic_clusters = self.summarizer.summarize(msgs, clusters)
        assert topic_clusters[0].label == "python, web"


# == Global Summarizer Tests =================================================


class TestGlobalSummarizer:
    def setup_method(self) -> None:
        self.summarizer = GlobalSummarizer()

    def test_empty_messages(self) -> None:
        assert self.summarizer.summarize(()) is None

    def test_single_message(self) -> None:
        msgs = (_msg("Python is great"),)
        result = self.summarizer.summarize(msgs)
        assert result is not None
        assert result.level == SummaryLevel.GLOBAL

    def test_covers_all_messages(self) -> None:
        msgs = (_msg("First"), _msg("Second"), _msg("Third"))
        result = self.summarizer.summarize(msgs)
        assert result is not None
        assert result.covers_messages == (0, 2)

    def test_respects_word_budget(self) -> None:
        # Create a long conversation
        msgs = tuple(
            _msg(f"This is message number {i} with some interesting content about topic {i}. "
                 f"It contains several sentences. Each one adds detail. More words here.")
            for i in range(50)
        )
        result = self.summarizer.summarize(msgs)
        assert result is not None
        assert result.token_count <= 250  # ~200 budget with some slack

    def test_all_empty_messages(self) -> None:
        msgs = (_msg(""), _msg(""), _msg(""))
        result = self.summarizer.summarize(msgs)
        assert result is None


# == Full Pipeline Tests =====================================================


class TestSummarizationPipeline:
    def setup_method(self) -> None:
        self.pipeline = SummarizationPipeline()

    def test_empty_conversation(self) -> None:
        conv = _conversation()
        result = self.pipeline.summarize(conv)
        assert isinstance(result, SummaryResult)
        assert result.summaries == ()
        assert result.topic_clusters == ()
        assert result.message_summary_count == 0
        assert result.topic_summary_count == 0
        assert not result.has_global_summary

    def test_single_message_conversation(self) -> None:
        conv = _conversation(_msg("Python is a versatile programming language for many tasks"))
        result = self.pipeline.summarize(conv)
        assert result.message_summary_count >= 1
        assert result.has_global_summary

    def test_multi_message_conversation(self) -> None:
        conv = _conversation(
            _msg("I need help building a REST API"),
            _msg("Let's use Python with FastAPI and PostgreSQL"),
            _msg("We should add authentication with JWT tokens"),
            _msg("Deploy to AWS using Docker containers"),
        )
        result = self.pipeline.summarize(conv)
        assert result.message_summary_count >= 1
        assert result.has_global_summary
        assert result.total_token_count > 0

    def test_all_summary_levels_present(self) -> None:
        conv = _conversation(
            _msg("Python web development with Django is powerful and productive"),
            _msg("Django provides an ORM and admin interface out of the box"),
            _msg("For REST APIs, Django REST Framework is the standard choice"),
        )
        result = self.pipeline.summarize(conv)
        levels = {s.level for s in result.summaries}
        assert SummaryLevel.MESSAGE in levels
        assert SummaryLevel.GLOBAL in levels

    def test_topic_clusters_detected(self) -> None:
        conv = _conversation(
            _msg("Python Django web development framework is excellent"),
            _msg("Django ORM makes database queries simple and pythonic"),
            _msg("The Italian restaurant has excellent pasta and fresh pizza"),
            _msg("Italian cuisine features fresh ingredients and olive oil"),
        )
        result = self.pipeline.summarize(conv)
        # Should have at least some topic clusters
        assert len(result.topic_clusters) >= 1

    def test_result_immutability(self) -> None:
        conv = _conversation(_msg("Hello world"))
        result = self.pipeline.summarize(conv)
        # SummaryResult is frozen dataclass
        with pytest.raises(AttributeError):
            result.has_global_summary = False  # type: ignore[misc]

    def test_summary_objects_immutable(self) -> None:
        conv = _conversation(_msg("Python is great for building web applications"))
        result = self.pipeline.summarize(conv)
        if result.summaries:
            with pytest.raises(Exception):
                result.summaries[0].content = "changed"  # type: ignore[misc]

    def test_realistic_conversation(self) -> None:
        conv = _conversation(
            _msg("I want to build a SaaS application for project management"),
            _msg("The backend should use Python FastAPI with PostgreSQL database"),
            _msg("For the frontend, let's use React with TypeScript and Tailwind CSS"),
            _msg("Authentication should use JWT with refresh token rotation"),
            _msg("We need real-time updates via WebSocket connections"),
            _msg("Deploy to AWS using ECS with auto-scaling and CloudFront CDN"),
            _msg("Add Stripe integration for subscription billing"),
            _msg("Monitoring with Prometheus, Grafana, and Sentry for error tracking"),
        )
        result = self.pipeline.summarize(conv)
        assert result.message_summary_count >= 4
        assert result.has_global_summary
        assert result.topic_summary_count >= 1
        assert len(result.topic_clusters) >= 1
        assert result.total_token_count > 0
