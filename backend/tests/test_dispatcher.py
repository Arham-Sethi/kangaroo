"""Tests for SmartDispatcher — auto-route tasks to best model.

Tests cover:
    - Task classification (coding, creative, analysis, explanation)
    - Model selection based on task type
    - Top-N selection
    - Confidence scoring
    - Custom capabilities
    - Unknown task type fallback
"""

import pytest

from app.core.cockpit.dispatcher import (
    DispatchDecision,
    ModelCapability,
    SmartDispatcher,
    TaskType,
)


class TestTaskClassification:
    def test_coding_task(self) -> None:
        dispatcher = SmartDispatcher()
        task_type, confidence = dispatcher.classify_task(
            "Implement a REST API endpoint with Python"
        )
        assert task_type == TaskType.CODING
        assert confidence > 0.3

    def test_creative_task(self) -> None:
        dispatcher = SmartDispatcher()
        task_type, _ = dispatcher.classify_task(
            "Write a creative story about brainstorming ideas"
        )
        assert task_type == TaskType.CREATIVE

    def test_analysis_task(self) -> None:
        dispatcher = SmartDispatcher()
        task_type, _ = dispatcher.classify_task(
            "Analyze the performance metrics and compare benchmarks"
        )
        assert task_type == TaskType.ANALYSIS

    def test_explanation_task(self) -> None:
        dispatcher = SmartDispatcher()
        task_type, _ = dispatcher.classify_task(
            "Explain how async await works in a tutorial"
        )
        assert task_type == TaskType.EXPLANATION

    def test_general_fallback(self) -> None:
        dispatcher = SmartDispatcher()
        task_type, confidence = dispatcher.classify_task("xyz abc 123")
        assert task_type == TaskType.GENERAL
        assert confidence <= 0.5


class TestModelSelection:
    def test_select_best_for_coding(self) -> None:
        dispatcher = SmartDispatcher()
        decision = dispatcher.select("Write a Python function to sort a list")
        assert isinstance(decision, DispatchDecision)
        assert decision.selected_model  # Not empty
        assert decision.task_type == TaskType.CODING
        assert decision.confidence > 0

    def test_select_has_alternatives(self) -> None:
        dispatcher = SmartDispatcher()
        decision = dispatcher.select("Build a REST API")
        assert len(decision.alternatives) >= 1

    def test_select_has_reasoning(self) -> None:
        dispatcher = SmartDispatcher()
        decision = dispatcher.select("Review this code for security")
        assert decision.reasoning
        assert len(decision.reasoning) > 10

    def test_select_with_limited_models(self) -> None:
        dispatcher = SmartDispatcher(
            available_models=["openai/gpt-4o"]
        )
        decision = dispatcher.select("Write code")
        assert decision.selected_model == "openai/gpt-4o"
        assert decision.alternatives == []


class TestTopNSelection:
    def test_top_2(self) -> None:
        dispatcher = SmartDispatcher()
        decisions = dispatcher.select_top_n("Write a REST API", n=2)
        assert len(decisions) == 2
        assert decisions[0].selected_model != decisions[1].selected_model

    def test_top_n_exceeds_available(self) -> None:
        dispatcher = SmartDispatcher(
            available_models=["openai/gpt-4o"]
        )
        decisions = dispatcher.select_top_n("Test", n=5)
        assert len(decisions) == 1


class TestCustomCapabilities:
    def test_custom_model(self) -> None:
        custom = [
            ModelCapability(
                model_id="custom/model",
                display_name="Custom",
                scores={
                    TaskType.CODING: 1.0,
                    TaskType.CREATIVE: 0.1,
                    TaskType.ANALYSIS: 0.1,
                    TaskType.EXPLANATION: 0.1,
                    TaskType.GENERAL: 0.1,
                },
            ),
        ]
        dispatcher = SmartDispatcher(capabilities=custom)
        decision = dispatcher.select("Write code to implement a function")
        assert decision.selected_model == "custom/model"

    def test_no_available_models(self) -> None:
        dispatcher = SmartDispatcher(available_models=[])
        decision = dispatcher.select("Anything")
        assert decision.selected_model == ""
        assert "No available models" in decision.reasoning
