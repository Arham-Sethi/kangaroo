"""Smart dispatcher — auto-route tasks to the best model.

Classifies tasks by type (coding, creative, analysis, general) and
routes them to the most suitable model based on a scoring matrix.

Production: Uses embeddings for task classification.
Fallback: Keyword-based classification.

Usage:
    dispatcher = SmartDispatcher()
    model = dispatcher.select("Write a Python REST API with FastAPI")
    # -> "anthropic/claude-sonnet-4" (strong at coding)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    """Categories for task classification."""

    CODING = "coding"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    EXPLANATION = "explanation"
    GENERAL = "general"


@dataclass(frozen=True)
class ModelCapability:
    """A model's capability score per task type.

    Attributes:
        model_id: Model identifier.
        scores: Map of TaskType -> score (0.0 to 1.0).
        display_name: Human-readable name.
    """

    model_id: str
    scores: dict[TaskType, float] = field(default_factory=dict)
    display_name: str = ""


@dataclass(frozen=True)
class DispatchDecision:
    """Result of smart dispatch selection.

    Attributes:
        selected_model: Best model for the task.
        task_type: Classified task type.
        confidence: How confident the dispatcher is (0..1).
        alternatives: Other models ranked by suitability.
        reasoning: Why this model was selected.
    """

    selected_model: str
    task_type: TaskType
    confidence: float
    alternatives: list[str]
    reasoning: str


# Keyword patterns for task classification
_TASK_PATTERNS: dict[TaskType, list[str]] = {
    TaskType.CODING: [
        "code", "implement", "function", "class", "api", "endpoint",
        "debug", "fix", "bug", "test", "refactor", "build", "deploy",
        "python", "javascript", "typescript", "rust", "go", "sql",
        "database", "migration", "schema", "rest", "graphql",
    ],
    TaskType.CREATIVE: [
        "write", "story", "poem", "creative", "brainstorm", "idea",
        "design", "name", "brand", "slogan", "marketing", "content",
        "blog", "article", "copy", "headline",
    ],
    TaskType.ANALYSIS: [
        "analyze", "compare", "evaluate", "pros", "cons", "tradeoff",
        "benchmark", "performance", "metrics", "data", "statistics",
        "research", "review", "assess", "audit",
    ],
    TaskType.EXPLANATION: [
        "explain", "how", "why", "what", "tutorial", "guide",
        "teach", "learn", "understand", "concept", "overview",
        "documentation", "docs",
    ],
}

# Default model capability scores
_DEFAULT_CAPABILITIES: list[ModelCapability] = [
    ModelCapability(
        model_id="anthropic/claude-sonnet-4",
        display_name="Claude Sonnet 4",
        scores={
            TaskType.CODING: 0.95,
            TaskType.CREATIVE: 0.80,
            TaskType.ANALYSIS: 0.90,
            TaskType.EXPLANATION: 0.90,
            TaskType.GENERAL: 0.90,
        },
    ),
    ModelCapability(
        model_id="openai/gpt-4o",
        display_name="GPT-4o",
        scores={
            TaskType.CODING: 0.88,
            TaskType.CREATIVE: 0.90,
            TaskType.ANALYSIS: 0.85,
            TaskType.EXPLANATION: 0.88,
            TaskType.GENERAL: 0.88,
        },
    ),
    ModelCapability(
        model_id="google/gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        scores={
            TaskType.CODING: 0.85,
            TaskType.CREATIVE: 0.75,
            TaskType.ANALYSIS: 0.92,
            TaskType.EXPLANATION: 0.85,
            TaskType.GENERAL: 0.85,
        },
    ),
    ModelCapability(
        model_id="openai/gpt-4o-mini",
        display_name="GPT-4o Mini",
        scores={
            TaskType.CODING: 0.70,
            TaskType.CREATIVE: 0.75,
            TaskType.ANALYSIS: 0.65,
            TaskType.EXPLANATION: 0.72,
            TaskType.GENERAL: 0.72,
        },
    ),
]


class SmartDispatcher:
    """Auto-routes tasks to the best available model.

    Classification is keyword-based for the fallback implementation.
    Production would use embeddings for more nuanced classification.
    """

    def __init__(
        self,
        capabilities: list[ModelCapability] | None = None,
        available_models: list[str] | None = None,
    ) -> None:
        self._capabilities = {
            c.model_id: c for c in (capabilities or _DEFAULT_CAPABILITIES)
        }
        self._available = set(
            available_models if available_models is not None
            else list(self._capabilities.keys())
        )

    def classify_task(self, task: str) -> tuple[TaskType, float]:
        """Classify a task description into a TaskType.

        Returns (task_type, confidence).
        """
        tokens = set(re.findall(r"[a-z]+", task.lower()))

        scores: dict[TaskType, int] = {}
        for task_type, patterns in _TASK_PATTERNS.items():
            score = sum(1 for p in patterns if p in tokens)
            scores[task_type] = score

        if not any(scores.values()):
            return TaskType.GENERAL, 0.3

        best_type = max(scores, key=lambda t: scores[t])
        total = sum(scores.values())
        confidence = scores[best_type] / total if total > 0 else 0.3

        return best_type, min(confidence, 1.0)

    def select(self, task: str) -> DispatchDecision:
        """Select the best model for a task.

        Args:
            task: Natural language task description.

        Returns:
            DispatchDecision with the selected model and reasoning.
        """
        task_type, confidence = self.classify_task(task)

        # Score each available model
        scored: list[tuple[float, str]] = []
        for model_id in self._available:
            cap = self._capabilities.get(model_id)
            if cap is None:
                continue
            score = cap.scores.get(task_type, 0.5)
            scored.append((score, model_id))

        scored.sort(key=lambda x: -x[0])

        if not scored:
            return DispatchDecision(
                selected_model="",
                task_type=task_type,
                confidence=0.0,
                alternatives=[],
                reasoning="No available models",
            )

        best_score, best_model = scored[0]
        alternatives = [m for _, m in scored[1:]]

        cap = self._capabilities[best_model]
        reasoning = (
            f"Task classified as '{task_type.value}' (confidence: {confidence:.0%}). "
            f"{cap.display_name or best_model} scores {best_score:.0%} for this task type."
        )

        return DispatchDecision(
            selected_model=best_model,
            task_type=task_type,
            confidence=confidence,
            alternatives=alternatives,
            reasoning=reasoning,
        )

    def select_top_n(self, task: str, n: int = 2) -> list[DispatchDecision]:
        """Select the top N models for a task."""
        task_type, confidence = self.classify_task(task)

        scored: list[tuple[float, str]] = []
        for model_id in self._available:
            cap = self._capabilities.get(model_id)
            if cap is None:
                continue
            score = cap.scores.get(task_type, 0.5)
            scored.append((score, model_id))

        scored.sort(key=lambda x: -x[0])

        results = []
        for i, (score, model_id) in enumerate(scored[:n]):
            others = [m for _, m in scored if m != model_id]
            cap = self._capabilities[model_id]
            results.append(
                DispatchDecision(
                    selected_model=model_id,
                    task_type=task_type,
                    confidence=confidence,
                    alternatives=others[:3],
                    reasoning=f"Rank {i+1}: {cap.display_name or model_id} ({score:.0%})",
                )
            )

        return results
