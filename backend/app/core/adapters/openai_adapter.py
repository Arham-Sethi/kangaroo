"""UCS to OpenAI message array adapter.

Converts a Universal Context Schema into the OpenAI Chat Completions
message format: [{"role": "...", "content": "..."}]

The adapter reconstructs context as a system prompt containing:
    - Conversation summary (global + topic)
    - Key entities and relationships
    - Active decisions and tasks
    - User preferences

This gives GPT full awareness of the prior conversation without
needing to replay every message.

Output format matches:
    https://platform.openai.com/docs/api-reference/chat/create
"""

from __future__ import annotations

from typing import Any

from app.core.adapters.base import AdaptedOutput, BaseAdapter
from app.core.models.ucs import UniversalContextSchema


class OpenAIAdapter(BaseAdapter):
    """Convert UCS to OpenAI Chat Completions message format."""

    @property
    def format_name(self) -> str:
        return "openai"

    def adapt(self, ucs: UniversalContextSchema) -> AdaptedOutput:
        """Convert UCS to OpenAI message array.

        Produces:
            - system message with full context summary
            - assistant message acknowledging the context (for smooth continuation)

        Args:
            ucs: Universal Context Schema to convert.

        Returns:
            AdaptedOutput with OpenAI-format messages.
        """
        context_summary = self._build_context_summary(ucs)
        system_prompt = self._build_system_prompt(ucs, context_summary)

        messages: list[dict[str, Any]] = []

        # System message with context
        messages.append({
            "role": "system",
            "content": system_prompt,
        })

        # Include artifacts as reference messages
        for artifact in ucs.artifacts[:5]:
            content = self._format_artifact(artifact)
            if content:
                messages.append({
                    "role": "assistant",
                    "content": content,
                })

        # Continuation message
        messages.append({
            "role": "assistant",
            "content": self._build_continuation_message(ucs),
        })

        total_text = "".join(m.get("content", "") for m in messages)
        token_estimate = self._estimate_tokens(total_text)

        return AdaptedOutput(
            format_name="openai",
            messages=messages,
            system_prompt=system_prompt,
            metadata={
                "model_suggestion": self._suggest_model(token_estimate),
                "source_llm": ucs.session_meta.source_llm.value,
                "entity_count": len(ucs.entities),
                "summary_count": len(ucs.summaries),
            },
            token_estimate=token_estimate,
        )

    def _build_system_prompt(
        self,
        ucs: UniversalContextSchema,
        context_summary: str,
    ) -> str:
        """Build the OpenAI system prompt with injected context."""
        parts = [
            "You are continuing a conversation that was started with a different AI assistant.",
            f"The conversation had {ucs.session_meta.message_count} messages "
            f"and covered the following context:\n",
            context_summary,
            "\nContinue the conversation naturally, respecting all decisions made "
            "and tasks identified. Do not repeat information the user already knows.",
        ]
        return "\n".join(parts)

    @staticmethod
    def _format_artifact(artifact: Any) -> str:
        """Format an artifact for inclusion as a message."""
        title = artifact.title or f"{artifact.type.value}"
        if artifact.language:
            return f"Here's the {title} we created:\n```{artifact.language}\n{artifact.content}\n```"
        return f"Here's the {title}:\n{artifact.content}"

    @staticmethod
    def _build_continuation_message(ucs: UniversalContextSchema) -> str:
        """Build a natural continuation message."""
        from app.core.models.ucs import TaskStatus
        active_tasks = [t for t in ucs.tasks if t.status == TaskStatus.ACTIVE]

        if active_tasks:
            task_list = ", ".join(t.description[:80] for t in active_tasks[:3])
            return (
                f"I have the full context from your previous conversation. "
                f"The open tasks are: {task_list}. How would you like to continue?"
            )
        return (
            "I have the full context from your previous conversation. "
            "How would you like to continue?"
        )

    @staticmethod
    def _suggest_model(token_estimate: int) -> str:
        """Suggest an OpenAI model based on context size."""
        if token_estimate > 100000:
            return "gpt-4o"
        if token_estimate > 16000:
            return "gpt-4o"
        return "gpt-4o-mini"
