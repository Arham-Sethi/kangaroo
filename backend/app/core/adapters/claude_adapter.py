"""UCS to Claude message format adapter.

Converts a Universal Context Schema into the Anthropic Claude Messages
API format with typed content blocks.

Claude's format is distinct from OpenAI:
    - System prompt is a top-level field, NOT a message
    - Content is a list of typed blocks [{type: "text", text: "..."}]
    - Messages must strictly alternate user/assistant

Output format matches:
    https://docs.anthropic.com/en/docs/api-reference#messages
"""

from __future__ import annotations

from typing import Any

from app.core.adapters.base import AdaptedOutput, BaseAdapter
from app.core.models.ucs import UniversalContextSchema


class ClaudeAdapter(BaseAdapter):
    """Convert UCS to Anthropic Claude message format."""

    @property
    def format_name(self) -> str:
        return "claude"

    def adapt(self, ucs: UniversalContextSchema) -> AdaptedOutput:
        """Convert UCS to Claude Messages API format.

        Produces:
            - system: top-level context summary (NOT a message)
            - messages: alternating user/assistant with context priming

        Args:
            ucs: Universal Context Schema to convert.

        Returns:
            AdaptedOutput with Claude-format messages.
        """
        context_summary = self._build_context_summary(ucs)
        system_prompt = self._build_system_prompt(ucs, context_summary)

        messages: list[dict[str, Any]] = []

        # Claude requires messages to start with user role
        # Inject a context-priming user message
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": self._build_priming_message(ucs),
                }
            ],
        })

        # Assistant acknowledgment with context awareness
        assistant_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": self._build_continuation_message(ucs),
            }
        ]

        # Include artifacts as code blocks in assistant message
        for artifact in ucs.artifacts[:5]:
            block = self._format_artifact_block(artifact)
            if block:
                assistant_blocks.append(block)

        messages.append({
            "role": "assistant",
            "content": assistant_blocks,
        })

        total_text = system_prompt + "".join(
            str(m.get("content", "")) for m in messages
        )
        token_estimate = self._estimate_tokens(total_text)

        return AdaptedOutput(
            format_name="claude",
            messages=messages,
            system_prompt=system_prompt,
            metadata={
                "model_suggestion": self._suggest_model(token_estimate),
                "source_llm": ucs.session_meta.source_llm.value,
                "entity_count": len(ucs.entities),
                "summary_count": len(ucs.summaries),
                "api_format": "messages",
            },
            token_estimate=token_estimate,
        )

    def _build_system_prompt(
        self,
        ucs: UniversalContextSchema,
        context_summary: str,
    ) -> str:
        """Build Claude's top-level system prompt."""
        parts = [
            "You are continuing a conversation that was started with a different AI assistant.",
            f"The original conversation had {ucs.session_meta.message_count} messages.",
            "",
            context_summary,
            "",
            "Continue naturally. Respect all prior decisions and open tasks. "
            "Do not repeat information already established.",
        ]
        return "\n".join(parts)

    @staticmethod
    def _build_priming_message(ucs: UniversalContextSchema) -> str:
        """Build the user priming message that establishes context."""
        from app.core.models.ucs import TaskStatus
        active_tasks = [t for t in ucs.tasks if t.status == TaskStatus.ACTIVE]

        parts = [
            "I'm continuing a conversation from another AI assistant. "
            "You should have the full context in your system prompt.",
        ]

        if active_tasks:
            parts.append(
                f"We have {len(active_tasks)} open task(s). "
                "Please pick up where we left off."
            )
        else:
            parts.append("Please pick up where we left off.")

        return " ".join(parts)

    @staticmethod
    def _build_continuation_message(ucs: UniversalContextSchema) -> str:
        """Build the assistant continuation acknowledgment."""
        from app.core.models.ucs import TaskStatus
        active_tasks = [t for t in ucs.tasks if t.status == TaskStatus.ACTIVE]

        if active_tasks:
            task_list = "\n".join(
                f"- {t.description[:100]}" for t in active_tasks[:5]
            )
            return (
                f"I have the full context from your previous conversation. "
                f"Here are the open tasks:\n{task_list}\n\n"
                f"What would you like to work on?"
            )
        return (
            "I have the full context from your previous conversation. "
            "I'm ready to continue. What would you like to do next?"
        )

    @staticmethod
    def _format_artifact_block(artifact: Any) -> dict[str, Any] | None:
        """Format an artifact as a Claude text content block."""
        if not artifact.content:
            return None
        title = artifact.title or artifact.type.value
        if artifact.language:
            text = f"Previous artifact ({title}):\n```{artifact.language}\n{artifact.content}\n```"
        else:
            text = f"Previous artifact ({title}):\n{artifact.content}"
        return {"type": "text", "text": text}

    @staticmethod
    def _suggest_model(token_estimate: int) -> str:
        """Suggest a Claude model based on context size."""
        if token_estimate > 100000:
            return "claude-sonnet-4-20250514"
        if token_estimate > 16000:
            return "claude-sonnet-4-20250514"
        return "claude-haiku-4-5-20251001"
