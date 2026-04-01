"""UCS to Gemini multi-part content adapter.

Converts a Universal Context Schema into the Google Gemini
generateContent format with parts-based messages.

Gemini's format differs from both OpenAI and Claude:
    - Roles are "user" and "model" (NOT "assistant")
    - Content is a list of "parts" [{text: "..."}]
    - System instruction is a top-level field

Output format matches:
    https://ai.google.dev/api/generate-content
"""

from __future__ import annotations

from typing import Any

from app.core.adapters.base import AdaptedOutput, BaseAdapter
from app.core.models.ucs import UniversalContextSchema


class GeminiAdapter(BaseAdapter):
    """Convert UCS to Google Gemini message format."""

    @property
    def format_name(self) -> str:
        return "gemini"

    def adapt(self, ucs: UniversalContextSchema) -> AdaptedOutput:
        """Convert UCS to Gemini generateContent format.

        Produces:
            - systemInstruction: context summary
            - contents: alternating user/model messages

        Args:
            ucs: Universal Context Schema to convert.

        Returns:
            AdaptedOutput with Gemini-format messages.
        """
        context_summary = self._build_context_summary(ucs)
        system_prompt = self._build_system_instruction(ucs, context_summary)

        messages: list[dict[str, Any]] = []

        # User priming message
        messages.append({
            "role": "user",
            "parts": [
                {"text": self._build_priming_message(ucs)},
            ],
        })

        # Model acknowledgment (Gemini uses "model" not "assistant")
        model_parts: list[dict[str, Any]] = [
            {"text": self._build_continuation_message(ucs)},
        ]

        # Include artifacts
        for artifact in ucs.artifacts[:5]:
            part = self._format_artifact_part(artifact)
            if part:
                model_parts.append(part)

        messages.append({
            "role": "model",
            "parts": model_parts,
        })

        total_text = system_prompt + "".join(
            str(m.get("parts", "")) for m in messages
        )
        token_estimate = self._estimate_tokens(total_text)

        return AdaptedOutput(
            format_name="gemini",
            messages=messages,
            system_prompt=system_prompt,
            metadata={
                "model_suggestion": self._suggest_model(token_estimate),
                "source_llm": ucs.session_meta.source_llm.value,
                "entity_count": len(ucs.entities),
                "summary_count": len(ucs.summaries),
                "api_format": "generateContent",
                "system_instruction": {
                    "parts": [{"text": system_prompt}],
                },
            },
            token_estimate=token_estimate,
        )

    def _build_system_instruction(
        self,
        ucs: UniversalContextSchema,
        context_summary: str,
    ) -> str:
        """Build Gemini's systemInstruction field."""
        parts = [
            "You are continuing a conversation started with a different AI assistant.",
            f"The original conversation had {ucs.session_meta.message_count} messages.",
            "",
            context_summary,
            "",
            "Continue the conversation naturally. Respect all prior decisions. "
            "Do not repeat established information.",
        ]
        return "\n".join(parts)

    @staticmethod
    def _build_priming_message(ucs: UniversalContextSchema) -> str:
        """Build the user priming message."""
        from app.core.models.ucs import TaskStatus
        active_tasks = [t for t in ucs.tasks if t.status == TaskStatus.ACTIVE]

        if active_tasks:
            return (
                "I'm continuing a conversation from another AI. "
                f"We have {len(active_tasks)} open task(s). "
                "Please continue where we left off."
            )
        return (
            "I'm continuing a conversation from another AI. "
            "Please continue where we left off."
        )

    @staticmethod
    def _build_continuation_message(ucs: UniversalContextSchema) -> str:
        """Build the model continuation message."""
        from app.core.models.ucs import TaskStatus
        active_tasks = [t for t in ucs.tasks if t.status == TaskStatus.ACTIVE]

        if active_tasks:
            task_list = ", ".join(t.description[:80] for t in active_tasks[:3])
            return (
                f"I have the full context. Open tasks: {task_list}. "
                f"What would you like to focus on?"
            )
        return "I have the full context from your previous conversation. How can I help?"

    @staticmethod
    def _format_artifact_part(artifact: Any) -> dict[str, Any] | None:
        """Format an artifact as a Gemini text part."""
        if not artifact.content:
            return None
        title = artifact.title or artifact.type.value
        if artifact.language:
            return {"text": f"Previous {title}:\n```{artifact.language}\n{artifact.content}\n```"}
        return {"text": f"Previous {title}:\n{artifact.content}"}

    @staticmethod
    def _suggest_model(token_estimate: int) -> str:
        """Suggest a Gemini model based on context size."""
        if token_estimate > 100000:
            return "gemini-1.5-pro"
        if token_estimate > 16000:
            return "gemini-1.5-pro"
        return "gemini-1.5-flash"
