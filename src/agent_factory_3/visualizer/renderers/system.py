"""System content renderer."""

import html
import json

from ..core.types import SystemPart


class SystemRenderer:
    """Renders SystemPart to HTML."""

    def can_render(self, part) -> bool:
        """Check if this renderer can handle the given part."""
        return getattr(part, "type", None) == "system"

    def render(self, part: SystemPart) -> str:
        """Render SystemPart to HTML string."""
        rows = []

        if part.model_identity:
            rows.append(
                f'<div class="config-row">'
                f'<span class="config-key">Model Identity:</span> '
                f"{html.escape(part.model_identity)}"
                f"</div>"
            )

        if part.reasoning_effort:
            rows.append(
                f'<div class="config-row">'
                f'<span class="config-key">Reasoning Effort:</span> '
                f"{part.reasoning_effort}"
                f"</div>"
            )

        if part.conversation_start_date:
            rows.append(
                f'<div class="config-row">'
                f'<span class="config-key">Start Date:</span> '
                f"{part.conversation_start_date}"
                f"</div>"
            )

        if part.knowledge_cutoff:
            rows.append(
                f'<div class="config-row">'
                f'<span class="config-key">Knowledge Cutoff:</span> '
                f"{part.knowledge_cutoff}"
                f"</div>"
            )

        if part.tools:
            rows.append(self._render_tools_collapsible(part.tools))

        return f'<div class="content-system">{"".join(rows)}</div>'

    def _render_tools_collapsible(self, tools: dict) -> str:
        """Render tools configuration as collapsible section."""
        tools_json = json.dumps(tools, indent=2, ensure_ascii=False)
        return f"""<details class="tools-details">
    <summary class="config-key">Tools ({len(tools)} namespaces)</summary>
    <pre>{html.escape(tools_json)}</pre>
</details>"""
