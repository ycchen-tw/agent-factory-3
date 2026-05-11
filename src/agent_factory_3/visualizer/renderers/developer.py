"""Developer content renderer."""

import html
import json

from ..core.types import DeveloperPart


class DeveloperRenderer:
    """Renders DeveloperPart to HTML."""

    def can_render(self, part) -> bool:
        """Check if this renderer can handle the given part."""
        return getattr(part, "type", None) == "developer"

    def render(self, part: DeveloperPart) -> str:
        """Render DeveloperPart to HTML string."""
        rows = []

        if part.instructions:
            rows.append(
                f'<div class="config-row">'
                f'<span class="config-key">Instructions:</span>'
                f"<pre>{html.escape(part.instructions)}</pre>"
                f"</div>"
            )

        if part.tools:
            rows.append(self._render_tools_collapsible(part.tools))

        return f'<div class="content-developer">{"".join(rows)}</div>'

    def _render_tools_collapsible(self, tools: dict) -> str:
        """Render tools configuration as collapsible section."""
        tools_json = json.dumps(tools, indent=2, ensure_ascii=False)
        return f"""<details class="tools-details">
    <summary class="config-key">Tools ({len(tools)} namespaces)</summary>
    <pre>{html.escape(tools_json)}</pre>
</details>"""
