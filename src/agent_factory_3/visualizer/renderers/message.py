"""Message renderer that composes content renderers."""

import html
from typing import List

from ..core.types import ContentPart, Message, Role
from .developer import DeveloperRenderer
from .system import SystemRenderer
from .text import TextRenderer


class MessageRenderer:
    """Renders a complete Message to HTML by composing content renderers."""

    def __init__(self):
        self._renderers = [
            TextRenderer(),
            SystemRenderer(),
            DeveloperRenderer(),
        ]

    def render(self, message: Message) -> str:
        """Render a Message to HTML string."""
        role = message.author.role.value
        author_name = message.author.name or role.title()

        # Header badges
        badges = []
        if message.recipient:
            badges.append(
                f'<span class="badge badge-recipient">'
                f"&rarr; {html.escape(message.recipient)}"
                f"</span>"
            )
        if message.channel:
            badges.append(
                f'<span class="badge badge-channel">'
                f"[{html.escape(message.channel)}]"
                f"</span>"
            )

        # Tool stats badges (for tool messages with execution stats)
        if message.author.role == Role.TOOL and "tool_stats" in message.metadata:
            stats = message.metadata["tool_stats"]
            # Elapsed time
            elapsed = stats.get("elapsed")
            if elapsed is not None:
                badges.append(f'<span class="badge badge-time">{elapsed:.2f}s</span>')
            # Error status
            error = stats.get("error")
            if error:
                badges.append(f'<span class="badge badge-error">{html.escape(error)}</span>')
            # Early exit
            if stats.get("early_exit"):
                badges.append('<span class="badge badge-early-exit">early exit</span>')

        # Failure marker (set by _inject_failure_visibility for ERROR rollouts)
        error_kind = message.metadata.get("error_kind")
        error_class = ""
        error_panel = ""
        if error_kind:
            badges.append(
                f'<span class="badge badge-error">{html.escape(error_kind)}</span>'
            )
            error_class = " message-failed"
            err_msg = message.metadata.get("error_message", "")
            if err_msg:
                error_panel = (
                    '<div class="message-error-detail">'
                    f'<strong>{html.escape(error_kind)}:</strong> '
                    f'{html.escape(err_msg)}'
                    '</div>'
                )

        badges_html = " ".join(badges)

        # Content
        content_html = self._render_content(message.content)

        return f"""
<div class="message message-{role}{error_class}">
    <div class="message-header">
        <span class="author">{html.escape(author_name)}</span>
        {badges_html}
    </div>
    <div class="message-body">
        {error_panel}
        {content_html}
    </div>
</div>"""

    def _render_content(self, parts: List[ContentPart]) -> str:
        """Render all content parts."""
        html_parts = []
        for part in parts:
            rendered = False
            for renderer in self._renderers:
                if renderer.can_render(part):
                    html_parts.append(renderer.render(part))
                    rendered = True
                    break
            if not rendered:
                raise ValueError(
                    f"No renderer for content type: {getattr(part, 'type', type(part))}"
                )
        return "\n".join(html_parts)
