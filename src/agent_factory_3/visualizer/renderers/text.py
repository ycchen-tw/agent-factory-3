"""Text content renderer."""

import html

from ..core.types import TextPart


class TextRenderer:
    """Renders TextPart to HTML."""

    def can_render(self, part) -> bool:
        """Check if this renderer can handle the given part."""
        return getattr(part, "type", None) == "text"

    def render(self, part: TextPart) -> str:
        """Render TextPart to HTML string."""
        text = html.escape(part.text)
        text = text.replace("\n", "<br>")
        return f'<div class="content-text">{text}</div>'
