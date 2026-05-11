"""HTML application generator for the Conversation Visualizer."""

import html
import json

from .asset_loader import AssetLoader
from .core.types import ConversationGroup, GroupData, RolloutViewerData, ViewerData
from .renderers.message import MessageRenderer


class ConversationApp:
    """Generates a complete HTML application for viewing conversations."""

    # Inline styles (minimal, no external CSS)
    INLINE_STYLES = """
    <style>
        body { font-family: system-ui, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
        .container { max-width: 1000px; margin: 0 auto; }
        .conversation { max-width: 900px; margin: 0 auto 20px; }
        .message { margin: 12px 0; padding: 12px; border-radius: 8px; border-left: 4px solid; }
        .message-system { background: #2d2d44; border-color: #8b5cf6; }
        .message-developer { background: #2d3748; border-color: #3182ce; }
        .message-user { background: #1e3a5f; border-color: #38bdf8; }
        .message-assistant { background: #1e3e3e; border-color: #10b981; }
        .message-tool { background: #3d2e1e; border-color: #f59e0b; }
        .message-header { font-size: 0.85em; margin-bottom: 8px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
        .author { font-weight: bold; text-transform: uppercase; }
        .badge { font-size: 0.75em; padding: 2px 6px; border-radius: 4px; }
        .badge-recipient { background: #4a3f6b; color: #c4b5fd; }
        .badge-channel { background: #3b4a3b; color: #86efac; }
        .message-body { line-height: 1.6; }
        .content-text { white-space: pre-wrap; word-wrap: break-word; }
        .content-system, .content-developer { font-size: 0.9em; }
        .config-row { margin: 4px 0; }
        .config-key { color: #a5b4fc; font-weight: 500; }
        pre { background: #0d0d1a; padding: 8px; border-radius: 4px; overflow-x: auto; margin: 4px 0; font-size: 0.85em; white-space: pre-wrap; word-wrap: break-word; }
        .group { margin-bottom: 40px; }
        .group-header { border-bottom: 1px solid #444; padding-bottom: 8px; margin-bottom: 16px; }
        .group-header h2 { margin: 0; color: #a5b4fc; }
        .nav { position: sticky; top: 0; background: #1a1a2e; padding: 10px 0; border-bottom: 1px solid #333; margin-bottom: 20px; z-index: 100; }
        .nav a { color: #60a5fa; margin-right: 16px; text-decoration: none; }
        .nav a:hover { text-decoration: underline; }
        h1 { text-align: center; color: #c4b5fd; margin-bottom: 30px; }
        h3 { color: #9ca3af; margin: 20px 0 10px; font-size: 1em; }
        /* Collapsible Tools */
        .tools-details { margin: 4px 0; }
        .tools-details summary { list-style: none; cursor: pointer; }
        .tools-details summary::-webkit-details-marker { display: none; }
        .tools-details summary::before { content: '\\25B6 '; font-size: 0.8em; }
        .tools-details[open] summary::before { content: '\\25BC '; }
        .tools-details pre { margin-top: 8px; max-height: 400px; overflow-y: auto; }
    </style>
    """

    def __init__(self, data: ViewerData):
        self.data = data
        self.message_renderer = MessageRenderer()

    def generate(self) -> str:
        """Generate the complete HTML document."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(self.data.title)}</title>
    {self.INLINE_STYLES}
</head>
<body>
    <div class="container">
        <h1>{html.escape(self.data.title)}</h1>
        {self._render_navigation()}
        {self._render_groups()}
    </div>
</body>
</html>"""

    def _render_navigation(self) -> str:
        """Render navigation links for multiple groups."""
        if len(self.data.groups) <= 1:
            return ""
        links = [
            f'<a href="#group-{html.escape(g.group_id)}">{html.escape(g.group_id)}</a>'
            for g in self.data.groups
        ]
        return f'<nav class="nav">{"".join(links)}</nav>'

    def _render_groups(self) -> str:
        """Render all conversation groups."""
        return "\n".join(self._render_group(g) for g in self.data.groups)

    def _render_group(self, group: ConversationGroup) -> str:
        """Render a single conversation group."""
        conversations_html = []
        for i, conv in enumerate(group.conversations):
            title = conv.title or f"Conversation {i + 1}"
            messages_html = "\n".join(
                self.message_renderer.render(m) for m in conv.messages
            )
            conv_id = conv.id or str(i)
            conversations_html.append(
                f"""
<div class="conversation" id="conv-{html.escape(conv_id)}">
    <h3>{html.escape(title)}</h3>
    {messages_html}
</div>"""
            )

        # Hide group header if only one default group
        show_header = not (
            len(self.data.groups) == 1 and group.group_id == "default"
        )
        header_html = ""
        if show_header:
            header_html = f"""
    <div class="group-header">
        <h2>{html.escape(group.group_id)}</h2>
    </div>"""

        return f"""
<div class="group" id="group-{html.escape(group.group_id)}">
    {header_html}
    {"".join(conversations_html)}
</div>"""


class RolloutViewerApp:
    """Generates an HTML application for viewing rollout results with groups.

    Features:
    - Resizable sidebar with drag handle
    - Group cards with collapsible rollout lists
    - Glass morphism visual design
    - Keyboard navigation support
    """

    def __init__(self, data: RolloutViewerData):
        self.data = data
        self.message_renderer = MessageRenderer()

    def generate(self) -> str:
        """Generate the complete HTML document."""
        styles = AssetLoader.get_rollout_viewer_styles()
        scripts = AssetLoader.get_rollout_viewer_scripts()
        conversations_data = self._build_conversations_data()

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(self.data.title)}</title>
    <style>{styles}</style>
</head>
<body>
    <div class="rollout-app">
        <div class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <h1>{html.escape(self.data.title)}</h1>
            </div>
            <div class="sidebar-content">
                {self._render_groups_list()}
            </div>
            <div class="sidebar-footer">
                {self._render_summary_stats()}
            </div>
        </div>
        <div class="resize-handle" id="resizeHandle"></div>
        <div class="main-content">
            <div class="main-header" id="mainHeader">
                <h2>Select a rollout</h2>
                <div class="rollout-info"></div>
            </div>
            <div class="info-panels" id="infoPanels"></div>
            <div class="conversation-container" id="conversationContainer">
                <div class="empty-state">
                    <div class="empty-state-icon">&#128196;</div>
                    <div>Click a rollout to view conversation</div>
                </div>
            </div>
        </div>
    </div>
    <script>
        window.conversationsData = {json.dumps(conversations_data)};
    </script>
    <script>
        {scripts}
    </script>
</body>
</html>"""

    def _build_conversations_data(self) -> dict:
        """Pre-render all conversations to a dict for JavaScript access."""
        data = {}
        for group in self.data.groups:
            for rollout in group.rollouts:
                conv_html = ""
                if rollout.conversation:
                    conv_html = "".join(
                        self.message_renderer.render(m)
                        for m in rollout.conversation.messages
                    )
                data[rollout.rollout_id] = {
                    "html": conv_html,
                    "rollout_id": rollout.rollout_id,
                    "success": rollout.success,
                    "weighted_reward": rollout.weighted_reward,
                    "raw_advantage": rollout.raw_advantage,
                    "advantage": rollout.advantage,
                    "num_rounds": rollout.num_rounds,
                    "completion_tokens": rollout.completion_tokens,
                    "elapsed_time": rollout.elapsed_time,
                    "end_reason": rollout.end_reason,
                    "weight_versions": rollout.weight_versions,
                    "error": rollout.error,
                    "reward_components": rollout.reward_components,
                    "traceback": rollout.traceback,
                    "trainable": rollout.trainable,
                    "skip_reason": rollout.skip_reason,
                    "config_snapshot": rollout.config_snapshot,
                }
        return data

    def _render_groups_list(self) -> str:
        """Render the sidebar groups list."""
        groups_html = []
        for i, group in enumerate(self.data.groups):
            expanded = "expanded" if i == 0 else ""
            filtered_class = "filtered" if group.is_filtered else ""

            # Group badges
            badges_html = ""
            if group.filter_reason:
                badges_html += f'<span class="stat-badge filter">{html.escape(group.filter_reason)}</span>'
            elif group.reward_baseline is not None:
                badges_html += f'<span class="stat-badge baseline">baseline: {group.reward_baseline:.2f}</span>'

            # Rollout items
            rollouts_html = []
            for rollout in group.rollouts:
                status_class = "success" if rollout.success else "failed"
                if not rollout.trainable:
                    status_class += " not-trainable"

                # Stat badges
                stats = []
                if rollout.weighted_reward is not None:
                    stats.append(f'<span class="stat-badge reward">{rollout.weighted_reward:.2f}</span>')
                if rollout.advantage is not None:
                    sign = "+" if rollout.advantage >= 0 else ""
                    adv_class = "positive" if rollout.advantage >= 0 else "negative"
                    stats.append(f'<span class="stat-badge advantage {adv_class}">{sign}{rollout.advantage:.2f}</span>')
                if not rollout.trainable:
                    reason_text = rollout.skip_reason or "skipped"
                    stats.append(f'<span class="stat-badge skip">{html.escape(reason_text)}</span>')

                # Data attributes for JS interaction
                rollout_id_short = rollout.rollout_id.split("_")[-1] if "_" in rollout.rollout_id else rollout.rollout_id
                data_attrs = f'data-rollout-id="{html.escape(rollout.rollout_id)}"'
                data_attrs += f' data-success="{str(rollout.success).lower()}"'
                if rollout.weighted_reward is not None:
                    data_attrs += f' data-reward="{rollout.weighted_reward}"'
                if rollout.advantage is not None:
                    data_attrs += f' data-advantage="{rollout.advantage}"'
                if rollout.num_rounds is not None:
                    data_attrs += f' data-rounds="{rollout.num_rounds}"'
                if rollout.completion_tokens is not None:
                    data_attrs += f' data-tokens="{rollout.completion_tokens}"'
                if rollout.elapsed_time is not None:
                    data_attrs += f' data-time="{rollout.elapsed_time}"'

                rollouts_html.append(f'''
                <div class="rollout-item {status_class}" {data_attrs}>
                    <span class="rollout-id">{html.escape(rollout_id_short)}</span>
                    <div class="rollout-stats">{"".join(stats)}</div>
                </div>''')

            groups_html.append(f'''
            <div class="group-card {filtered_class} {expanded}">
                <div class="group-header">
                    <div class="group-header-left">
                        <span class="group-title">{html.escape(group.group_id)}</span>
                        <span class="group-meta">({group.group_size})</span>
                    </div>
                    <div class="group-badges">
                        {badges_html}
                        <span class="group-chevron">&#9654;</span>
                    </div>
                </div>
                <div class="group-content">
                    {"".join(rollouts_html)}
                </div>
            </div>''')

        return "".join(groups_html)

    def _render_summary_stats(self) -> str:
        """Render summary statistics."""
        total_groups = len(self.data.groups)
        filtered_groups = sum(1 for g in self.data.groups if g.is_filtered)
        retained_groups = total_groups - filtered_groups
        total_rollouts = sum(g.group_size for g in self.data.groups)

        return f"Groups: {retained_groups}/{total_groups} | Rollouts: {total_rollouts}"
