"""Rollout Visualizer — generates self-contained HTML for rollout analysis.

Usage:
    from agent_factory_3.visualizer import create_groups_viewer

    html = create_groups_viewer(group_results, title="Batch #1")
    with open("rollouts.html", "w") as f:
        f.write(html)
"""

from typing import List

from ..orchestrator.types import GroupResult
from .adapters.group_results import GroupResultsAdapter
from .app import RolloutViewerApp
from .core.types import RolloutViewerData

__all__ = ["create_groups_viewer"]


def create_groups_viewer(
    group_results: List[GroupResult],
    *,
    title: str = "Rollout Groups",
) -> str:
    """Create a rollout viewer HTML from GroupResults.

    Args:
        group_results: Completed groups from Orchestrator.
        title: Page title.

    Returns:
        Complete self-contained HTML string.
    """
    adapter = GroupResultsAdapter()
    groups = adapter.adapt(group_results)
    data = RolloutViewerData(groups=groups, title=title)
    return RolloutViewerApp(data).generate()
