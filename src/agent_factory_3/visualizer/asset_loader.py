"""Asset loader for CSS and JavaScript files.

This module provides a centralized way to load and cache static assets
(CSS, JavaScript) for the visualizer applications.
"""

from pathlib import Path
from typing import ClassVar

ASSETS_DIR = Path(__file__).parent / "assets"


class AssetLoader:
    """Loads and caches CSS/JS assets from the assets directory.

    This class provides class methods to load static files and caches them
    in memory to avoid repeated disk reads.

    Example:
        # Load individual files
        css = AssetLoader.load_css("variables.css", "base.css")

        # Load preset bundles
        styles = AssetLoader.get_rollout_viewer_styles()
        scripts = AssetLoader.get_rollout_viewer_scripts()
    """

    _cache: ClassVar[dict[str, str]] = {}

    @classmethod
    def _load_file(cls, path: Path) -> str:
        """Load a file from disk, using cache if available."""
        cache_key = str(path)
        if cache_key not in cls._cache:
            cls._cache[cache_key] = path.read_text(encoding="utf-8")
        return cls._cache[cache_key]

    @classmethod
    def load_css(cls, *filenames: str) -> str:
        """Load and concatenate CSS files from assets/styles/.

        Args:
            *filenames: CSS filenames to load (e.g., "variables.css", "base.css")

        Returns:
            Concatenated CSS content
        """
        parts = []
        for filename in filenames:
            path = ASSETS_DIR / "styles" / filename
            parts.append(cls._load_file(path))
        return "\n".join(parts)

    @classmethod
    def load_js(cls, *filenames: str) -> str:
        """Load and concatenate JavaScript files from assets/scripts/.

        Args:
            *filenames: JS filenames to load (e.g., "rollout-viewer.js")

        Returns:
            Concatenated JavaScript content
        """
        parts = []
        for filename in filenames:
            path = ASSETS_DIR / "scripts" / filename
            parts.append(cls._load_file(path))
        return "\n".join(parts)

    @classmethod
    def get_rollout_viewer_styles(cls) -> str:
        """Get all styles for RolloutViewerApp.

        Returns:
            Concatenated CSS for the rollout viewer application
        """
        return cls.load_css(
            "variables.css",
            "base.css",
            "components.css",
            "animations.css",
            "rollout-viewer.css",
        )

    @classmethod
    def get_rollout_viewer_scripts(cls) -> str:
        """Get all scripts for RolloutViewerApp.

        Returns:
            Concatenated JavaScript for the rollout viewer application
        """
        return cls.load_js("rollout-viewer.js")

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the asset cache.

        Useful for development/testing when assets are being modified.
        """
        cls._cache.clear()
