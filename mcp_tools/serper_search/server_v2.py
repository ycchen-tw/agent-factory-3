"""Serper Search MCP Server v2.0

A clean, modular MCP server for Google Search via Serper API.

Features:
- 7 search endpoints (web, news, images, videos, places, autocomplete, scrape)
- Flexible transport support (stdio, http)
- Pydantic-based configuration and validation
- Improved error handling with ToolError
- Optional tokenizer for webpage scraping
"""
import os
import ssl
import asyncio
from typing import Optional, Literal
from functools import lru_cache

import certifi
import aiohttp
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

class Config(BaseSettings):
    """Server configuration with environment variable support."""

    # API Configuration
    serper_api_key: str = Field(..., validation_alias="SERPER_API_KEY")

    # Request Settings
    aiohttp_timeout: int = Field(default=15, validation_alias="AIOHTTP_TIMEOUT")
    max_web_tokens: int = Field(default=4000, validation_alias="MAX_WEB_TOKENS")

    # Tokenizer Settings (optional, only needed for scraping)
    tokenizer_path: str = Field(
        default="Qwen/Qwen2.5-0.5B",
        validation_alias="TOKENIZER_PATH"
    )
    enable_tokenizer: bool = Field(default=True, validation_alias="ENABLE_TOKENIZER")

    class Config:
        env_file = ".env"
        extra = "ignore"


# Global config instance
config = Config()


# -----------------------------------------------------------------------
# Tokenizer (Lazy Loading)
# -----------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_tokenizer():
    """Lazy load tokenizer (only when needed for scraping)."""
    if not config.enable_tokenizer:
        return None

    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(config.tokenizer_path)
    except Exception as e:
        print(f"Warning: Failed to load tokenizer: {e}")
        print("Scraping will return full text without truncation.")
        return None


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to specified number of tokens."""
    tokenizer = get_tokenizer()
    if tokenizer is None:
        # Fallback: simple character-based truncation
        return text[:max_tokens * 4]  # Rough estimate: 1 token ≈ 4 chars

    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text

    return tokenizer.decode(tokens[:max_tokens])


# -----------------------------------------------------------------------
# Request Models
# -----------------------------------------------------------------------

class BaseSearchRequest(BaseModel):
    """Base request model with common search parameters."""
    q: str = Field(..., description="Search query")
    gl: Optional[str] = Field(None, description="Country code (e.g., us, uk, ca)")
    hl: Optional[str] = Field(None, description="Language code (e.g., en, es, fr)")


class SearchRequest(BaseSearchRequest):
    """Web search request parameters."""
    num: int = Field(10, ge=1, le=100, description="Number of results (max 100)")
    page: Optional[int] = Field(1, ge=1, description="Page number (starting from 1)")
    tbs: Optional[str] = Field(None, description="Time filter (qdr:h, qdr:d, qdr:w, qdr:m, qdr:y)")
    autocorrect: Optional[bool] = Field(True, description="Enable autocorrect")


class NewsRequest(BaseSearchRequest):
    """News search request parameters."""
    num: int = Field(10, ge=1, le=100, description="Number of results")
    tbs: Optional[str] = Field(None, description="Time filter (qdr:h, qdr:d, qdr:w, qdr:m, qdr:y)")


class ImagesRequest(BaseSearchRequest):
    """Image search request parameters."""
    num: int = Field(10, ge=1, le=100, description="Number of results")


class VideosRequest(BaseSearchRequest):
    """Video search request parameters."""
    num: int = Field(10, ge=1, le=100, description="Number of results")


class PlacesRequest(BaseSearchRequest):
    """Places/Maps search request parameters."""
    pass


class AutocompleteRequest(BaseModel):
    """Autocomplete request parameters."""
    q: str = Field(..., description="Query prefix for autocomplete")


class ScrapeRequest(BaseModel):
    """Webpage scraping request parameters."""
    url: str = Field(..., description="URL of the webpage to scrape")
    include_markdown: Optional[bool] = Field(
        False,
        description="Include markdown formatting"
    )


# -----------------------------------------------------------------------
# Serper API Client
# -----------------------------------------------------------------------

class SerperClient:
    """HTTP client for Serper API with error handling."""

    BASE_URL = "https://google.serper.dev"
    SCRAPE_URL = "https://scrape.serper.dev"

    def __init__(self, api_key: str, timeout: int = 15):
        self.api_key = api_key
        self.timeout = timeout
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())

    async def _make_request(self, endpoint: str, payload: dict) -> dict:
        """Make HTTP POST request to Serper API."""
        url = f"{self.BASE_URL}/{endpoint}" if endpoint != "scrape" else self.SCRAPE_URL

        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }

        connector = aiohttp.TCPConnector(ssl=self._ssl_context)
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        try:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 401:
                        raise ToolError("Invalid Serper API key. Please check SERPER_API_KEY.")
                    elif response.status == 429:
                        raise ToolError("Rate limit exceeded. Please try again later.")
                    elif response.status >= 500:
                        raise ToolError(f"Serper API server error (status {response.status}).")

                    response.raise_for_status()
                    return await response.json()

        except aiohttp.ClientError as e:
            raise ToolError(f"Network request failed: {str(e)}")
        except asyncio.TimeoutError:
            raise ToolError(f"Request timed out after {self.timeout} seconds")

    async def search(self, request: SearchRequest) -> dict:
        """Perform web search."""
        payload = request.model_dump(exclude_none=True)
        return await self._make_request("search", payload)

    async def search_news(self, request: NewsRequest) -> dict:
        """Perform news search."""
        payload = request.model_dump(exclude_none=True)
        return await self._make_request("news", payload)

    async def search_images(self, request: ImagesRequest) -> dict:
        """Perform image search."""
        payload = request.model_dump(exclude_none=True)
        return await self._make_request("images", payload)

    async def search_videos(self, request: VideosRequest) -> dict:
        """Perform video search."""
        payload = request.model_dump(exclude_none=True)
        return await self._make_request("videos", payload)

    async def search_places(self, request: PlacesRequest) -> dict:
        """Perform places/maps search."""
        payload = request.model_dump(exclude_none=True)
        return await self._make_request("places", payload)

    async def autocomplete(self, request: AutocompleteRequest) -> dict:
        """Get autocomplete suggestions."""
        payload = request.model_dump(exclude_none=True)
        return await self._make_request("autocomplete", payload)

    async def scrape(self, request: ScrapeRequest) -> dict:
        """Scrape webpage content."""
        payload = request.model_dump(exclude_none=True)
        return await self._make_request("scrape", payload)


# -----------------------------------------------------------------------
# FastMCP Server
# -----------------------------------------------------------------------

mcp = FastMCP(
    name="Serper Search Server v2",
    instructions=(
        "This server provides comprehensive Google Search capabilities through Serper API. "
        "Supports web, news, images, videos, places search, autocomplete, and webpage scraping."
    )
)

# Initialize Serper client
client = SerperClient(api_key=config.serper_api_key, timeout=config.aiohttp_timeout)


# -----------------------------------------------------------------------
# Search Tools
# -----------------------------------------------------------------------

@mcp.tool
async def search(
    q: str = Field(..., description="Search query"),
    num: int = Field(10, ge=1, le=100, description="Number of results"),
    gl: Optional[str] = Field(None, description="Country code (us, uk, ca, etc.)"),
    hl: Optional[str] = Field(None, description="Language code (en, es, fr, etc.)"),
    page: Optional[int] = Field(1, ge=1, description="Page number"),
    tbs: Optional[str] = Field(None, description="Time filter (qdr:h=hour, qdr:d=day, qdr:w=week, qdr:m=month, qdr:y=year)"),
) -> dict:
    """Search Google for web results with advanced filtering options."""
    request = SearchRequest(q=q, num=num, gl=gl, hl=hl, page=page, tbs=tbs)
    return await client.search(request)


@mcp.tool
async def search_news(
    q: str = Field(..., description="News search query"),
    num: int = Field(10, ge=1, le=100, description="Number of results"),
    gl: Optional[str] = Field(None, description="Country code"),
    hl: Optional[str] = Field(None, description="Language code"),
    tbs: Optional[str] = Field(None, description="Time filter (e.g., qdr:d for last day)"),
) -> dict:
    """Search Google News for recent articles."""
    request = NewsRequest(q=q, num=num, gl=gl, hl=hl, tbs=tbs)
    return await client.search_news(request)


@mcp.tool
async def search_images(
    q: str = Field(..., description="Image search query"),
    num: int = Field(10, ge=1, le=100, description="Number of results"),
    gl: Optional[str] = Field(None, description="Country code"),
) -> dict:
    """Search Google Images."""
    request = ImagesRequest(q=q, num=num, gl=gl)
    return await client.search_images(request)


@mcp.tool
async def search_videos(
    q: str = Field(..., description="Video search query"),
    num: int = Field(10, ge=1, le=100, description="Number of results"),
    gl: Optional[str] = Field(None, description="Country code"),
) -> dict:
    """Search for videos."""
    request = VideosRequest(q=q, num=num, gl=gl)
    return await client.search_videos(request)


@mcp.tool
async def search_places(
    q: str = Field(..., description="Place/business name or location query"),
    gl: Optional[str] = Field(None, description="Country code"),
) -> dict:
    """Search Google Maps/Places for locations and businesses."""
    request = PlacesRequest(q=q, gl=gl)
    return await client.search_places(request)


@mcp.tool
async def autocomplete(
    q: str = Field(..., description="Query prefix to get suggestions for"),
) -> dict:
    """Get Google search autocomplete suggestions."""
    request = AutocompleteRequest(q=q)
    return await client.autocomplete(request)


@mcp.tool
async def scrape_webpage(
    url: str = Field(..., description="URL of the webpage to scrape"),
    include_markdown: Optional[bool] = Field(False, description="Include markdown formatting"),
) -> str:
    """Scrape and extract text content from a webpage.

    Returns the webpage text content, truncated to avoid token limits.
    """
    request = ScrapeRequest(url=url, include_markdown=include_markdown)
    result = await client.scrape(request)

    # Extract text and truncate to max tokens
    text = result.get('text', '')
    if not text:
        raise ToolError("No text content extracted from webpage")

    return truncate_to_tokens(text, config.max_web_tokens)


# -----------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------

def main():
    """Main entry point with flexible CLI options."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Serper Search MCP Server v2.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Transport options
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport protocol"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for HTTP transport"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Log level"
    )

    args = parser.parse_args()

    # Run server with specified transport
    if args.transport == "stdio":
        mcp.run(transport="stdio", log_level=args.log_level)
    else:
        mcp.run(
            transport="http",
            host=args.host,
            port=args.port,
            log_level=args.log_level
        )


if __name__ == "__main__":
    main()
