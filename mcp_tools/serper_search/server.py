import os
import ssl
import asyncio
import json
from typing import Dict, Any, Optional

import certifi
import aiohttp
from pydantic import BaseModel, Field
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from transformers import AutoTokenizer

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
AIOHTTP_TIMEOUT = int(os.getenv("AIOHTTP_TIMEOUT", "15"))
MAX_WEB_NUM_TOKENS = int(os.getenv("MAX_WEB_NUM_TOKENS", "4000"))

if not SERPER_API_KEY:
    raise ValueError("SERPER_API_KEY environment variable is required")

# Global tokenizer variable to be initialized in main()
TOKENIZER = None

def initialize_tokenizer(tokenizer_path: str):
    """Initialize the global tokenizer with the specified path."""
    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path)

def truncate_to_token_length(text: str, num_tokens: int) -> str:
    """Truncate text to specified number of tokens."""
    if TOKENIZER is None:
        raise RuntimeError("Tokenizer not initialized")
    return TOKENIZER.decode(TOKENIZER.encode(text)[:num_tokens])

# -----------------------------------------------------------------------
# Request Schemas
# -----------------------------------------------------------------------

class BaseRequest(BaseModel):
    q: str = Field(..., description="The query to search for")
    gl: Optional[str] = Field(None, description="The country to search in, e.g. us, uk, ca, au, etc.")
    location: Optional[str] = Field(None, description="The location to search in, e.g. San Francisco, CA, USA")
    hl: Optional[str] = Field(None, description="The language to search in, e.g. en, es, fr, de, etc.")
    

class SearchRequest(BaseRequest):
    tbs: Optional[str] = Field(None, description="The time period to search in, e.g. d, w, m, y")
    num: int = Field(10, le=100, description="The number of results to return, max is 100")
    page: Optional[int] = Field(1, ge=1, description="The page number to return, first page is 1")


class WebpageRequest(BaseModel):
    url: str = Field(..., description="The URL of the webpage to scrape")
    includeMarkdown: Optional[bool] = Field(False, description="Whether to include markdown formatting")

# -----------------------------------------------------------------------
# HTTP Client Utilities
# -----------------------------------------------------------------------

async def make_serper_request(endpoint: str, request_data: BaseModel) -> Dict[str, Any]:
    """Make a request to the Serper API."""
    payload = request_data.model_dump(exclude_none=True)
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)

    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.post(endpoint, headers=headers, json=payload) as response:
                response.raise_for_status()
                return await response.json()
    except aiohttp.ClientError as e:
        raise ToolError(f"API request failed: {str(e)}")
    except asyncio.TimeoutError:
        raise ToolError(f"Request timed out after {AIOHTTP_TIMEOUT} seconds")

# -----------------------------------------------------------------------
# FastMCP Server
# -----------------------------------------------------------------------

mcp = FastMCP(
    name="Serper Search Server",
    instructions="This server provides access to Google Search through the Serper API. "
                "It supports web search, places search, and webpage scraping."
)

# -----------------------------------------------------------------------
# Search Tools
# -----------------------------------------------------------------------

@mcp.tool
async def search(
    q: str = Field(..., description="The search query"),
    gl: Optional[str] = Field(None, description="Country code (us, uk, ca, etc.)"),
    page: Optional[int] = Field(1, ge=1, description="Page number (starting from 1)"),
) -> dict:
    """Search Google for web results."""
    request = SearchRequest(q=q, gl=gl, page=page, num=10)
    return await make_serper_request("https://google.serper.dev/search", request)


@mcp.tool
async def search_places(
    q: str = Field(..., description="The search query"),
    gl: Optional[str] = Field(None, description="Country code (us, uk, ca, etc.)"),
) -> dict:
    """Search Google Places."""
    request = BaseRequest(q=q, gl=gl)
    return await make_serper_request("https://google.serper.dev/places", request)


@mcp.tool
async def scrape_webpage(
    url: str = Field(..., description="URL of the webpage to scrape"),
    includeMarkdown: Optional[bool] = Field(False, description="Include markdown formatting in the output")
) -> dict:
    """Scrape and extract content from a webpage."""
    request = WebpageRequest(url=url, includeMarkdown=includeMarkdown)
    serper_result = await make_serper_request("https://scrape.serper.dev", request)
    web_text = serper_result['text']
    return truncate_to_token_length(web_text, MAX_WEB_NUM_TOKENS)

# -----------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------

def main():
    """Main entry point for the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Serper MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="streamable-http",
        help="Transport protocol to use (default: stdio)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to for HTTP transports (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to for HTTP transports (default: 8000)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Log level (default: INFO)"
    )
    parser.add_argument(
        "--tokenizer-path",
        default=os.getenv("TOKENIZER_PATH", "Qwen/Qwen3-0.6B"),
        help="Path to the tokenizer directory (default: Qwen/Qwen3-0.6B or TOKENIZER_PATH env var)"
    )
    
    args = parser.parse_args()
    
    # Initialize tokenizer with the specified path
    initialize_tokenizer(args.tokenizer_path)
    
    # Run server with specified transport and options
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
            log_level=args.log_level
        )


if __name__ == "__main__":
    main()