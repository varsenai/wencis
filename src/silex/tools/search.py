"""
Web Search Tool using DuckDuckGo.
"""

from __future__ import annotations

from ddgs import DDGS
from silex.tools.base import BaseTool
from silex.memory.vector_store import VectorStore
from silex.utils.logger import setup_logger

log = setup_logger("silex.tools.search")

class WebSearchTool(BaseTool):
    name = "web_search"
    risk_level = "network"
    description = "Searches the live internet for current facts, news, or general knowledge."
    schema = {
        "query": "string (the exact search query to execute)",
        "max_results": "integer (optional, default 3, max 5)"
    }

    async def execute(self, **kwargs) -> str:
        query = kwargs.get("query")
        if not query:
            return "Error: 'query' argument is required."

        # Sanitize: cap query length to prevent abuse
        query = str(query)[:200]

        max_results = min(int(kwargs.get("max_results", 3)), 5)
        log.info(f"Executing web_search for: '{query}'")

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                
            if not results:
                return f"No results found for query: '{query}'"
                
            formatted = f"Search Results for '{query}':\n\n"
            for i, r in enumerate(results, 1):
                formatted += f"[{i}] {r.get('title', 'No Title')}\n"
                formatted += f"URL: {r.get('href', 'No URL')}\n"
                formatted += f"Snippet: {r.get('body', 'No Snippet')}\n\n"
                
            return formatted.strip()

        except Exception as e:
            log.error(f"Web search failed: {e}")
            return f"Error executing search: {str(e)}"

class SemanticSearchTool(BaseTool):
    name = "semantic_search"
    risk_level = "read_only"
    description = "Search your local workspace using natural language. Useful for finding code patterns, related files, or old memories."
    schema = {
        "query": "string (the semantic query)",
        "n_results": "integer (optional, default 5, max 10)"
    }

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    async def execute(self, **kwargs) -> str:
        query = kwargs.get("query")
        if not query:
            return "Error: 'query' argument is required."

        n_results = min(int(kwargs.get("n_results", 5)), 10)
        log.info(f"Executing semantic_search for: '{query}'")

        if not getattr(self.vs, "is_active", False):
            return (
                "Semantic search is disabled: vector memory (ChromaDB) is not installed. "
                'Install with pip install "openyfai-vyn[vector]" or `pip install chromadb`, then restart VYN.'
            )

        try:
            results = self.vs.search(query, n_results=n_results)
            
            if not results:
                return "No semantic matches found in the local workspace."

            formatted = f"Semantic Matches for '{query}':\n\n"
            for r in results:
                path = r['metadata'].get('path', 'Unknown')
                formatted += f"FILE: {path}\n"
                formatted += f"CONTENT: {r['content'][:300]}...\n"
                formatted += "---"
                
            return formatted.strip()

        except Exception as e:
            log.error(f"Semantic search failed: {e}")
            return f"Error executing semantic search: {str(e)}"
