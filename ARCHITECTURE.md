# Silex Architecture & Internal Systems

This document serves as the primary technical context for any developer or AI Agent working inside the `silex` repository.

## 1. The Core Loop (`CognitiveLoop`)
Located in `src/silex/core/cognitive_loop.py`.

The `CognitiveLoop` is the heart of Silex. Unlike typical LLM chat applications, Silex does not just pass strings back and forth. It treats every user message as an "Event" that triggers a cascade of internal reasoning pipelines.

**The Pipeline:**
1. **Input Parsing**: The semantic parser translates the user's raw input into actionable intents.
2. **Context Assembly**: The `ContextBuilder` gathers active goals, relevant memories from ChromaDB, and active nodes from the Causal Graph.
3. **Drafting**: An LLM provider drafts a response.
4. **Critique**: The `ResponseCritic` evaluates the draft for Accuracy, Depth, and Honesty. If the score is too low, the engine automatically rejects the draft and initiates a "Debug/Rewrite" cycle.
5. **Output**: The final structured JSON is returned to the HTTP Daemon.

## 2. Multi-Tier Memory Store
Silex possesses localized, long-term memory that does not leak to cloud providers. 

* **SQLite (`Database`)**: Stores hard facts, active goals, and session provenance.
* **ChromaDB (`VectorStore`)**: Stores semantic embeddings of past turns. This allows Silex to recall a conversation from 3 months ago simply because the current topic is semantically similar.

## 3. The Causal Knowledge Graph (KG)
Located in `src/silex/world/graph.py`.

Silex maps the user's reality into a Causal Knowledge Graph.
* Instead of flat text, knowledge is stored as `Nodes` (e.g., `Node: User`) and `Edges` (e.g., `Edge: IS_WORKING_ON -> Node: Silex`).
* The graph utilizes **Recursive SQLite CTEs** (Common Table Expressions) to perform deep traversal. If you ask "Why did my server crash?", Silex can traverse the `CAUSED_BY` edges in the graph to find the root node without pulling the entire graph into memory.

## 4. The Native HTTP Daemon
Located in `src/silex/server/daemon.py`.

To maintain a tiny compiled executable footprint, Silex uses the standard library `http.server` instead of heavy frameworks like FastAPI.
* **Concurrency**: The daemon bridges synchronous HTTP requests to the background `asyncio` loop using `asyncio.run_coroutine_threadsafe`.
* **State Persistence**: Because it is a persistent daemon, the Database and ChromaDB connection pools stay warm in RAM, reducing the latency of cognitive turns from seconds to milliseconds compared to a one-shot CLI binary.

## 5. Adding New Features (Agent Guidelines)
If you are an AI Agent tasked with upgrading Silex, adhere to these rules:

1. **Zero Frontend Logic**: Silex must NEVER return ANSI colors, terminal tables, or Rich formatting. It must only return structured Pydantic models (JSON) through the `/cognitive` endpoint. The public VYN client handles all formatting.
2. **Maintain the PyInstaller Boundary**: Be extremely careful adding third-party dependencies to `pyproject.toml`. Every new dependency increases the size of `silex.exe` and increases the risk of PyInstaller failing to locate hidden imports.
3. **Daemon Stability**: If Silex crashes, the VYN client breaks. All operations inside `CognitiveLoop` must be wrapped in strong `try/except` blocks. In the event of a catastrophic error, return a gracefully degraded JSON payload with `type: "error"`.
