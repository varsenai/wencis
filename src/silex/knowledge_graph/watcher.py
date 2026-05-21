"""
Debounced File System Watcher (World Model V2)

Monitors the ARIA_WORKSPACE for changes to code files. Features a strict debounce
mechanism to prevent database storms during massive file operations (e.g., npm install)
and explicit deletion handling to prune the Knowledge Graph.
"""
from __future__ import annotations

import asyncio
import time
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from silex.utils.logger import setup_logger
from silex.utils.config import WORKSPACE_DIR

log = setup_logger("silex.world_model.watcher")


class WorkspaceEventHandler(FileSystemEventHandler):
    """Handles file events with filtering for ignored paths."""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        
        # V2 Constraint: Strict Ignore Lists
        self.ignored_dirs = {".git", "node_modules", ".venv", "__pycache__", "web_dist"}
        self.monitored_extensions = {".py", ".ts", ".tsx", ".js", ".jsx"}

    def _is_valid_file(self, path: str) -> bool:
        p = Path(path)
        
        # Check extensions
        if p.suffix not in self.monitored_extensions:
            return False
            
        # Check ignored directories in path parts
        for part in p.parts:
            if part in self.ignored_dirs or part.startswith("."):
                # Exclude hidden folders like .next, .aria, etc.
                return False
                
        return True

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory and self._is_valid_file(event.src_path):
            self.callback(event.src_path, "created")

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory and self._is_valid_file(event.src_path):
            self.callback(event.src_path, "modified")

    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory and self._is_valid_file(event.src_path):
            self.callback(event.src_path, "deleted")

    def on_moved(self, event: FileSystemEvent):
        if not event.is_directory:
            if self._is_valid_file(event.src_path):
                self.callback(event.src_path, "deleted")
            if self._is_valid_file(event.dest_path):
                self.callback(event.dest_path, "created")


class DebouncedWatcher:
    """Watches the workspace with a 15-second debounce."""

    def __init__(self, db, debounce_seconds: float = 15.0, knowledge_graph=None):
        self.db = db
        self.kg = knowledge_graph
        self.debounce_seconds = debounce_seconds
        self.pending_events: dict[str, str] = {}
        self.last_event_time = 0.0
        self.observer = None
        self._loop_task = None

    def _on_event(self, filepath: str, action: str):
        """Record the event and reset the debounce timer."""
        # If a file was created then modified, keep 'created' or 'modified'
        # If deleted, it overwrites 'modified' to 'deleted'
        self.pending_events[filepath] = action
        self.last_event_time = time.time()
        log.debug(f"FS Event tracked: {action} on {Path(filepath).name}")

    async def _process_batch(self):
        """Process all queued events after the debounce window closes."""
        from silex.knowledge_graph.mapper import SkeletonMapper
        
        events_to_process = self.pending_events.copy()
        self.pending_events.clear()
        
        log.info(f"Processing batch of {len(events_to_process)} FS events...")
        
        for filepath, action in events_to_process.items():
            path_obj = Path(filepath)
            
            if action == "deleted":
                # V2 Constraint: Explicit Deletion Handling
                log.info(f"Purging deleted file from World Model: {path_obj.name}")
                if self.kg:
                    await self.kg.remove_nodes_by_source(f"file://{filepath}")
                else:
                    # Fallback: direct SQL — must also clean edges to avoid FK orphans
                    source_uri = f"file://{filepath}"
                    orphan_rows = await self.db.fetch_all(
                        "SELECT id FROM knowledge_nodes WHERE source = ?", (source_uri,)
                    )
                    orphan_ids = [r["id"] for r in orphan_rows]
                    async with self.db.transaction():
                        await self.db.execute(
                            "DELETE FROM knowledge_nodes WHERE source = ?",
                            (source_uri,)
                        )
                        if orphan_ids:
                            ph = ",".join(["?"] * len(orphan_ids))
                            await self.db.execute(
                                f"DELETE FROM causal_edges WHERE source_node IN ({ph}) OR target_node IN ({ph})",
                                tuple(orphan_ids + orphan_ids)
                            )
                continue
                
            # For created/modified, parse the file
            meta = SkeletonMapper.map_file(path_obj)
            if not meta or "error" in meta:
                continue
                
            # Log the successful map
            log.info(f"Mapped {path_obj.name}: {len(meta.get('imports', []))} imports, {len(meta.get('functions', []))} functions.")
            
            # Write to Knowledge Graph
            source_uri = f"file://{filepath}"

            if self.kg:
                # Route through KG API to keep in-memory graph consistent
                await self.kg.remove_nodes_by_source(source_uri)
                from silex.models.schemas import KnowledgeNode
                node = KnowledgeNode(
                    id=f"file_node_{uuid.uuid4().hex[:8]}",
                    content=json.dumps(meta),
                    node_type="code_structure",
                    confidence=1.0,
                    source=source_uri,
                )
                await self.kg.add_node(node)
            else:
                # Fallback: direct SQL (legacy path) — clean edges + atomic upsert
                node_id = f"file_node_{uuid.uuid4().hex[:8]}"
                content = json.dumps(meta)
                now = datetime.now(timezone.utc).isoformat()
                orphan_rows = await self.db.fetch_all(
                    "SELECT id FROM knowledge_nodes WHERE source = ?", (source_uri,)
                )
                orphan_ids = [r["id"] for r in orphan_rows]
                async with self.db.transaction():
                    await self.db.execute("DELETE FROM knowledge_nodes WHERE source = ?", (source_uri,))
                    if orphan_ids:
                        ph = ",".join(["?"] * len(orphan_ids))
                        await self.db.execute(
                            f"DELETE FROM causal_edges WHERE source_node IN ({ph}) OR target_node IN ({ph})",
                            tuple(orphan_ids + orphan_ids)
                        )
                    await self.db.execute(
                        """
                        INSERT INTO knowledge_nodes
                        (id, content, node_type, confidence, source, created_at, last_validated)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (node_id, content, "code_structure", 1.0, source_uri, now, now)
                    )

    async def _initial_scan(self):
        """Issue 8: Full workspace scan on startup to catch offline changes."""
        target_dir = Path(WORKSPACE_DIR)
        log.info(f"Starting initial FS scan on {target_dir}...")
        
        # We reuse the EventHandler logic to filter
        handler = WorkspaceEventHandler(lambda p, a: None)
        files_to_map = []
        
        for p in target_dir.rglob("*"):
            if not p.is_dir() and handler._is_valid_file(str(p)):
                files_to_map.append(str(p))
                
        if not files_to_map:
            return
            
        log.info(f"Found {len(files_to_map)} files for initial scan. Processing...")
        # Queue them all as 'modified' so _process_batch handles DB logic cleanly
        for f in files_to_map:
            self.pending_events[f] = "modified"
            
        # Manually trigger process batch immediately
        await self._process_batch()

    async def run_loop(self):
        """The main async loop to check the debounce timer."""
        # V2 Constraint: Scope tightly to WORKSPACE_DIR
        target_dir = str(WORKSPACE_DIR)
        
        await self._initial_scan()
        
        event_handler = WorkspaceEventHandler(self._on_event)
        self.observer = Observer()
        
        self.observer.schedule(event_handler, target_dir, recursive=True)
        self.observer.start()
        
        log.info(f"FS Watcher started on {target_dir} (Debounce: {self.debounce_seconds}s)")
        
        try:
            while True:
                await asyncio.sleep(1.0)
                if self.pending_events:
                    elapsed = time.time() - self.last_event_time
                    if elapsed >= self.debounce_seconds:
                        await self._process_batch()
        except asyncio.CancelledError:
            self.observer.stop()
            self.observer.join()
            log.info("FS Watcher gracefully shut down.")
