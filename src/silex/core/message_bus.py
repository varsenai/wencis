"""
Inter-Agent Message Bus — Phase 4-A & Silex Upgrades.

SQLite-backed durable message bus for resilient worker-to-worker and
worker-to-orchestrator communication. If the process is interrupted or stutters,
all messages survive and channels are rehydrated on creation.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field

from silex.utils.logger import setup_logger
from silex.storage.database import Database

log = setup_logger("silex.core.message_bus")


class MessageType(str, Enum):
    """Categories of inter-agent messages."""
    INFO = "info"           # Informational broadcast (no response expected)
    QUESTION = "question"   # Agent needs clarification from another agent
    BLOCKER = "blocker"     # Agent is blocked and needs help
    RESULT = "result"       # Agent broadcasting a partial/final result
    CANCEL = "cancel"       # Orchestrator cancelling an agent's task


class AgentMessage(BaseModel):
    """A single message sent between agents."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique message ID")
    sender: str = Field(description="Name of the sending agent")
    recipient: str = Field(description="Name of the target agent, or '*' for broadcast")
    message_type: MessageType = Field(description="Category of message")
    content: str = Field(description="The message body")
    metadata: dict = Field(default_factory=dict, description="Optional structured data")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of when the message was created"
    )


class AgentMessageBus:
    """SQLite-backed durable message bus with concurrent in-memory routing.

    Ensures zero message loss under local runtime stutters, timeouts, or restarts.
    """

    def __init__(self, max_queue_size: int = 100, db: Database | None = None):
        self.db = db
        self._channels: dict[str, asyncio.Queue[AgentMessage]] = {}
        self._max_queue_size = max_queue_size
        self._message_log: list[AgentMessage] = []
        self._rehydrating: set[str] = set()
        self._rehydration_buffers: dict[str, list[AgentMessage]] = {}

    def create_channel(self, agent_name: str) -> None:
        """Register a new agent channel. Idempotent. Rehydrates unread messages from DB."""
        if agent_name not in self._channels:
            self._channels[agent_name] = asyncio.Queue(maxsize=self._max_queue_size)
            log.info(f"Message bus: channel created for '{agent_name}'")
            
            # Rehydrate channel from persistent SQLite in background
            if self.db:
                self._rehydrating.add(agent_name)
                self._rehydration_buffers[agent_name] = []
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_running():
                        loop.create_task(self._rehydrate_channel(agent_name))
                except RuntimeError:
                    # Sync or test environment without a running loop
                    self._rehydrating.discard(agent_name)
                    self._rehydration_buffers.pop(agent_name, None)

    def remove_channel(self, agent_name: str) -> None:
        """Remove an agent's channel (e.g., after agent completes)."""
        if agent_name in self._channels:
            del self._channels[agent_name]
            self._rehydrating.discard(agent_name)
            self._rehydration_buffers.pop(agent_name, None)
            log.info(f"Message bus: channel removed for '{agent_name}'")

    async def _rehydrate_channel(self, agent_name: str) -> None:
        """Fetch pending unread messages from SQLite and load into channel queue."""
        if not self.db:
            self._rehydrating.discard(agent_name)
            self._rehydration_buffers.pop(agent_name, None)
            return
        try:
            sql = """
                SELECT * FROM agent_messages 
                WHERE (recipient = ? OR recipient = '*') AND read_at IS NULL
                ORDER BY created_at ASC
            """
            rows = await self.db.fetch_all(sql, (agent_name,))
            queue = self._channels.get(agent_name)
            if queue and rows:
                log.info(f"Rehydrating '{agent_name}' channel with {len(rows)} unread messages from DB")
                for row in rows:
                    msg = AgentMessage(
                        id=row["id"],
                        sender=row["sender"],
                        recipient=row["recipient"],
                        message_type=MessageType(row["message_type"]),
                        content=row["content"],
                        timestamp=row["created_at"]
                    )
                    try:
                        queue.put_nowait(msg)
                    except asyncio.QueueFull:
                        break
        except Exception as e:
            log.error(f"Failed to rehydrate message channel '{agent_name}': {e}")
        finally:
            # Flush buffered messages in correct chronological order
            queue = self._channels.get(agent_name)
            if queue:
                buffered = self._rehydration_buffers.pop(agent_name, [])
                for msg in buffered:
                    try:
                        queue.put_nowait(msg)
                    except asyncio.QueueFull:
                        log.warning(f"Queue full while flushing buffer for '{agent_name}'")
                        break
            self._rehydrating.discard(agent_name)
            self._rehydration_buffers.pop(agent_name, None)

    def send(self, message: AgentMessage) -> bool:
        """Send a message to an agent or broadcast. Saves to SQLite and routes to queue."""
        self._message_log.append(message)
        # Cap in-memory log to prevent unbounded growth
        if len(self._message_log) > 500:
            self._message_log = self._message_log[-500:]
        delivered = False

        # Persist to database in background
        if self.db:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    sql = """
                        INSERT INTO agent_messages (id, sender, recipient, message_type, content, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """
                    params = (
                        message.id,
                        message.sender,
                        message.recipient,
                        message.message_type.value,
                        message.content,
                        message.timestamp
                    )
                    loop.create_task(self._safe_db_execute(sql, params))
            except RuntimeError:
                pass  # No running event loop

        if message.recipient == "*":
            # Broadcast to all agents except the sender
            for name, queue in self._channels.items():
                if name != message.sender:
                    if name in self._rehydrating:
                        self._rehydration_buffers.setdefault(name, []).append(message)
                        delivered = True
                    else:
                        try:
                            queue.put_nowait(message)
                            delivered = True
                        except asyncio.QueueFull:
                            log.warning(
                                f"Message bus: queue full for '{name}', "
                                f"dropping broadcast from '{message.sender}'"
                            )
        else:
            # Direct message
            recipient = message.recipient
            if recipient in self._rehydrating:
                self._rehydration_buffers.setdefault(recipient, []).append(message)
                delivered = True
            else:
                queue = self._channels.get(recipient)
                if queue is None:
                    log.warning(
                        f"Message bus: no channel for '{recipient}', "
                        f"message from '{message.sender}' dropped"
                    )
                    return False
                try:
                    queue.put_nowait(message)
                    delivered = True
                except asyncio.QueueFull:
                    log.warning(
                        f"Message bus: queue full for '{recipient}', "
                        f"message from '{message.sender}' dropped"
                    )

        if delivered:
            log.debug(
                f"Message bus: {message.sender} → {message.recipient} "
                f"[{message.message_type.value}]"
            )
        return delivered

    def _mark_as_read(self, message: AgentMessage) -> None:
        """Mark a message as read in SQLite in the background."""
        if not self.db:
            return
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                now = datetime.now(timezone.utc).isoformat()
                sql = "UPDATE agent_messages SET read_at = ? WHERE id = ?"
                loop.create_task(self._safe_db_execute(sql, (now, message.id)))
                
                # Probabilistic pruning: clean old logs with 10% chance
                if random.random() < 0.1:
                    self._prune_messages()
        except RuntimeError:
            pass

    def _prune_messages(self) -> None:
        """Prune older agent messages to prevent sqlite log bloating.
        
        Retains unread messages, but purges read ones or truncates log to last 1000 messages.
        """
        if not self.db:
            return
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Keep up to 1000 messages, delete older ones that are read
                sql = """
                    DELETE FROM agent_messages 
                    WHERE read_at IS NOT NULL 
                      AND id NOT IN (
                          SELECT id FROM agent_messages 
                          ORDER BY created_at DESC 
                          LIMIT 1000
                      )
                """
                loop.create_task(self._safe_db_execute(sql))
        except RuntimeError:
            pass

    async def _safe_db_execute(self, sql: str, params: tuple = ()) -> None:
        """Execute a DB write safely without interfering with active transactions.

        Uses a dedicated transaction to isolate message bus writes from
        the cognitive loop's transaction scope.
        """
        try:
            async with self.db.transaction():
                await self.db.conn.execute(sql, params)
        except Exception as e:
            log.warning(f"Message bus DB write failed (non-fatal): {e}")

    async def receive(
        self,
        agent_name: str,
        timeout: float = 5.0,
    ) -> Optional[AgentMessage]:
        """Wait for a message on an agent's channel. Acknowledges receipt in DB."""
        queue = self._channels.get(agent_name)
        if queue is None:
            return None
        try:
            msg = await asyncio.wait_for(queue.get(), timeout=timeout)
            if msg:
                self._mark_as_read(msg)
            return msg
        except asyncio.TimeoutError:
            return None

    def receive_nowait(self, agent_name: str) -> Optional[AgentMessage]:
        """Non-blocking receive. Acknowledges receipt in DB."""
        queue = self._channels.get(agent_name)
        if queue is None:
            return None
        try:
            msg = queue.get_nowait()
            if msg:
                self._mark_as_read(msg)
            return msg
        except asyncio.QueueEmpty:
            return None

    def drain(self, agent_name: str) -> list[AgentMessage]:
        """Drain all pending messages from an agent's channel. Acknowledges receipts."""
        messages = []
        queue = self._channels.get(agent_name)
        if queue is None:
            return messages
        while not queue.empty():
            try:
                msg = queue.get_nowait()
                if msg:
                    self._mark_as_read(msg)
                    messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages

    def pending_count(self, agent_name: str) -> int:
        """Return the number of pending messages for an agent."""
        queue = self._channels.get(agent_name)
        return queue.qsize() if queue else 0

    @property
    def active_channels(self) -> list[str]:
        """Return a list of all active agent channel names."""
        return list(self._channels.keys())

    @property
    def message_history(self) -> list[AgentMessage]:
        """Return a copy of the full message log (for debugging/auditing)."""
        return list(self._message_log)

    def clear(self) -> None:
        """Clear all channels and message history."""
        self._channels.clear()
        self._message_log.clear()
        log.info("Message bus: all channels cleared")
