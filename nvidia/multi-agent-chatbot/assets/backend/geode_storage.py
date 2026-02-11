#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Geode-based conversation storage with caching and I/O optimization."""

import json
import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from geode_client import Client, TransportType
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage

from logger import logger


@dataclass
class CacheEntry:
    """Cache entry with TTL support."""
    data: Any
    timestamp: float
    ttl: float = 300

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


def _escape_gql_string(value: str) -> str:
    """Escape single quotes in a string for use in GQL literals."""
    return value.replace("'", "\\'")


def _val(row: dict, key: str, default=None):
    """Extract a plain Python value from a Geode row entry.

    The Python client returns Value objects (with .as_string, .as_int, etc.)
    rather than plain types. This helper extracts the raw value.
    """
    v = row.get(key)
    if v is None:
        return default
    # If it's already a plain type (str, int, etc.), return it directly
    if isinstance(v, (str, int, float, bool)):
        return v
    # Geode Value object — use as_string for safe conversion
    if hasattr(v, 'as_string'):
        return v.as_string
    return default


class GeodeConversationStorage:
    """Geode-based conversation storage with intelligent caching and I/O optimization."""

    def __init__(
        self,
        host: str = "geode",
        port: int = 3141,
        username: str = "",
        password: str = "",
        cache_ttl: int = 300
    ):
        """Initialize Geode connection and caching.

        Args:
            host: Geode server host
            port: Geode server port
            username: Geode authentication username
            password: Geode authentication password
            cache_ttl: Cache TTL in seconds
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.cache_ttl = cache_ttl

        self.client: Optional[Client] = None
        self._conn = None

        self._message_cache: Dict[str, CacheEntry] = {}
        self._metadata_cache: Dict[str, CacheEntry] = {}
        self._image_cache: Dict[str, CacheEntry] = {}
        self._chat_list_cache: Optional[CacheEntry] = None

        self._pending_saves: Dict[str, List[BaseMessage]] = {}
        self._save_lock = asyncio.Lock()
        self._batch_save_task: Optional[asyncio.Task] = None

        self._cache_hits = 0
        self._cache_misses = 0
        self._db_operations = 0

    async def init_pool(self) -> None:
        """Initialize the Geode client and persistent connection."""
        try:
            transport_str = os.getenv("GEODE_TRANSPORT", "quic").lower()
            transport = TransportType.GRPC if transport_str == "grpc" else TransportType.QUIC
            kwargs = {"skip_verify": True, "transport_type": transport}
            if self.username:
                kwargs["username"] = self.username
            if self.password:
                kwargs["password"] = self.password
            self.client = Client(
                host=self.host,
                port=self.port,
                **kwargs
            )
            self._conn = await self.client.connect()
            logger.debug("Geode client initialized with persistent connection")

            self._batch_save_task = asyncio.create_task(self._batch_save_worker())

        except Exception as e:
            logger.error(f"Failed to initialize Geode client: {e}")
            raise

    async def _get_conn(self):
        """Get the persistent connection, reconnecting if needed."""
        if self._conn is None:
            self._conn = await self.client.connect()
        return self._conn

    async def _reset_conn(self):
        """Reset the persistent connection after an error."""
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:
                pass
        self._conn = None

    async def close(self) -> None:
        """Cancel background tasks and cleanup."""
        if self._batch_save_task:
            self._batch_save_task.cancel()
            try:
                await self._batch_save_task
            except asyncio.CancelledError:
                pass

        await self._reset_conn()
        logger.debug("Geode storage closed")

    def _message_to_dict(self, message: BaseMessage) -> Dict:
        """Convert a message object to a dictionary for storage."""
        result = {
            "type": message.__class__.__name__,
            "content": message.content,
        }

        if hasattr(message, "tool_calls") and message.tool_calls:
            result["tool_calls"] = message.tool_calls

        if isinstance(message, ToolMessage):
            result["tool_call_id"] = getattr(message, "tool_call_id", None)
            result["name"] = getattr(message, "name", None)

        return result

    def _dict_to_message(self, data: Dict) -> BaseMessage:
        """Convert a dictionary back to a message object."""
        msg_type = data["type"]
        content = data["content"]

        if msg_type == "AIMessage":
            msg = AIMessage(content=content)
            if "tool_calls" in data:
                msg.tool_calls = data["tool_calls"]
            return msg
        elif msg_type == "HumanMessage":
            return HumanMessage(content=content)
        elif msg_type == "SystemMessage":
            return SystemMessage(content=content)
        elif msg_type == "ToolMessage":
            return ToolMessage(
                content=content,
                tool_call_id=data.get("tool_call_id", ""),
                name=data.get("name", "")
            )
        else:
            return HumanMessage(content=content)

    def _get_cached_messages(self, chat_id: str) -> Optional[List[BaseMessage]]:
        """Get messages from cache if available and not expired."""
        cache_entry = self._message_cache.get(chat_id)
        if cache_entry and not cache_entry.is_expired():
            self._cache_hits += 1
            return cache_entry.data

        self._cache_misses += 1
        return None

    def _cache_messages(self, chat_id: str, messages: List[BaseMessage]) -> None:
        """Cache messages with TTL."""
        self._message_cache[chat_id] = CacheEntry(
            data=messages.copy(),
            timestamp=time.time(),
            ttl=self.cache_ttl
        )

    def _invalidate_cache(self, chat_id: str) -> None:
        """Invalidate cache entries for a chat."""
        self._message_cache.pop(chat_id, None)
        self._metadata_cache.pop(chat_id, None)
        self._chat_list_cache = None

    async def exists(self, chat_id: str) -> bool:
        """Check if a conversation exists (with caching)."""
        cached_messages = self._get_cached_messages(chat_id)
        if cached_messages is not None:
            return len(cached_messages) > 0

        conn = await self._get_conn()
        safe_id = _escape_gql_string(chat_id)
        page, _ = await conn.query(
            f"MATCH (c:Conversation {{chatId: '{safe_id}'}}) RETURN c.chatId"
        )
        self._db_operations += 1
        return len(page.rows) > 0

    async def get_messages(self, chat_id: str, limit: Optional[int] = None) -> List[BaseMessage]:
        """Retrieve messages for a chat session with caching."""
        cached_messages = self._get_cached_messages(chat_id)
        if cached_messages is not None:
            return cached_messages[-limit:] if limit else cached_messages

        conn = await self._get_conn()
        safe_id = _escape_gql_string(chat_id)
        page, _ = await conn.query(
            f"MATCH (c:Conversation {{chatId: '{safe_id}'}}) RETURN c.messages"
        )
        self._db_operations += 1

        if not page.rows:
            return []

        messages_json = _val(page.rows[0], "messages", "[]")
        if isinstance(messages_json, str):
            messages_data = json.loads(messages_json)
        else:
            messages_data = messages_json

        messages = [self._dict_to_message(msg_data) for msg_data in messages_data]

        self._cache_messages(chat_id, messages)

        return messages[-limit:] if limit else messages

    async def save_messages(self, chat_id: str, messages: List[BaseMessage]) -> None:
        """Save messages with batching for performance."""
        async with self._save_lock:
            self._pending_saves[chat_id] = messages.copy()

        self._cache_messages(chat_id, messages)

    async def save_messages_immediate(self, chat_id: str, messages: List[BaseMessage]) -> None:
        """Save messages immediately without batching - for critical operations."""
        serialized_messages = [self._message_to_dict(msg) for msg in messages]
        messages_json = _escape_gql_string(json.dumps(serialized_messages))
        now = datetime.utcnow().isoformat()

        conn = await self._get_conn()
        safe_id = _escape_gql_string(chat_id)
        await conn.execute(
            f"MERGE (c:Conversation {{chatId: '{safe_id}'}}) "
            f"ON CREATE SET c.messages = '{messages_json}', "
            f"c.messageCount = {len(messages)}, "
            f"c.updatedAt = '{now}', "
            f"c.createdAt = '{now}' "
            f"ON MATCH SET c.messages = '{messages_json}', "
            f"c.messageCount = {len(messages)}, "
            f"c.updatedAt = '{now}'"
        )
        self._db_operations += 1

        self._cache_messages(chat_id, messages)
        self._chat_list_cache = None

    async def _batch_save_worker(self) -> None:
        """Background worker to batch save operations."""
        while True:
            try:
                await asyncio.sleep(1.0)

                async with self._save_lock:
                    if not self._pending_saves:
                        continue

                    saves_to_process = self._pending_saves.copy()
                    self._pending_saves.clear()

                conn = await self._get_conn()
                for chat_id, messages in saves_to_process.items():
                    serialized_messages = [self._message_to_dict(msg) for msg in messages]
                    messages_json = _escape_gql_string(json.dumps(serialized_messages))
                    now = datetime.utcnow().isoformat()
                    safe_id = _escape_gql_string(chat_id)

                    await conn.execute(
                        f"MERGE (c:Conversation {{chatId: '{safe_id}'}}) "
                        f"ON CREATE SET c.messages = '{messages_json}', "
                        f"c.messageCount = {len(messages)}, "
                        f"c.updatedAt = '{now}', "
                        f"c.createdAt = '{now}' "
                        f"ON MATCH SET c.messages = '{messages_json}', "
                        f"c.messageCount = {len(messages)}, "
                        f"c.updatedAt = '{now}'"
                    )

                self._db_operations += len(saves_to_process)
                if saves_to_process:
                    logger.debug(f"Batch saved {len(saves_to_process)} conversations")
                    self._chat_list_cache = None

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch save worker: {e}")

    async def add_message(self, chat_id: str, message: BaseMessage) -> None:
        """Add a single message to conversation (optimized)."""
        current_messages = await self.get_messages(chat_id)
        current_messages.append(message)

        await self.save_messages(chat_id, current_messages)

    async def delete_conversation(self, chat_id: str) -> bool:
        """Delete a conversation by chat_id."""
        try:
            conn = await self._get_conn()
            safe_id = _escape_gql_string(chat_id)
            await conn.execute(
                f"MATCH (m:ChatMetadata {{chatId: '{safe_id}'}}) DETACH DELETE m"
            )
            await conn.execute(
                f"MATCH (c:Conversation {{chatId: '{safe_id}'}}) DETACH DELETE c"
            )
            self._db_operations += 2

            self._invalidate_cache(chat_id)

            return True
        except Exception as e:
            logger.error(f"Error deleting conversation {chat_id}: {e}")
            return False

    async def list_conversations(self) -> List[str]:
        """List all conversation IDs with caching."""
        if self._chat_list_cache and not self._chat_list_cache.is_expired():
            self._cache_hits += 1
            return self._chat_list_cache.data

        conn = await self._get_conn()
        page, _ = await conn.query(
            "MATCH (c:Conversation) RETURN c.chatId ORDER BY c.updatedAt DESC"
        )
        self._db_operations += 1

        chat_ids = [_val(row, "chatId", "") for row in page.rows]

        self._chat_list_cache = CacheEntry(
            data=chat_ids,
            timestamp=time.time(),
            ttl=60
        )
        self._cache_misses += 1

        return chat_ids

    async def store_image(self, image_id: str, image_base64: str) -> None:
        """Store base64 image data with TTL."""
        now = datetime.utcnow()
        expires = (now + timedelta(hours=1)).isoformat()
        now_str = now.isoformat()
        safe_id = _escape_gql_string(image_id)
        safe_data = _escape_gql_string(image_base64)

        conn = await self._get_conn()
        await conn.execute(
            f"MERGE (img:Image {{imageId: '{safe_id}'}}) "
            f"ON CREATE SET img.imageData = '{safe_data}', "
            f"img.createdAt = '{now_str}', "
            f"img.expiresAt = '{expires}' "
            f"ON MATCH SET img.imageData = '{safe_data}', "
            f"img.createdAt = '{now_str}', "
            f"img.expiresAt = '{expires}'"
        )
        self._db_operations += 1

        self._image_cache[image_id] = CacheEntry(
            data=image_base64,
            timestamp=time.time(),
            ttl=3600
        )

    async def get_image(self, image_id: str) -> Optional[str]:
        """Retrieve base64 image data with caching."""
        cache_entry = self._image_cache.get(image_id)
        if cache_entry and not cache_entry.is_expired():
            self._cache_hits += 1
            return cache_entry.data

        now = datetime.utcnow().isoformat()
        safe_id = _escape_gql_string(image_id)

        conn = await self._get_conn()
        page, _ = await conn.query(
            f"MATCH (img:Image {{imageId: '{safe_id}'}}) "
            f"WHERE img.expiresAt > '{now}' "
            f"RETURN img.imageData"
        )
        self._db_operations += 1

        if page.rows:
            image_data = _val(page.rows[0], "imageData", "")
            self._image_cache[image_id] = CacheEntry(
                data=image_data,
                timestamp=time.time(),
                ttl=3600
            )
            self._cache_misses += 1
            return image_data

        return None

    async def get_chat_metadata(self, chat_id: str) -> Optional[Dict]:
        """Get chat metadata with caching."""
        cache_entry = self._metadata_cache.get(chat_id)
        if cache_entry and not cache_entry.is_expired():
            self._cache_hits += 1
            return cache_entry.data

        safe_id = _escape_gql_string(chat_id)

        conn = await self._get_conn()
        page, _ = await conn.query(
            f"MATCH (m:ChatMetadata {{chatId: '{safe_id}'}}) "
            f"RETURN m.name, m.createdAt"
        )
        self._db_operations += 1

        if page.rows:
            row = page.rows[0]
            metadata = {
                "name": _val(row, "name", f"Chat {chat_id[:8]}"),
                "created_at": _val(row, "createdAt", "")
            }
        else:
            metadata = {"name": f"Chat {chat_id[:8]}"}

        self._metadata_cache[chat_id] = CacheEntry(
            data=metadata,
            timestamp=time.time(),
            ttl=self.cache_ttl
        )
        self._cache_misses += 1

        return metadata

    async def set_chat_metadata(self, chat_id: str, name: str) -> None:
        """Set chat metadata."""
        now = datetime.utcnow().isoformat()
        safe_id = _escape_gql_string(chat_id)
        safe_name = _escape_gql_string(name)

        conn = await self._get_conn()
        await conn.execute(
            f"MERGE (m:ChatMetadata {{chatId: '{safe_id}'}}) "
            f"ON CREATE SET m.name = '{safe_name}', "
            f"m.updatedAt = '{now}', "
            f"m.createdAt = '{now}' "
            f"ON MATCH SET m.name = '{safe_name}', "
            f"m.updatedAt = '{now}'"
        )
        self._db_operations += 1

        self._metadata_cache[chat_id] = CacheEntry(
            data={"name": name},
            timestamp=time.time(),
            ttl=self.cache_ttl
        )

    async def cleanup_expired_images(self) -> int:
        """Clean up expired images and return count of deleted images."""
        now = datetime.utcnow().isoformat()

        conn = await self._get_conn()
        # Count expired images first
        page, _ = await conn.query(
            f"MATCH (img:Image) WHERE img.expiresAt < '{now}' "
            f"RETURN COUNT(img) AS cnt"
        )
        deleted_count = 0
        if page.rows:
            deleted_count = int(_val(page.rows[0], "cnt", 0))

        if deleted_count > 0:
            await conn.execute(
                f"MATCH (img:Image) WHERE img.expiresAt < '{now}' DELETE img"
            )

        self._db_operations += 2

        expired_keys = [
            key for key, entry in self._image_cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self._image_cache[key]

        if deleted_count > 0:
            logger.debug(f"Cleaned up {deleted_count} expired images")

        return deleted_count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "db_operations": self._db_operations,
            "cached_conversations": len(self._message_cache),
            "cached_metadata": len(self._metadata_cache),
            "cached_images": len(self._image_cache)
        }

    def load_conversation_history(self, chat_id: str) -> List[Dict]:
        """Legacy method - converts to async call."""
        import asyncio
        return asyncio.create_task(self._load_conversation_history_dict(chat_id))

    async def _load_conversation_history_dict(self, chat_id: str) -> List[Dict]:
        """Load conversation history in dict format for compatibility."""
        messages = await self.get_messages(chat_id)
        return [self._message_to_dict(msg) for msg in messages]

    def save_conversation_history(self, chat_id: str, messages: List[Dict]) -> None:
        """Legacy method - converts to async call."""
        import asyncio
        message_objects = [self._dict_to_message(msg) for msg in messages]
        return asyncio.create_task(self.save_messages(chat_id, message_objects))
