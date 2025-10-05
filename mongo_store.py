from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, date, time, timezone
import uuid
import re

from pymongo import MongoClient, ReturnDocument
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from decimal import Decimal
from bson.decimal128 import Decimal128


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_transcript_id() -> str:
    return f"transc_{uuid.uuid4()}"


def new_chat_id() -> str:
    return f"chat_{uuid.uuid4().hex[:8]}"


class MongoChatStore:
    """MongoDB-backed store for transcripts and chats.

    Schema:
      transcripts (collection)
        - transcript_id: str (unique)
        - title: str
        - created_at: datetime (UTC)
        - updated_at: datetime (UTC)
        - metadata: dict
        - chats: [
            {
              chat_id: str,
              role: "user" | "assistant",
              created_at: datetime,
              meta: dict,
              content: [
                { type: str, payload: dict, meta: dict, timestamp: datetime }
              ]
            }
          ]
    """

    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "analytics_chat") -> None:
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.col: Collection = self.db["transcripts"]
        self.users: Collection = self.db["users"]
        self.graphs: Collection = self.db["dashboard_graphs"]
        self._ensure_indexes()
        self._ensure_graph_indexes()

    def _ensure_indexes(self) -> None:
        self.col.create_index("transcript_id", unique=True)
        self.col.create_index([("updated_at", -1)])
        self.col.create_index("chats.chat_id")
        self.col.create_index("user_id")
        # users collection indexes
        try:
            self.users.create_index("email", unique=True)
        except Exception:
            pass
        try:
            # Sparse keeps legacy docs without username from violating uniqueness until we backfill.
            self.users.create_index("username", unique=True, sparse=True)
        except Exception:
            pass
        self._backfill_user_usernames()

    def _backfill_user_usernames(self) -> None:
        """Ensure every user document has a username to satisfy the unique index."""
        try:
            cursor = self.users.find(
                {"$or": [
                    {"username": {"$exists": False}},
                    {"username": None},
                    {"username": ""},
                ]},
                {"_id": 1, "email": 1, "user_id": 1},
            )
            for doc in cursor:
                candidate = (doc.get("email") or doc.get("user_id") or f"user_{uuid.uuid4().hex[:10]}")
                try:
                    self.users.update_one({"_id": doc["_id"]}, {"$set": {"username": candidate}})
                except Exception:
                    fallback = f"user_{uuid.uuid4().hex[:12]}"
                    self.users.update_one({"_id": doc["_id"]}, {"$set": {"username": fallback}})
        except Exception:
            # Swallow to avoid failing app startup; registration will still set the field going forward.
            pass

    def _ensure_graph_indexes(self) -> None:
        try:
            self.graphs.create_index([("user_id", 1), ("graph_id", 1)], unique=True)
        except Exception:
            pass
        try:
            self.graphs.create_index([("user_id", 1), ("title_normalized", 1)], unique=True)
        except Exception:
            pass
        try:
            self.graphs.create_index([("user_id", 1), ("active", 1)])
        except Exception:
            pass

    # Transcript lifecycle
    def create_transcript(self, title: str, metadata: Optional[Dict[str, Any]] = None,
                          transcript_id: Optional[str] = None,
                          user_id: Optional[str] = None) -> str:
        tid = transcript_id or new_transcript_id()
        doc = {
            "transcript_id": tid,
            "title": title,
            "created_at": _utcnow(),
            "updated_at": _utcnow(),
            "metadata": {**(metadata or {}), **({"user_id": user_id} if user_id else {})},
            "user_id": user_id,
            "chats": [],
        }
        self.col.insert_one(self._to_bson_safe(doc))
        return tid

    def get_transcript(self, transcript_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        query: Dict[str, Any] = {"transcript_id": transcript_id}
        if user_id is not None:
            query["$or"] = [
                {"user_id": user_id},
                {"user_id": {"$exists": False}, "metadata.user_id": user_id},
            ]
        return self.col.find_one(query, {"_id": 0})

    def list_transcripts(self, limit: int = 50, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return brief transcript docs (no chats) for sidebar listings."""
        projection = {
            "_id": 0,
            "transcript_id": 1,
            "title": 1,
            "created_at": 1,
            "updated_at": 1,
            "metadata": 1,
        }
        query: Dict[str, Any] = {}
        if user_id is not None:
            query = {"$or": [
                {"user_id": user_id},
                {"user_id": {"$exists": False}, "metadata.user_id": user_id},
            ]}
        cur = self.col.find(query, projection).sort("updated_at", -1).limit(limit)
        return list(cur)

    # Chat operations
    def append_chat(self, transcript_id: str, role: str, content: List[Dict[str, Any]],
                    meta: Optional[Dict[str, Any]] = None,
                    created_at: Optional[datetime] = None,
                    chat_id: Optional[str] = None,
                    user_id: Optional[str] = None) -> str:
        chat = {
            "chat_id": chat_id or new_chat_id(),
            "role": role,
            "created_at": created_at or _utcnow(),
            "meta": meta or {},
            "content": content,
        }
        chat_safe = self._to_bson_safe(chat)

        query: Dict[str, Any] = {"transcript_id": transcript_id}
        if user_id is not None:
            query["$or"] = [
                {"user_id": user_id},
                {"user_id": {"$exists": False}, "metadata.user_id": user_id},
            ]
        updated = self.col.find_one_and_update(
            query,
            {"$push": {"chats": chat_safe}, "$set": {"updated_at": _utcnow()}},
            return_document=ReturnDocument.AFTER,
        )
        if not updated:
            raise ValueError(f"Transcript not found: {transcript_id}")
        return chat["chat_id"]

    def get_chat(self, transcript_id: str, chat_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        query: Dict[str, Any] = {
            "transcript_id": transcript_id,
            "chats.chat_id": chat_id,
        }
        if user_id is not None:
            query["$or"] = [
                {"user_id": user_id},
                {"user_id": {"$exists": False}, "metadata.user_id": user_id},
            ]
        doc = self.col.find_one(query, {"_id": 0, "chats.$": 1})
        if not doc or "chats" not in doc or not doc["chats"]:
            return None
        return doc["chats"][0]

    def build_context_from_descriptions(self, transcript_id: str, max_messages: int = 10) -> str:
        """Build conversation_context from prior user queries and assistant descriptions.

        - Includes user text (as "User: ...") and assistant description (as "Assistant: ...")
        - Preserves chronological order using the stored chat sequence
        - Limits to last `max_messages` chat entries to keep context concise
        """
        doc = self.get_transcript(transcript_id)
        if not doc:
            return ""
        parts: List[str] = []
        recent = doc.get("chats", [])[-max_messages:]
        for chat in recent:
            role = chat.get("role")
            content = chat.get("content", [])
            if role == "user":
                # Take first text payload from the user message
                for item in content:
                    if item.get("type") == "text":
                        text = (item.get("payload") or {}).get("text")
                        if text:
                            parts.append(f"User: {text}")
                            break
            elif role == "assistant":
                # Include every description payload from the assistant message
                for item in content:
                    if item.get("type") == "description":
                        text = (item.get("payload") or {}).get("text")
                        if text:
                            parts.append(f"Assistant: {text}")
        if not parts:
            return ""
        joined = "\n".join(parts)
        # Soft length guard
        if len(joined) > 8000:
            joined = joined[-8000:]
        return f"Previous conversation context:\n{joined}"

    # -----------------
    # Dashboard graph management
    # -----------------
    def _normalize_graph_title(self, title: str) -> str:
        return re.sub(r"\s+", " ", (title or "").strip().lower())

    def _build_graph_selector(self, user_id: str, graph_identifier: str) -> Optional[Dict[str, Any]]:
        if not graph_identifier:
            return None
        normalized = self._normalize_graph_title(graph_identifier)
        selector: Dict[str, Any] = {"user_id": user_id}
        if normalized:
            selector["$or"] = [
                {"title_normalized": normalized},
                {"graph_id": graph_identifier},
            ]
        else:
            selector["graph_id"] = graph_identifier
        return selector

    def register_dashboard_graph(self, user_id: str, graph_payload: Dict[str, Any]) -> Dict[str, Any]:
        if not user_id:
            raise ValueError("user_id is required to register a graph")
        if not graph_payload:
            raise ValueError("graph_payload is required")

        title = graph_payload.get("title")
        if not title or not title.strip():
            raise ValueError("Graph title is required")

        normalized = self._normalize_graph_title(title)
        graph_id = graph_payload.get("graph_id") or f"graph_{uuid.uuid4().hex[:12]}"
        now = _utcnow()

        data_payload = graph_payload.get("data")
        if data_payload is None:
            data_list: Optional[List[Dict[str, Any]]] = None
        elif isinstance(data_payload, list):
            data_list = data_payload
        else:
            data_list = [data_payload]

        base_doc: Dict[str, Any] = {
            "graph_id": graph_id,
            "user_id": user_id,
            "title": title.strip(),
            "title_normalized": normalized,
            "graph_type": graph_payload.get("graph_type"),
            "data_source": graph_payload.get("data_source"),
            "description": graph_payload.get("description"),
            "metadata": graph_payload.get("metadata") or {},
            "active": graph_payload.get("active", True),
            "updated_at": now,
            "last_synced_at": now,
        }

        if data_list is not None:
            base_doc["data"] = self._to_bson_safe(data_list)
            base_doc["row_count"] = len(data_list)
            if data_list:
                sample = data_list[0]
                if isinstance(sample, dict):
                    base_doc["fields"] = sorted(sample.keys())

        html_blob = graph_payload.get("html_content") or graph_payload.get("html")
        if html_blob:
            base_doc["html_content"] = html_blob

        update_doc = {
            "$set": self._to_bson_safe(base_doc),
            "$setOnInsert": {"created_at": now},
        }

        self.graphs.update_one(
            {"user_id": user_id, "title_normalized": normalized},
            update_doc,
            upsert=True,
        )

        stored = self.graphs.find_one({"user_id": user_id, "title_normalized": normalized})
        if not stored:
            raise PyMongoError("Failed to persist dashboard graph metadata")
        return self._from_bson(stored)

    def bulk_register_dashboard_graphs(self, user_id: str, graph_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Register multiple dashboard graphs in sequence and return the stored documents."""
        results: List[Dict[str, Any]] = []
        for payload in graph_payloads or []:
            try:
                results.append(self.register_dashboard_graph(user_id, payload))
            except ValueError:
                continue
        return results

    def unregister_dashboard_graph(self, user_id: str, graph_identifier: str) -> bool:
        selector = self._build_graph_selector(user_id, graph_identifier)
        if not selector:
            return False
        result = self.graphs.delete_one(selector)
        return result.deleted_count > 0

    def activate_dashboard_graph(self, user_id: str, graph_identifier: str, active: bool = True) -> bool:
        selector = self._build_graph_selector(user_id, graph_identifier)
        if not selector:
            return False
        result = self.graphs.update_one(selector, {"$set": {"active": active, "updated_at": _utcnow()}})
        return result.modified_count > 0

    def set_dashboard_graphs_active_state(self, user_id: str, graph_identifiers: List[str], active: bool) -> int:
        normalized_targets = {self._normalize_graph_title(identifier) for identifier in graph_identifiers or [] if identifier}
        raw_identifiers = {identifier for identifier in graph_identifiers or [] if identifier}
        clauses: List[Dict[str, Any]] = []
        if normalized_targets:
            clauses.append({"title_normalized": {"$in": list(normalized_targets)}})
        if raw_identifiers:
            clauses.append({"graph_id": {"$in": list(raw_identifiers)}})
        if not clauses:
            return 0
        selector: Dict[str, Any] = {"user_id": user_id, "$or": clauses}
        result = self.graphs.update_many(selector, {"$set": {"active": active, "updated_at": _utcnow()}})
        return result.modified_count

    def activate_only_dashboard_graphs(self, user_id: str, graph_identifiers: List[str]) -> Dict[str, int]:
        normalized_targets = {self._normalize_graph_title(identifier) for identifier in graph_identifiers or [] if identifier}
        if not normalized_targets:
            raise ValueError("At least one graph identifier is required")

        cursor = self.graphs.find({"user_id": user_id})
        activated = 0
        deactivated = 0

        for doc in cursor:
            normalized_title = doc.get("title_normalized")
            matches = normalized_title in normalized_targets or doc.get("graph_id") in graph_identifiers
            if matches and not doc.get("active", True):
                self.graphs.update_one({"_id": doc["_id"]}, {"$set": {"active": True, "updated_at": _utcnow()}})
                activated += 1
            elif not matches and doc.get("active", True):
                self.graphs.update_one({"_id": doc["_id"]}, {"$set": {"active": False, "updated_at": _utcnow()}})
                deactivated += 1

        return {"activated": activated, "deactivated": deactivated}

    def list_dashboard_graphs(self, user_id: str, only_active: bool = False) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {"user_id": user_id}
        if only_active:
            query["active"] = True
        cursor = self.graphs.find(query, {"_id": 0}).sort("updated_at", -1)
        return [self._from_bson(doc) for doc in cursor]

    def get_dashboard_graph_scope(self, user_id: str) -> Dict[str, List[Dict[str, Any]]]:
        scope = {"active": [], "inactive": []}
        cursor = self.graphs.find({"user_id": user_id}, {"_id": 0, "graph_id": 1, "title": 1, "active": 1, "updated_at": 1}).sort("title", 1)
        for doc in cursor:
            clean = self._from_bson(doc)
            entry = {
                "graph_id": clean.get("graph_id"),
                "title": clean.get("title"),
                "active": bool(clean.get("active", True)),
                "updated_at": clean.get("updated_at"),
            }
            key = "active" if entry["active"] else "inactive"
            scope[key].append(entry)
        return scope

    def get_dashboard_graph(self, user_id: str, graph_identifier: str) -> Optional[Dict[str, Any]]:
        selector = self._build_graph_selector(user_id, graph_identifier)
        if not selector:
            return None
        doc = self.graphs.find_one(selector, {"_id": 0})
        return self._from_bson(doc) if doc else None

    def refresh_dashboard_graph_cache(self, user_id: str) -> List[Dict[str, Any]]:
        """Persist a simplified snapshot of active graphs on the user document for legacy clients."""
        graphs = self.list_dashboard_graphs(user_id, only_active=True)
        snapshot: List[Dict[str, Any]] = []
        for graph in graphs:
            snapshot.append({
                "graph_id": graph.get("graph_id"),
                "title": graph.get("title"),
                "graph_type": graph.get("graph_type"),
                "html": (graph.get("html_content") or "")[:20000],
                "data_source": graph.get("data_source"),
                "row_count": graph.get("row_count"),
                "fields": graph.get("fields"),
            })
        self.users.update_one({"user_id": user_id}, {"$set": {"dashboard_graphs": snapshot}}, upsert=False)
        return snapshot

    # -----------------
    # Utility
    # -----------------
    def _to_bson_safe(self, obj: Any) -> Any:
        """Recursively convert Python values into Mongo/BSON-safe values.

        - Decimal -> Decimal128 (precision preserved)
        - tuple -> list
        - datetime.date -> datetime.datetime (UTC, time.min)
        - datetime.datetime -> ensure timezone-aware (UTC if naive)
        - lists/dicts traversed
        """
        # Decimal first
        if isinstance(obj, Decimal):
            return Decimal128(str(obj))

        # datetime.datetime -> ensure tz-aware (UTC)
        if isinstance(obj, datetime):
            if obj.tzinfo is None:
                return obj.replace(tzinfo=timezone.utc)
            return obj

        # datetime.date (but not datetime) -> convert to datetime at midnight UTC
        if isinstance(obj, date) and not isinstance(obj, datetime):
            return datetime.combine(obj, time.min, tzinfo=timezone.utc)

        # Containers
        if isinstance(obj, tuple):
            return [self._to_bson_safe(x) for x in obj]
        if isinstance(obj, list):
            return [self._to_bson_safe(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._to_bson_safe(v) for k, v in obj.items()}

        # Fallback: return object as-is (PyMongo will raise if unsupported)
        return obj

    def _from_bson(self, obj: Any) -> Any:
        """Recursively convert BSON-specific values back into native Python types."""
        if isinstance(obj, Decimal128):
            return Decimal(str(obj.to_decimal()))
        if isinstance(obj, list):
            return [self._from_bson(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._from_bson(v) for k, v in obj.items() if k != "_id"}
        return obj
