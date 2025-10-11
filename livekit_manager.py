from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket

try:
    from .mongo_store import MongoChatStore, _utcnow
except ImportError:  # pragma: no cover - fallback for direct script usage
    from mongo_store import MongoChatStore, _utcnow

logger = logging.getLogger(__name__)


class LiveKitConfigurationError(RuntimeError):
    """Raised when LiveKit configuration is missing or invalid."""


class LiveKitSessionNotFound(KeyError):
    """Raised when a LiveKit session cannot be located."""


@dataclass
class TranscriptEntry:
    entry_id: str
    role: str
    text: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.entry_id,
            "role": self.role,
            "text": self.text,
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class LiveKitSession:
    session_id: str
    user_id: str
    display_name: str
    room_name: str
    participant_identity: str
    token: str
    url: str
    created_at: datetime
    expires_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_key_hash: Optional[bytes] = None
    watchers: Set[WebSocket] = field(default_factory=set)
    transcripts: list[TranscriptEntry] = field(default_factory=list)
    transcript_id: Optional[str] = None
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "room_name": self.room_name,
            "participant_identity": self.participant_identity,
            "token": self.token,
            "url": self.url,
            "display_name": self.display_name,
            "created_at": self.created_at.astimezone(timezone.utc).isoformat(),
            "expires_at": self.expires_at.astimezone(timezone.utc).isoformat(),
            "transcripts": [entry.to_dict() for entry in self.transcripts],
            "metadata": self.metadata,
        }


class LiveKitSessionManager:
    """Stateful manager for LiveKit sessions and transcript streaming."""

    def __init__(self, store: MongoChatStore) -> None:
        self.store = store
        self._sessions: Dict[str, LiveKitSession] = {}
        self._lock = asyncio.Lock()

    def _get_session(self, session_id: str) -> LiveKitSession:
        session = self._sessions.get(session_id)
        if not session:
            session = self._rehydrate_session(session_id)
        if not session:
            raise LiveKitSessionNotFound(session_id)
        return session

    def _rehydrate_session(self, session_id: str) -> Optional[LiveKitSession]:
        doc = self.store.get_livekit_session(session_id)
        if not doc:
            logger.debug("LiveKit session %s not found in Mongo for rehydration.", session_id)
            return None
        if doc.get("active") is False:
            logger.debug("LiveKit session %s found in Mongo but inactive.", session_id)
            return None
        logger.debug("Rehydrating LiveKit session %s from Mongo.", session_id)
        created_at = doc.get("created_at") or _utcnow()
        expires_at = doc.get("expires_at") or (created_at + timedelta(hours=1))
        if isinstance(created_at, datetime) and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if isinstance(expires_at, datetime) and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        session = LiveKitSession(
            session_id=session_id,
            user_id=doc.get("user_id", ""),
            display_name=doc.get("display_name") or doc.get("participant_identity") or session_id,
            room_name=doc.get("room_name") or (os.getenv("LIVEKIT_ROOM_PREFIX", "indus") + f"-{session_id}"),
            participant_identity=doc.get("participant_identity") or f"user-{doc.get('user_id', 'unknown')}",
            token=doc.get("token") or "",
            url=doc.get("url") or os.getenv("LIVEKIT_URL", ""),
            created_at=created_at,
            expires_at=expires_at,
            metadata=doc.get("metadata") or {},
            transcripts=[],
            transcript_id=doc.get("transcript_id"),
            active=doc.get("active", True),
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> LiveKitSession:
        return self._get_session(session_id)

    async def create_session(
        self,
        user_id: str,
        display_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LiveKitSession:
        session_id = uuid.uuid4().hex
        participant_identity = f"user-{user_id}-{uuid.uuid4().hex[:6]}"
        token, url, room_name, expires_at = self._generate_access_token(
            identity=participant_identity,
            display_name=display_name,
            session_id=session_id,
        )
        transcript_ref = None
        if metadata and isinstance(metadata, dict):
            transcript_value = metadata.get("transcript_id")
            if isinstance(transcript_value, str) and transcript_value:
                transcript_ref = transcript_value
        session = LiveKitSession(
            session_id=session_id,
            user_id=user_id,
            display_name=display_name or participant_identity,
            room_name=room_name,
            participant_identity=participant_identity,
            token=token,
            url=url,
            created_at=_utcnow(),
            expires_at=expires_at,
            metadata=metadata or {},
            transcript_id=transcript_ref,
        )
        async with self._lock:
            self._sessions[session_id] = session
        self.store.upsert_livekit_session({
            "session_id": session.session_id,
            "user_id": session.user_id,
            "display_name": session.display_name,
            "room_name": session.room_name,
            "participant_identity": session.participant_identity,
            "token": session.token,
            "url": session.url,
            "created_at": session.created_at,
            "expires_at": session.expires_at,
            "active": True,
            "metadata": session.metadata,
            "transcript_id": session.transcript_id,
        })
        logger.info("Created LiveKit session %s for user %s", session_id, user_id)
        return session

    async def end_session(self, session_id: str) -> Optional[str]:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if not session:
            self.store.set_livekit_session_active(session_id, False)
            return None
        session.active = False
        self.store.set_livekit_session_active(session_id, False)
        await self._close_watchers(session)
        transcript_id = await self._persist_session_transcript(session)
        logger.info("Ended LiveKit session %s (transcript_id=%s)", session_id, transcript_id)
        return transcript_id

    async def append_transcript(
        self,
        session_id: str,
        role: str,
        text: str,
        *,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TranscriptEntry:
        session = self._get_session(session_id)
        entry = TranscriptEntry(
            entry_id=uuid.uuid4().hex,
            role=role,
            text=text,
            timestamp=timestamp or _utcnow(),
            metadata=metadata or {},
        )
        session.transcripts.append(entry)
        await self._broadcast_transcript(session, entry)
        return entry

    async def register_websocket(self, session_id: str, websocket: WebSocket) -> None:
        session = self._get_session(session_id)
        session.watchers.add(websocket)
        await websocket.accept()
        history_payload = {
            "type": "history",
            "session_id": session.session_id,
            "transcripts": [entry.to_dict() for entry in session.transcripts],
        }
        await websocket.send_json(history_payload)
        logger.debug("Attached websocket listener to LiveKit session %s", session_id)

    async def unregister_websocket(self, session_id: str, websocket: WebSocket) -> None:
        try:
            session = self._get_session(session_id)
        except LiveKitSessionNotFound:
            return
        session.watchers.discard(websocket)
        logger.debug("Removed websocket listener from LiveKit session %s", session_id)

    def is_authorized_agent(self, provided_key: Optional[str]) -> bool:
        expected = os.getenv("LIVEKIT_AGENT_API_KEY")
        if not expected:
            return False
        if not provided_key:
            return False
        return hmac.compare_digest(provided_key, expected)

    def build_conversation_context(self, session_id: str, max_messages: int = 10) -> str:
        try:
            session = self._get_session(session_id)
        except LiveKitSessionNotFound:
            return ""
        entries = session.transcripts[-max_messages:]
        if not entries:
            return ""
        parts = []
        for entry in entries:
            prefix = "Assistant" if entry.role == "assistant" else "User"
            parts.append(f"{prefix}: {entry.text}")
        combined = "\n".join(parts)
        return combined[-4000:]

    async def _broadcast_transcript(self, session: LiveKitSession, entry: TranscriptEntry) -> None:
        if not session.watchers:
            return
        payload = {
            "type": "transcript",
            "session_id": session.session_id,
            "entry": entry.to_dict(),
        }
        stale: Set[WebSocket] = set()
        for ws in set(session.watchers):
            try:
                await ws.send_json(payload)
            except Exception as exc:
                logger.debug("Removing stale websocket for session %s: %s", session.session_id, exc)
                stale.add(ws)
        for ws in stale:
            session.watchers.discard(ws)

    async def _close_watchers(self, session: LiveKitSession) -> None:
        if not session.watchers:
            return
        for ws in list(session.watchers):
            try:
                await ws.close(code=1000)
            except Exception:
                pass
        session.watchers.clear()

    def issue_viewer_token(
        self,
        *,
        session_id: str,
        user_id: str,
        display_name: Optional[str] = None,
        viewer_identity: Optional[str] = None,
    ) -> tuple[str, str, str, str, datetime]:
        try:
            session = self._get_session(session_id)
        except LiveKitSessionNotFound:
            session = None

        if session and session.user_id != user_id:
            raise PermissionError("Forbidden")

        if not session:
            doc = self.store.get_livekit_session(session_id)
            if not doc or doc.get("active") is False:
                raise LiveKitSessionNotFound(session_id)
            if doc.get("user_id") != user_id:
                raise PermissionError("Forbidden")
            display_name = display_name or doc.get("display_name") or doc.get("participant_identity")
            identity = viewer_identity or f"viewer-{user_id}-{uuid.uuid4().hex[:6]}"
            token, url, room_name, expires_at = self._generate_access_token(
                identity=identity,
                display_name=display_name,
                session_id=session_id,
            )
            return token, url, room_name, identity, expires_at

        identity = viewer_identity or f"viewer-{user_id}-{uuid.uuid4().hex[:6]}"
        token, url, room_name, expires_at = self._generate_access_token(
            identity=identity,
            display_name=display_name or session.display_name,
            session_id=session_id,
        )
        return token, url, room_name, identity, expires_at

    async def _persist_session_transcript(self, session: LiveKitSession) -> Optional[str]:
        if not session.transcripts:
            return None

        def _store() -> Optional[str]:
            transcript_id = session.transcript_id
            if transcript_id:
                existing = self.store.get_transcript(transcript_id, user_id=session.user_id)
            else:
                existing = None
            if not existing:
                title = f"Live session {session.created_at.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
                metadata = {
                    "source": "livekit",
                    "session_id": session.session_id,
                    "room_name": session.room_name,
                }
                transcript_id = self.store.create_transcript(
                    title=title,
                    metadata=metadata,
                    user_id=session.user_id,
                )
            if not transcript_id:
                return None
            for entry in session.transcripts:
                role = entry.role if entry.role in {"user", "assistant"} else "assistant"
                content = [{
                    "type": "text",
                    "payload": {"text": entry.text},
                    "meta": entry.metadata,
                    "timestamp": entry.timestamp,
                }]
                self.store.append_chat(
                    transcript_id=transcript_id,
                    role=role,
                    content=content,
                    meta={"source": "livekit", **(entry.metadata or {})},
                    user_id=session.user_id,
                )
            return transcript_id

        try:
            transcript_id = await asyncio.to_thread(_store)
        except Exception as exc:
            logger.warning("Failed to persist LiveKit session transcript %s: %s", session.session_id, exc)
            return None
        session.transcript_id = transcript_id
        self.store.upsert_livekit_session({
            "session_id": session.session_id,
            "user_id": session.user_id,
            "display_name": session.display_name,
            "room_name": session.room_name,
            "participant_identity": session.participant_identity,
            "token": session.token,
            "url": session.url,
            "created_at": session.created_at,
            "expires_at": session.expires_at,
            "active": session.active,
            "transcript_id": transcript_id,
            "metadata": session.metadata,
        })
        return transcript_id

    def _generate_access_token(
        self,
        *,
        identity: str,
        display_name: Optional[str],
        session_id: str,
    ) -> tuple[str, str, str, datetime]:
        livekit_url = os.getenv("LIVEKIT_URL")
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")
        if not livekit_url or not api_key or not api_secret:
            raise LiveKitConfigurationError("LiveKit environment variables are not fully configured.")
        ttl_seconds = int(os.getenv("LIVEKIT_TTL_SECONDS", "3600"))
        expires_at = _utcnow() + timedelta(seconds=ttl_seconds)
        room_name = os.getenv("LIVEKIT_ROOM_PREFIX", "indus") + f"-{session_id}"
        now = int(time.time())
        payload = {
            "iss": api_key,
            "sub": identity,
            "name": display_name or identity,
            "nbf": now,
            "exp": now + ttl_seconds,
            "video": {
                "room": room_name,
                "roomJoin": True,
                "canPublish": True,
                "canPublishSources": ["microphone", "camera", "screen_share", "screen_share_audio"],
                "canSubscribe": True,
                "canPublishData": True,
            },
        }
        header = {"alg": "HS256", "typ": "JWT"}
        encoded_header = base64.urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode("utf-8")).rstrip(b"=")
        encoded_payload = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).rstrip(b"=")
        signing_input = encoded_header + b"." + encoded_payload
        signature = base64.urlsafe_b64encode(
            hmac.new(api_secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        ).rstrip(b"=")
        jwt = b".".join([encoded_header, encoded_payload, signature]).decode("ascii")
        return jwt, livekit_url, room_name, expires_at
