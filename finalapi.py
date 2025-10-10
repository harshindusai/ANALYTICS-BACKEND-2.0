
import asyncio
from typing import Dict, Any, List, Optional, Literal
from decimal import Decimal
from bson.decimal128 import Decimal128
import uuid
import json
import os
import base64
import hmac
import hashlib
import datetime
import re
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, status, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone, date, time
import numpy as np


# Import your existing classes (assuming they're in main.py)
from groqnopenai import (
    SmartNL2SQLProcessor,
    DashboardQueryEngine,
)
from mongo_store import MongoChatStore
from livekit_manager import (
    LiveKitSessionManager,
    LiveKitConfigurationError,
    LiveKitSessionNotFound,
)

OPENAPI_TAGS = [
    {"name": "General", "description": "Utility and informational endpoints for service discovery."},
    {"name": "Health", "description": "Operational health and readiness checks."},
    {"name": "Authentication", "description": "User registration, login, and identity introspection."},
    {"name": "Query", "description": "Natural-language query processing and related data retrieval APIs."},
    {"name": "Transcripts", "description": "Transcript lifecycle management and chat retrieval."},
    {"name": "Dashboard", "description": "User dashboard graph management APIs."},
    {"name": "LiveKit", "description": "Realtime voice agent session management APIs."},
]

app = FastAPI(
    title="NL2SQL API Service",
    description="Natural Language to SQL conversion with smart visualization and Mongo-backed transcripts",
    version="2.0.0",
    openapi_tags=OPENAPI_TAGS,
)

logger = logging.getLogger(__name__)

# CORS (safe default, if frontend hits through proxy this is mostly unused)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mongo-backed storage
store = MongoChatStore(uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
                       db_name=os.getenv("MONGO_DB", "analytics_chat"))

# Request/Response models
class QueryRequest(BaseModel):
    natural_language_query: str
    transcript_id: Optional[str] = None
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None  # e.g., {"user_id": "user_123", "channel": "web"}
    conversation_context: Optional[str] = None

class QueryProcessResponse(BaseModel):
    transcript_id: str
    chat_id_user: str
    chat_id_assistant: str
    status: str
    message: str
    processing_time_seconds: float
    is_relevant: bool
    error: Optional[str] = None

class SQLResponse(BaseModel):
    transcript_id: str
    chat_id: str
    sql_query: str
    is_safe: bool
    retry_attempts: int

class TablesResponse(BaseModel):
    transcript_id: str
    chat_id: str
    tables: List[Dict[str, Any]]
    record_count: int

class DescriptionResponse(BaseModel):
    transcript_id: str
    chat_id: str
    description: str
    query_executed_successfully: bool

class GraphItem(BaseModel):
    type: str
    graph_type: Optional[str] = None
    title: Optional[str] = None
    figure: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    query: Optional[str] = None
    insight: Optional[str] = None
    sub_query_index: Optional[int] = None
    html: Optional[str] = None

class GraphsResponse(BaseModel):
    transcript_id: str
    chat_id: str
    graphs: List[GraphItem]

class LiveKitSessionCreateRequest(BaseModel):
    display_name: Optional[str] = None


class LiveKitSessionStartResponse(BaseModel):
    session_id: str
    room_name: str
    participant_identity: str
    token: str
    url: str
    display_name: str
    created_at: datetime
    expires_at: datetime
    transcripts: List[Dict[str, Any]]


class LiveKitTranscriptIngest(BaseModel):
    role: Literal["user", "assistant", "system"]
    text: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LiveKitQueryRequest(BaseModel):
    question: str
    context: Optional[str] = None


class LiveKitTokenRequest(BaseModel):
    display_name: Optional[str] = None


class LiveKitTokenResponse(BaseModel):
    session_id: str
    room_name: str
    participant_identity: str
    token: str
    url: str
    expires_at: datetime


processor = SmartNL2SQLProcessor()
dashboard_query_engine = DashboardQueryEngine(store=store)
livekit_manager = LiveKitSessionManager(store=store)


def _resolve_livekit_client_bundle() -> Optional[str]:
    configured = os.getenv("LIVEKIT_CLIENT_BUNDLE_PATH", "").strip()
    candidates: List[Path] = []
    project_root = Path(__file__).resolve().parent
    cwd_root = Path.cwd()

    if configured:
        path = Path(configured).expanduser()
        candidates.append(path)
        if path.is_dir():
            candidates.append(path / "livekit-client.esm.min.js")

    default_candidates = [
        project_root / "livekit-client.esm.min.js",
        project_root / "static" / "livekit-client.esm.min.js",
        cwd_root / "livekit-client.esm.min.js",
        cwd_root / "static" / "livekit-client.esm.min.js",
        project_root / "node_modules" / "livekit-client" / "dist" / "livekit-client.esm.min.js",
        cwd_root / "node_modules" / "livekit-client" / "dist" / "livekit-client.esm.min.js",
    ]
    candidates.extend(default_candidates)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved.is_file():
            return str(resolved)

    logger.warning("LiveKit client bundle not found. Checked: %s", ", ".join(str(c) for c in candidates))
    return None


@app.get(
    "/livekit/client",
    tags=["LiveKit"],
    summary="Serve LiveKit browser SDK bundle",
)
async def serve_livekit_client() -> FileResponse:
    bundle_path = _resolve_livekit_client_bundle()
    if not bundle_path:
        raise HTTPException(status_code=404, detail="LiveKit client bundle not configured")
    return FileResponse(bundle_path, media_type="application/javascript")

class DashboardGraphsPayload(BaseModel):
    graphs: List[Dict[str, Any]]


class DashboardGraphMetadataModel(BaseModel):
    graph_id: Optional[str] = None
    title: str
    graph_type: Optional[str] = None
    data_source: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    active: Optional[bool] = True
    row_count: Optional[int] = None
    fields: Optional[List[str]] = None
    last_synced_at: Optional[str] = None
    figure: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    html_content: Optional[str] = None


class DashboardGraphRegistrationRequest(DashboardGraphMetadataModel):
    pass


class DashboardGraphRegistrationResponse(BaseModel):
    graph: DashboardGraphMetadataModel


class DashboardGraphListResponse(BaseModel):
    graphs: List[DashboardGraphMetadataModel]


class DashboardScopeUpdateRequest(BaseModel):
    action: str
    graphs: List[str]


class DashboardScopeUpdateResponse(BaseModel):
    action: str
    updated: int
    message: str
    graphs: Optional[List[str]] = None


class DashboardGraphQueryRequest(BaseModel):
    question: str


class DashboardGraphQueryResponse(BaseModel):
    type: str
    message: str
    graphs_used: Optional[List[str]] = None
    analyses: Optional[List[Dict[str, Any]]] = None
    updated: Optional[int] = None
    scope: Optional[Dict[str, Any]] = None


# -----------------
# Auth helpers
# -----------------
SECRET = os.getenv("AUTH_SECRET", "dev-secret-change-me")
TOKEN_TTL_MINUTES = int(os.getenv("AUTH_TTL_MINUTES", "60"))
bearer = HTTPBearer(auto_error=False)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_json(obj: Dict[str, Any]) -> str:
    return _b64url(json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def create_token(user_id: str, email: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    exp = int((datetime.now(timezone.utc) + timedelta(minutes=TOKEN_TTL_MINUTES)).timestamp())
    payload = {"sub": user_id, "email": email, "exp": exp}
    h = _b64url_json(header)
    p = _b64url_json(payload)
    sig = hmac.new(SECRET.encode("utf-8"), msg=f"{h}.{p}".encode("utf-8"), digestmod=hashlib.sha256).digest()
    s = _b64url(sig)
    return f"{h}.{p}.{s}"


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        h, p, s = parts
        expected = _b64url(hmac.new(SECRET.encode("utf-8"), msg=f"{h}.{p}".encode("utf-8"), digestmod=hashlib.sha256).digest())
        if not hmac.compare_digest(s, expected):
            return None
        # decode payload
        padded = p + "=" * (-len(p) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(datetime.now(timezone.utc).timestamp()):
            return None
        return payload
    except Exception:
        return None


async def auth_required(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> Dict[str, Any]:
    # Allow unauthenticated access only for public endpoints
    public_paths = {"/", "/health", "/auth/login", "/auth/register"}
    if request.url.path in public_paths:
        return {"user": None}
    token = None
    if credentials and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Missing auth token")
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    # Attach to request.state for handlers
    request.state.user = {"id": payload["sub"], "email": payload.get("email")}
    return request.state.user


def _require_user(http: Request) -> Dict[str, Any]:
    user = getattr(http.state, "user", None) if http else None
    if not user or not user.get("id"):
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def _require_agent(http: Request) -> None:
    expected = os.getenv("LIVEKIT_AGENT_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="LiveKit agent API key is not configured.")
    provided = http.headers.get("x-agent-key") or http.headers.get("X-Agent-Key")
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid agent credentials")


def _to_bson_safe(obj: Any) -> Any:
    """Recursively convert Python objects to Mongo/BSON-safe values.
    - Decimal -> Decimal128 to preserve precision
    - Lists/Dicts traversed recursively
    """
    if isinstance(obj, datetime):
        # Mongo expects naive UTC timestamps
        if obj.tzinfo is not None:
            obj = obj.astimezone(timezone.utc).replace(tzinfo=None)
        return obj
    if isinstance(obj, date) and not isinstance(obj, datetime):
        return datetime.combine(obj, time.min)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_to_bson_safe(x) for x in obj.tolist()]
    if isinstance(obj, Decimal):
        # Convert to Decimal128 for MongoDB
        return Decimal128(str(obj))
    if isinstance(obj, list):
        return [_to_bson_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_bson_safe(v) for k, v in obj.items()}
    return obj


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert BSON-only types to JSON-serializable ones.
    - Decimal128/Decimal -> float
    - Lists/Dicts traversed recursively
    """
    # Handle BSON Decimal128 or any object exposing to_decimal()
    try:
        if isinstance(obj, Decimal128):
            return float(obj.to_decimal())
    except Exception:
        pass
    # Duck-typing fallback (e.g., if import path differs)
    if hasattr(obj, 'to_decimal') and callable(getattr(obj, 'to_decimal')):
        try:
            return float(obj.to_decimal())
        except Exception:
            try:
                return str(obj.to_decimal())
            except Exception:
                return str(obj)
    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=timezone.utc)
        return obj.isoformat()
    if isinstance(obj, date) and not isinstance(obj, datetime):
        return datetime.combine(obj, time.min, tzinfo=timezone.utc).isoformat()
    if isinstance(obj, Decimal):
        try:
            return float(obj)
        except Exception:
            return str(obj)
    if isinstance(obj, list):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    return obj





def _derive_graph_title_from_context(ctx: Dict[str, Any], fallback_prefix: str, index: int) -> str:
    for key in ("title", "insight", "query"):
        candidate = (ctx.get(key) or "").strip()
        if candidate:
            return candidate[:160]
    return f"{fallback_prefix} {index + 1}"


def _match_data_for_context(ctx: Dict[str, Any], result: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    target_query = (ctx.get("query") or "").strip().lower()
    target_index = ctx.get("sub_query_index")
    sub_results = result.get("sub_results") or []

    if target_index is not None:
        for sub in sub_results:
            if sub.get("sub_query_index") == target_index:
                return sub.get("execution_results")

    if target_query:
        for sub in sub_results:
            candidate = (sub.get("executed_sub_query") or sub.get("original_query") or "").strip().lower()
            if candidate and candidate == target_query:
                return sub.get("execution_results")

    return result.get("execution_results")


def _collect_graph_metadata_payloads(
    result: Dict[str, Any],
    graph_contexts: List[Dict[str, Any]],
    transcript_id: str,
    chat_id: str,
    user_query: str,
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    if graph_contexts:
        for idx, ctx in enumerate(graph_contexts):
            title = _derive_graph_title_from_context(ctx, "Visualization", idx)
            figure_payload = ctx.get("figure")
            summary_payload = ctx.get("summary")
            payloads.append({
                "title": title,
                "graph_type": ctx.get("graph_type"),
                "data_source": None,
                "data": _match_data_for_context(ctx, result),
                "description": ctx.get("insight"),
                "figure": figure_payload,
                "config": ctx.get("config"),
                "summary": summary_payload,
                "html_content": ctx.get("html"),
                "metadata": {
                    "query": ctx.get("query"),
                    "insight": ctx.get("insight"),
                    "transcript_id": transcript_id,
                    "chat_id": chat_id,
                    "user_query": user_query,
                    "sub_query_index": ctx.get("sub_query_index"),
                },
                "active": True,
            })
    elif result.get("visualization"):
        viz = result["visualization"]
        title = (viz.get("title") or result.get("description") or result.get("user_request") or user_query or "Visualization").strip()
        payloads.append({
            "title": title[:160],
            "graph_type": viz.get("graph_type") or viz.get("type"),
            "data_source": None,
            "data": result.get("execution_results"),
            "description": result.get("description"),
            "figure": viz.get("figure"),
            "config": viz.get("config"),
            "summary": viz if viz.get("type") == "summary_card" else None,
            "html_content": viz.get("html"),
            "metadata": {
                "query": result.get("executed_sub_query") or result.get("user_request") or user_query,
                "transcript_id": transcript_id,
                "chat_id": chat_id,
            },
            "active": True,
        })
    return payloads





@app.post(
    "/process_query",
    response_model=QueryProcessResponse,
    dependencies=[Depends(auth_required)],
    tags=["Query"],
    summary="Process a natural-language analytics request",
    description="Converts a user's prompt into SQL, executes it, persists the transcript, and returns processing metadata.",
)
async def process_query(request: QueryRequest, http: Request = None):
    """Process a user query within a transcript (creating transcript if needed),
    auto-build conversation_context from prior assistant descriptions,
    persist both user and assistant messages in MongoDB, and return IDs.
    """
    try:
        # Ensure transcript
        transcript_id = request.transcript_id
        user = _require_user(http)
        user_id = user.get("id")
        if transcript_id:
            # Verify ownership
            if not store.get_transcript(transcript_id, user_id=user_id):
                raise HTTPException(status_code=404, detail="Transcript not found")
        else:
            title = request.title or request.natural_language_query[:60]
            transcript_id = store.create_transcript(title=title, metadata=request.metadata or {}, user_id=user_id)

        # Voice-first mode works stateless; honor explicit context only
        conversation_context = request.conversation_context or ""

        # Append user chat
        user_chat_content = [{
            "type": "text",
            "payload": {"text": request.natural_language_query},
            "meta": {},
            "timestamp": datetime.utcnow()
        }]
        chat_id_user = store.append_chat(
            transcript_id=transcript_id,
            role="user",
            content=user_chat_content,
            meta={"source": "api", **(request.metadata or {})},
            user_id=user_id,
        )

        # Process the query
        result = processor.process_query_with_smart_visualization(
            request.natural_language_query,
            conversation_context
        )

        # Determine status
        if result.get('error'):
            status_msg = "error"
            message = f"Query processing failed: {result['error']}"
        elif not result.get('is_relevant', True):
            status_msg = "irrelevant"
            message = "Query is not relevant to business data analysis"
        elif result.get('dashboard_only'):
            status_msg = "success"
            message = "Dashboard updated per your request."
        elif result.get('warning'):
            status_msg = "warning"
            message = f"Query processed with warning: {result['warning']}"
        else:
            status_msg = "success"
            message = "Query processed successfully."

        total_rows = result.get('row_count')
        execution_rows = result.get('execution_results')
        if total_rows is None and isinstance(execution_rows, list):
            total_rows = len(execution_rows)

        if status_msg == "success" and not result.get('dashboard_only') and total_rows is not None:
            if total_rows == 0:
                message = "No data was found matching your query criteria."
            else:
                message = f"Query processed successfully. Found {total_rows} record(s)."

        # Build assistant chat content (includes attempts, sql, table, graph, description)
        content_items: List[Dict[str, Any]] = []

        # Store failed attempts (if any) with attempt number, sql, and error
        for retry in result.get('retry_details', []) or []:
            content_items.append({
                "type": "attempt",
                "payload": {
                    "attempt": retry.get("attempt"),
                    "sql": retry.get("sql_query"),
                    "error": retry.get("error"),
                },
                "meta": {"status": "failed"},
                "timestamp": datetime.utcnow(),
            })

        if result.get('sql_query'):
            content_items.append({
                "type": "sql",
                "payload": {"sql": result['sql_query']},
                "meta": {"engine": "nl2sql", "safe": result.get('query_safe', True)},
                "timestamp": datetime.utcnow(),
            })

        rows_payload = result.get('execution_results')
        if isinstance(rows_payload, list) and not result.get('dashboard_only'):
            rows = rows_payload
            # Flatten to columns/rows for compact storage
            if rows:
                columns = list(rows[0].keys())
                table_rows = [[r.get(c) for c in columns] for r in rows]
            else:
                columns, table_rows = [], []
            content_items.append({
                "type": "table",
                "payload": {"columns": columns, "rows": table_rows},
                "meta": {"row_count": len(rows)},
                "timestamp": datetime.utcnow(),
            })

        dashboard_action = (result.get('dashboard_action') or 'none').lower()
        dashboard_target = (result.get('dashboard_target') or '').strip()
        dashboard_reasoning = result.get('dashboard_reasoning') or ''
        dashboard_action_result: Optional[str] = None

        description_text = result.get('combined_description') or result.get('description') or ""
        if not description_text and not result.get('dashboard_only') and total_rows == 0:
            description_text = "No data was found matching your query criteria."

        graph_contexts: List[Dict[str, Any]] = []
        viz_list_raw = result.get('visualizations') or []
        if isinstance(viz_list_raw, dict):
            viz_list_raw = [viz_list_raw]

        sub_results_index: Dict[Any, Dict[str, Any]] = {}
        for sub_res in result.get('sub_results') or []:
            if not isinstance(sub_res, dict):
                continue
            idx = sub_res.get('sub_query_index')
            if idx is not None:
                sub_results_index[idx] = sub_res
            original_query_key = (sub_res.get('original_query') or "").strip().lower()
            if original_query_key:
                sub_results_index[original_query_key] = sub_res

        def _append_context(payload: Optional[Dict[str, Any]], query: Optional[str], insight: Optional[str], sub_index: Optional[int] = None) -> None:
            if not payload:
                return
            html_blob = payload.get('html') or payload.get('html_content')
            payload_type = payload.get('type')
            if not payload_type:
                if payload.get('figure'):
                    payload_type = 'plotly'
                elif payload.get('summary'):
                    payload_type = 'summary_card'
                elif html_blob:
                    payload_type = 'html'
                else:
                    payload_type = 'plotly'
            summary_payload = payload.get('summary')
            if payload_type == 'summary_card' and not summary_payload:
                summary_payload = payload
            graph_contexts.append({
                "type": payload_type,
                "graph_type": payload.get('graph_type') or payload.get('type'),
                "title": payload.get('title'),
                "figure": payload.get('figure'),
                "config": payload.get('config'),
                "summary": summary_payload,
                "query": query,
                "insight": insight,
                "sub_query_index": sub_index,
                "html": html_blob,
            })

        if viz_list_raw:
            for viz in viz_list_raw:
                if not isinstance(viz, dict):
                    continue
                payload = viz.get('payload') if isinstance(viz.get('payload'), dict) else None
                if payload is None:
                    payload = viz if any(key in viz for key in ('figure', 'summary', 'type')) else None
                if payload is None:
                    continue
                sub_idx = viz.get('sub_query_index')
                base_query = viz.get('sub_query')
                if not base_query and sub_idx in sub_results_index:
                    base_query = sub_results_index[sub_idx].get('original_query')
                if not base_query:
                    base_query = result.get('executed_sub_query') or result.get('original_query') or request.natural_language_query
                insight_text = None
                if sub_idx in sub_results_index:
                    insight_text = sub_results_index[sub_idx].get('description')
                elif base_query and base_query.strip().lower() in sub_results_index:
                    insight_text = sub_results_index[base_query.strip().lower()].get('description')
                _append_context(payload, base_query, insight_text, sub_index=sub_idx)

        if not graph_contexts and result.get('visualization'):
            viz_payload = result['visualization']
            base_query = result.get('executed_sub_query') or result.get('original_query') or request.natural_language_query
            insight_text = result.get('description') or description_text
            _append_context(viz_payload, base_query, insight_text, sub_index=None)

        if not graph_contexts:
            desc_text_fallback = description_text or "No visual required."
            graph_contexts.append({
                "type": "summary_card",
                "graph_type": "summary_card",
                "title": "Insight",
                "figure": None,
                "config": None,
                "summary": {
                    "type": "summary_card",
                    "title": "Insight",
                    "description": desc_text_fallback,
                    "metrics": [],
                    "primary_metric": None,
                    "sub_query_index": None,
                },
                "query": request.natural_language_query,
                "insight": desc_text_fallback,
                "sub_query_index": None,
            })

        for ctx in graph_contexts:
            if not ctx.get('insight'):
                ctx['insight'] = description_text or ctx.get('query') or request.natural_language_query

        for ctx in graph_contexts:
            content_items.append({
                "type": "graph",
                "payload": {
                    "type": ctx.get('type'),
                    "graph_type": ctx.get('graph_type'),
                    "title": ctx.get('title'),
                    "figure": ctx.get('figure'),
                    "config": ctx.get('config'),
                    "summary": ctx.get('summary'),
                    "query": ctx.get('query'),
                    "insight": ctx.get('insight'),
                    "sub_query_index": ctx.get('sub_query_index'),
                    "html": ctx.get('html'),
                },
                "meta": {},
                "timestamp": datetime.utcnow(),
            })

        # Description: prefer combined_description in multi query, else single description
        desc_text = description_text
        if desc_text:
            content_items.append({
                "type": "description",
                "payload": {"text": desc_text},
                "meta": {},
                "timestamp": datetime.utcnow(),
            })

        dashboard_update_details: Dict[str, Any] = {}

        if dashboard_action == "add":
            if graph_contexts:
                registered_count = len(graph_contexts)
                noun = "visualization" if registered_count == 1 else "visualizations"
                dashboard_action_result = f"Registered {registered_count} {noun} for dashboard querying."
                dashboard_update_details["graphs_registered"] = registered_count
            else:
                dashboard_action_result = "No visualizations were generated to register."

        elif dashboard_action == "remove":
            all_graphs = dashboard_query_engine.list_graphs(user_id, active_only=False)
            matched_graphs = []
            if not all_graphs:
                dashboard_action_result = "No dashboard graphs exist yet."
            else:
                target_text = (dashboard_target or "").strip()
                target_key = target_text.lower()
                if target_key in {"all", "everything"}:
                    matched_graphs = all_graphs
                    identifiers = [graph.graph_id for graph in matched_graphs]
                    updated = dashboard_query_engine.exclude_graphs_from_scope(user_id, identifiers)
                    if updated:
                        dashboard_action_result = f"Deactivated {updated} graph(s) from the query scope."
                    else:
                        dashboard_action_result = "All graphs were already inactive."
                else:
                    if target_key in {"", "latest", "last"}:
                        matched_graphs = [all_graphs[0]]
                    else:
                        matched_graphs = dashboard_query_engine.resolve_graphs_from_text(
                            user_id,
                            target_text,
                            graphs=all_graphs,
                        )
                    if not matched_graphs:
                        dashboard_action_result = f"No dashboard graph matched '{target_text or 'the request'}'."
                    else:
                        identifiers = [graph.graph_id for graph in matched_graphs]
                        updated = dashboard_query_engine.exclude_graphs_from_scope(user_id, identifiers)
                        titles = ", ".join(graph.title for graph in matched_graphs)
                        if updated:
                            dashboard_action_result = f"Removed {updated} visualization(s) from query scope: {titles}."
                        else:
                            dashboard_action_result = f"Graphs already inactive for querying: {titles}."
            dashboard_update_details["graphs_modified"] = [graph.title for graph in matched_graphs]

        assistant_meta = {
            "processing_time_seconds": result.get('processing_time_seconds', 0),
            "is_relevant": result.get('is_relevant', True),
            "warning": result.get('warning'),
            "error": result.get('error'),
            "query_safe": result.get('query_safe', True),
            "retry_attempts": result.get('retry_attempts', 0),
            "dashboard_action": dashboard_action,
            "dashboard_target": dashboard_target,
            "dashboard_result": dashboard_action_result,
            "dashboard_reasoning": dashboard_reasoning,
            "row_count": total_rows,
        }

        # Sanitize content and meta for Mongo (Decimal -> Decimal128)
        content_items_safe = _to_bson_safe(content_items)
        assistant_meta_safe = _to_bson_safe(assistant_meta)

        chat_id_assistant = store.append_chat(
            transcript_id=transcript_id,
            role="assistant",
            content=content_items_safe,
            meta=assistant_meta_safe,
            user_id=user_id,
        )

        # Register generated graphs for dashboard querying only when explicitly requested
        registered_graph_titles: List[str] = []
        should_register_graphs = dashboard_action == "add" or bool(result.get("register_dashboard_graphs"))
        if should_register_graphs:
            try:
                graph_payloads = _collect_graph_metadata_payloads(
                    result=result,
                    graph_contexts=graph_contexts,
                    transcript_id=transcript_id,
                    chat_id=chat_id_assistant,
                    user_query=request.natural_language_query,
                )
                for payload in graph_payloads:
                    try:
                        stored_graph = dashboard_query_engine.register_graph(user_id, payload)
                        if stored_graph and stored_graph.title:
                            registered_graph_titles.append(stored_graph.title)
                    except Exception as register_exc:
                        logger.warning("Failed to register graph metadata for user %s: %s", user_id, register_exc)
            except Exception as graph_error:
                logger.warning("Graph metadata collection failed: %s", graph_error)

        if dashboard_action in {"add", "remove"}:
            if dashboard_action == "add" and registered_graph_titles:
                dashboard_update_details.setdefault("graphs_registered_titles", registered_graph_titles)
                preview = ", ".join(registered_graph_titles[:3])
                if len(registered_graph_titles) > 3:
                    preview += ", ..."
                suffix = f" Saved as: {preview}."
                if dashboard_action_result:
                    dashboard_action_result = f"{dashboard_action_result.rstrip('.')}" + suffix
                else:
                    dashboard_action_result = suffix.lstrip()
            update_payload = {
                "action": dashboard_action,
                "target": dashboard_target,
                "result": dashboard_action_result,
                "reasoning": dashboard_reasoning,
            }
            if dashboard_update_details:
                update_payload["details"] = dashboard_update_details
            content_items.append({
                "type": "dashboard_update",
                "payload": update_payload,
                "meta": {},
                "timestamp": datetime.utcnow(),
            })
            if dashboard_action_result:
                message = f"{message} {dashboard_action_result}".strip()

        return QueryProcessResponse(
            transcript_id=transcript_id,
            chat_id_user=chat_id_user,
            chat_id_assistant=chat_id_assistant,
            status=status_msg,
            message=message,
            processing_time_seconds=result.get('processing_time_seconds', 0),
            is_relevant=result.get('is_relevant', True),
            error=result.get('error')
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get(
    "/get_sql/{transcript_id}/{chat_id}",
    response_model=SQLResponse,
    dependencies=[Depends(auth_required)],
    tags=["Query"],
    summary="Fetch generated SQL",
    description="Returns the SQL statement produced for a specific assistant response along with safety metadata.",
)
async def get_sql_query(transcript_id: str, chat_id: str, http: Request = None):
    user = _require_user(http)
    user_id = user.get("id")
    chat = store.get_chat(transcript_id, chat_id, user_id=user_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    sql = ""
    is_safe = True
    retry_attempts = 0
    # Look into content and meta
    for item in chat.get("content", []):
        if item.get("type") == "sql":
            sql = (item.get("payload") or {}).get("sql", "")
            is_safe = (item.get("meta") or {}).get("safe", True)
            break
    retry_attempts = int((chat.get("meta") or {}).get("retry_attempts", 0))
    return SQLResponse(transcript_id=transcript_id, chat_id=chat_id, sql_query=sql, is_safe=is_safe, retry_attempts=retry_attempts)

@app.get(
    "/get_tables/{transcript_id}/{chat_id}",
    response_model=TablesResponse,
    dependencies=[Depends(auth_required)],
    tags=["Query"],
    summary="Retrieve tabular query results",
    description="Returns the rows and schema for a processed query result from a specific assistant message.",
)
async def get_tables(transcript_id: str, chat_id: str, http: Request = None):
    user = _require_user(http)
    user_id = user.get("id")
    chat = store.get_chat(transcript_id, chat_id, user_id=user_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    tables: List[Dict[str, Any]] = []
    record_count = 0
    for item in chat.get("content", []):
        if item.get("type") == "table":
            payload = item.get("payload") or {}
            cols = payload.get("columns") or []
            rows = payload.get("rows") or []
            record_count = int((item.get("meta") or {}).get("row_count", len(rows)))
            # Reconstruct list[dict] shape similar to previous API
            tables = [dict(zip(cols, r)) for r in rows]
            break
    tables = _to_json_safe(tables)
    return TablesResponse(transcript_id=transcript_id, chat_id=chat_id, tables=tables, record_count=record_count)

@app.get(
    "/get_description/{transcript_id}/{chat_id}",
    response_model=DescriptionResponse,
    dependencies=[Depends(auth_required)],
    tags=["Query"],
    summary="Retrieve narrative insight",
    description="Returns the narrative description generated by the LLM for an assistant response.",
)
async def get_description(transcript_id: str, chat_id: str, http: Request = None):
    user = _require_user(http)
    user_id = user.get("id")
    chat = store.get_chat(transcript_id, chat_id, user_id=user_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    description = ""
    for item in chat.get("content", []):
        if item.get("type") == "description":
            description = (item.get("payload") or {}).get("text", "")
            break
    query_ok = not bool((chat.get("meta") or {}).get("error"))
    return DescriptionResponse(transcript_id=transcript_id, chat_id=chat_id, description=description or "No description available", query_executed_successfully=query_ok)

@app.get(
    "/get_graph/{transcript_id}/{chat_id}",
    response_model=GraphsResponse,
    dependencies=[Depends(auth_required)],
    tags=["Query"],
    summary="Retrieve rendered visualizations",
    description="Returns visualization artifacts (inline HTML or file-backed) associated with a processed query.",
)
async def get_graph_html(transcript_id: str, chat_id: str, http: Request = None):
    user = _require_user(http)
    user_id = user.get("id")
    chat = store.get_chat(transcript_id, chat_id, user_id=user_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    graphs: List[GraphItem] = []
    for item in chat.get("content", []):
        if item.get("type") == "graph":
            payload = item.get("payload") or {}
            graphs.append(GraphItem(
                type=payload.get("type") or ("summary_card" if payload.get("summary") else "plotly"),
                graph_type=payload.get("graph_type") or payload.get("type"),
                title=payload.get("title"),
                figure=_to_json_safe(payload.get("figure")),
                config=_to_json_safe(payload.get("config")),
                summary=_to_json_safe(payload.get("summary")),
                query=payload.get("query"),
                insight=payload.get("insight"),
                sub_query_index=payload.get("sub_query_index"),
                html=payload.get("html") or payload.get("html_content"),
            ))
    if not graphs:
        desc_text = ""
        for item in chat.get("content", []):
            if item.get("type") == "description":
                desc_text = (item.get("payload") or {}).get("text") or ""
                break
        summary_payload = {
            "type": "summary_card",
            "title": "Insight",
            "description": desc_text or "No visual required.",
            "metrics": [],
            "primary_metric": None,
            "sub_query_index": None,
        }
        graphs = [GraphItem(
            type="summary_card",
            graph_type="summary_card",
            title="Insight",
            summary=_to_json_safe(summary_payload),
            insight=desc_text or "No visual required.",
        )]
    return GraphsResponse(transcript_id=transcript_id, chat_id=chat_id, graphs=graphs)



@app.get(
    "/dashboard/graphs",
    response_model=DashboardGraphListResponse,
    dependencies=[Depends(auth_required)],
    tags=["Dashboard"],
    summary="List dashboard graphs",
    description="Returns registered dashboard graphs for the authenticated user.",
)
async def list_dashboard_graphs(active_only: bool = False, http: Request = None):
    user = _require_user(http)
    user_id = user.get("id")
    graphs = dashboard_query_engine.list_graphs(user_id, active_only=active_only)
    return DashboardGraphListResponse(
        graphs=[DashboardGraphMetadataModel(**_to_json_safe(graph.to_dict())) for graph in graphs]
    )


@app.post(
    "/dashboard/graphs",
    response_model=DashboardGraphRegistrationResponse,
    dependencies=[Depends(auth_required)],
    tags=["Dashboard"],
    summary="Register dashboard graph",
    description="Creates or updates metadata for a dashboard graph so it can be queried.",
)
async def register_dashboard_graph(payload: DashboardGraphRegistrationRequest, http: Request = None):
    user = _require_user(http)
    user_id = user.get("id")
    graph = dashboard_query_engine.register_graph(user_id, payload.dict())
    return DashboardGraphRegistrationResponse(graph=DashboardGraphMetadataModel(**_to_json_safe(graph.to_dict())))


@app.delete(
    "/dashboard/graphs/{graph_identifier}",
    dependencies=[Depends(auth_required)],
    tags=["Dashboard"],
    summary="Unregister dashboard graph",
    description="Removes a graph from the dashboard metadata store.",
)
async def unregister_dashboard_graph(graph_identifier: str, http: Request = None):
    user = _require_user(http)
    user_id = user.get("id")
    success = dashboard_query_engine.unregister_graph(user_id, graph_identifier)
    if not success:
        raise HTTPException(status_code=404, detail="Graph not found")
    return {"status": "success", "graph": graph_identifier}


@app.post(
    "/dashboard/graphs/scope",
    response_model=DashboardScopeUpdateResponse,
    dependencies=[Depends(auth_required)],
    tags=["Dashboard"],
    summary="Update dashboard query scope",
    description="Activate, deactivate, or exclusively scope dashboard graphs for querying.",
)
async def update_dashboard_scope(payload: DashboardScopeUpdateRequest, http: Request = None):
    user = _require_user(http)
    user_id = user.get("id")
    action = (payload.action or "").strip().lower()
    identifiers = payload.graphs or []
    if not identifiers:
        raise HTTPException(status_code=400, detail="At least one graph identifier is required")

    if action == "exclusive":
        result = dashboard_query_engine.set_graph_scope(user_id, identifiers)
        return DashboardScopeUpdateResponse(
            action="exclusive",
            updated=result.get("activated", 0),
            message="Activated selected graphs and deactivated the rest.",
            graphs=identifiers,
        )
    elif action in {"activate", "deactivate"}:
        desired_state = action == "activate"
        updated = 0
        for identifier in identifiers:
            if dashboard_query_engine.set_graph_active(user_id, identifier, active=desired_state):
                updated += 1
        message = "Activated graph(s)." if desired_state else "Deactivated graph(s)."
        return DashboardScopeUpdateResponse(
            action=action,
            updated=updated,
            message=message,
            graphs=identifiers,
        )

    raise HTTPException(status_code=400, detail="Unsupported scope action. Use activate, deactivate, or exclusive.")


@app.post(
    "/dashboard/graphs/query",
    response_model=DashboardGraphQueryResponse,
    dependencies=[Depends(auth_required)],
    tags=["Dashboard"],
    summary="Query dashboard graphs",
    description="Answer natural-language questions against dashboard graph data and metadata.",
)
async def query_dashboard_graphs(payload: DashboardGraphQueryRequest, http: Request = None):
    user = _require_user(http)
    user_id = user.get("id")
    result = dashboard_query_engine.handle_request(user_id, payload.question)

    response = DashboardGraphQueryResponse(
        type=result.get("type", "query"),
        message=result.get("message", ""),
        graphs_used=result.get("graphs_used") or result.get("graphs"),
        analyses=_to_json_safe(result.get("analyses")) if result.get("analyses") is not None else None,
        updated=result.get("updated"),
        scope=_to_json_safe(result.get("scope")) if result.get("scope") is not None else None,
    )
    return response


@app.get(
    "/transcripts",
    dependencies=[Depends(auth_required)],
    tags=["Transcripts"],
    summary="List transcripts",
    description="Returns recent transcripts for the authenticated user ordered by last update time.",
)
async def list_transcripts(limit: int = 50, http: Request = None):
    user = _require_user(http)
    user_id = user.get("id")
    items = store.list_transcripts(limit=limit, user_id=user_id)
    return _to_json_safe(items)

@app.post(
    "/transcripts",
    dependencies=[Depends(auth_required)],
    tags=["Transcripts"],
    summary="Create a transcript",
    description="Creates a new transcript container that chats can later be appended to.",
)
async def create_transcript(payload: Dict[str, Any], http: Request = None):
    title = payload.get("title") or "New Transcript"
    metadata = payload.get("metadata") or {}
    user = _require_user(http)
    user_id = user.get("id")
    tid = store.create_transcript(title=title, metadata=metadata, user_id=user_id)
    return {"transcript_id": tid}

@app.get(
    "/transcripts/{transcript_id}",
    dependencies=[Depends(auth_required)],
    tags=["Transcripts"],
    summary="Get transcript details",
    description="Fetches a transcript's metadata for the authenticated user.",
)
async def get_transcript(transcript_id: str, http: Request = None):
    user = _require_user(http)
    user_id = user.get("id")
    doc = store.get_transcript(transcript_id, user_id=user_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return _to_json_safe(doc)

@app.get(
    "/transcripts/{transcript_id}/chats/{chat_id}",
    dependencies=[Depends(auth_required)],
    tags=["Transcripts"],
    summary="Get chat message",
    description="Returns a single chat entry (user or assistant) from a transcript.",
)
async def get_chat(transcript_id: str, chat_id: str, http: Request = None):
    user = _require_user(http)
    user_id = user.get("id")
    chat = store.get_chat(transcript_id, chat_id, user_id=user_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return _to_json_safe(chat)

@app.get(
    "/",
    tags=["General"],
    summary="Service overview",
    description="Provides high-level API metadata and enumerates available endpoints.",
)
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "NL2SQL API Service",
        "version": "2.0.0",
        "endpoints": {
            "POST /process_query": "Process natural language query and persist to Mongo",
            "GET /get_sql/{transcript_id}/{chat_id}": "Get generated SQL",
            "GET /get_tables/{transcript_id}/{chat_id}": "Get extracted table data",
            "GET /get_description/{transcript_id}/{chat_id}": "Get LLM description",
            "GET /get_graph/{transcript_id}/{chat_id}": "Get graph payload (Plotly JSON or summary data)",
            "GET /transcripts": "List transcripts",
            "POST /transcripts": "Create a new transcript",
            "GET /transcripts/{transcript_id}": "Get a transcript",
            "GET /transcripts/{transcript_id}/chats/{chat_id}": "Get a chat"
        },
        "mongo": True
    }

# Health check endpoint
@app.get(
    "/health",
    tags=["Health"],
    summary="Health check",
    description="Returns a heartbeat payload that can be used for readiness and liveness probes.",
)
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# -----------------
# Auth routes
# -----------------
class AuthBody(BaseModel):
    email: str
    password: str


class RegisterBody(AuthBody):
    name: str
    mobile: str


def _hash_password(password: str, salt: Optional[bytes] = None, iterations: int = 200_000) -> str:
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256$%d$%s$%s" % (
        iterations,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(dk).decode("ascii"),
    )


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iters, salt_b64, hash_b64 = encoded.split("$")
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iters)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(hash_b64.encode("ascii"))
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


def _normalize_mobile(mobile: str) -> str:
    digits = re.sub(r"\D", "", mobile or "")
    if not digits:
        return ""
    if mobile.strip().startswith("+"):
        return f"+{digits}"
    return f"+{digits}"


@app.post(
    "/auth/register",
    tags=["Authentication"],
    summary="Register a new user",
    description="Creates a user account with email, password, and profile attributes.",
)
async def register(body: RegisterBody):
    email = (body.email or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")
    if not body.password or len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password too short")
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    mobile = _normalize_mobile(body.mobile)
    if not mobile:
        raise HTTPException(status_code=400, detail="Mobile number is required")
    # Check if exists
    existing = store.users.find_one({"email": email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    if store.users.find_one({"mobile": mobile}):
        raise HTTPException(status_code=400, detail="Mobile number already registered")
    hashed = _hash_password(body.password)
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    doc = {
        "user_id": user_id,
        "email": email,
        "password_hash": hashed,
        "created_at": datetime.now(timezone.utc),
        "name": name,
        "mobile": mobile,
        "dashboard_graphs": [],
    }
    store.users.insert_one(doc)
    token = create_token(user_id, email)
    return {"token": token, "user": {"id": user_id, "email": email, "name": name, "mobile": mobile}}


@app.post(
    "/auth/login",
    tags=["Authentication"],
    summary="Authenticate a user",
    description="Validates credentials and returns a JWT for subsequent requests.",
)
async def login(body: AuthBody):
    email = (body.email or "").strip().lower()
    user = store.users.find_one({"email": email})
    if not user or not _verify_password(body.password or "", user.get("password_hash") or ""):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    uid = user.get("user_id") or str(user.get("_id"))
    token = create_token(uid, email)
    return {
        "token": token,
        "user": {
            "id": uid,
            "email": email,
            "name": user.get("name"),
            "mobile": user.get("mobile"),
        },
    }


@app.get(
    "/auth/me",
    dependencies=[Depends(auth_required)],
    tags=["Authentication"],
    summary="Current user profile",
    description="Returns profile details and dashboard configuration for the authenticated user.",
)
async def auth_me(request: Request):
    user = _require_user(request)
    doc = store.users.find_one({"user_id": user.get("id")}) or store.users.find_one({"email": user.get("email")})
    if not doc:
        raise HTTPException(status_code=404, detail="User not found")
    created_at = doc.get("created_at")
    if isinstance(created_at, datetime):
        created_at = created_at.isoformat()
    user_id = doc.get("user_id") or str(doc.get("_id"))
    graph_metadata = [
        _to_json_safe(graph.to_dict())
        for graph in dashboard_query_engine.list_graphs(user_id, active_only=True)
    ]
    scope_snapshot = _to_json_safe(dashboard_query_engine.get_scope_snapshot(user_id))
    return {
        "user": {
            "id": user_id,
            "email": doc.get("email"),
            "name": doc.get("name"),
            "mobile": doc.get("mobile"),
            "created_at": created_at,
            "dashboard_graphs": graph_metadata,
            "dashboard_graphs_legacy": doc.get("dashboard_graphs") or [],
            "dashboard_scope": scope_snapshot,
        }
    }


@app.get(
    "/dashboard",
    dependencies=[Depends(auth_required)],
    tags=["Dashboard"],
    summary="Get dashboard graphs",
    description="Returns the stored dashboard visualization entries for the authenticated user.",
)
async def get_dashboard(request: Request):
    user = _require_user(request)
    doc = store.users.find_one({"user_id": user.get("id")}) or store.users.find_one({"email": user.get("email")})
    if not doc:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = doc.get("user_id") or str(doc.get("_id"))
    graphs = [_to_json_safe(graph.to_dict()) for graph in dashboard_query_engine.list_graphs(user_id, active_only=True)]
    scope = _to_json_safe(dashboard_query_engine.get_scope_snapshot(user_id))
    return {
        "graphs": graphs,
        "scope": scope,
        "legacy_graphs": doc.get("dashboard_graphs") or [],
    }


@app.put(
    "/dashboard",
    dependencies=[Depends(auth_required)],
    tags=["Dashboard"],
    summary="Replace dashboard layout",
    description="Overwrites the dashboard visualization list with the provided configuration.",
)
async def update_dashboard(payload: DashboardGraphsPayload, request: Request):
    user = _require_user(request)
    graphs_payload: List[Dict[str, Any]] = []
    for item in payload.graphs:
        if isinstance(item, dict):
            graphs_payload.append(item)
        else:
            try:
                graphs_payload.append(item.dict())  # type: ignore[attr-defined]
            except AttributeError:
                graphs_payload.append(dict(item))

    def _sanitize_graph(doc: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in doc.items():
            if key in {"data", "rows", "columns", "tables"}:
                continue
            sanitized[key] = value
        if "id" not in sanitized:
            sanitized["id"] = doc.get("id") or f"graph_{uuid.uuid4().hex[:12]}"
        if isinstance(sanitized.get("query"), str):
            sanitized["query"] = sanitized["query"][:2000]
        if isinstance(sanitized.get("insight"), str):
            sanitized["insight"] = sanitized["insight"][:4000]
        return sanitized

    graphs_payload = [_sanitize_graph(graph) for graph in graphs_payload]

    user_id = user.get("id")

    # Sync dashboard query engine state with sanitized payloads
    new_active_ids: List[str] = []
    for graph_doc in graphs_payload:
        title = graph_doc.get("title") or graph_doc.get("insight") or graph_doc.get("query") or "Pinned visualization"
        payload = {
            "graph_id": graph_doc.get("graph_id") or graph_doc.get("id"),
            "title": title,
            "graph_type": graph_doc.get("graph_type"),
            "data_source": graph_doc.get("data_source"),
            "description": graph_doc.get("insight") or graph_doc.get("description"),
            "metadata": graph_doc.get("metadata") or {},
            "figure": graph_doc.get("figure"),
            "config": graph_doc.get("config"),
            "summary": graph_doc.get("summary"),
            "html_content": graph_doc.get("html") or graph_doc.get("html_content"),
            "active": True,
        }
        try:
            registered = dashboard_query_engine.register_graph(user_id, payload)
            new_active_ids.append(registered.graph_id)
            graph_doc["graph_id"] = registered.graph_id
            graph_doc["id"] = graph_doc.get("id") or registered.graph_id
            graph_doc["title"] = title
        except Exception as register_err:
            logger.warning("Failed to register dashboard graph payload for user %s: %s", user_id, register_err)

    existing_graphs = dashboard_query_engine.list_graphs(user_id, active_only=False)
    keep_set = set(new_active_ids)
    for graph in existing_graphs:
        try:
            dashboard_query_engine.set_graph_active(user_id, graph.graph_id, active=graph.graph_id in keep_set)
        except Exception as toggle_err:
            logger.warning("Failed to update graph %s active state: %s", graph.graph_id, toggle_err)

    result = store.users.update_one(
        {"user_id": user_id},
        {"$set": {"dashboard_graphs": graphs_payload}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"graphs": graphs_payload}

# -----------------
# LiveKit session APIs
# -----------------

@app.post(
    "/livekit/session",
    dependencies=[Depends(auth_required)],
    tags=["LiveKit"],
    summary="Start LiveKit voice session",
    response_model=LiveKitSessionStartResponse,
)
async def create_livekit_session(payload: LiveKitSessionCreateRequest, request: Request):
    user = _require_user(request)
    try:
        session = await livekit_manager.create_session(
            user_id=user["id"],
            display_name=payload.display_name or user.get("name"),
        )
    except LiveKitConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    response = session.to_dict()
    return response


@app.delete(
    "/livekit/session/{session_id}",
    dependencies=[Depends(auth_required)],
    tags=["LiveKit"],
    summary="End LiveKit voice session",
)
async def end_livekit_session(session_id: str, request: Request):
    user = _require_user(request)
    try:
        session = livekit_manager.get_session(session_id)
    except LiveKitSessionNotFound:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != user.get("id"):
        raise HTTPException(status_code=403, detail="Forbidden")
    transcript_id = await livekit_manager.end_session(session_id)
    return {"status": "terminated", "transcript_id": transcript_id}


@app.post(
    "/livekit/session/{session_id}/token",
    dependencies=[Depends(auth_required)],
    response_model=LiveKitTokenResponse,
    tags=["LiveKit"],
    summary="Issue LiveKit viewer token",
)
async def issue_livekit_token(
    session_id: str,
    request: Request,
    payload: LiveKitTokenRequest | None = None,
):
    user = _require_user(request)
    try:
        session = livekit_manager.get_session(session_id)
    except LiveKitSessionNotFound:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != user.get("id"):
        raise HTTPException(status_code=403, detail="Forbidden")
    display_name = payload.display_name if payload else None
    try:
        token, url, room_name, participant_identity, expires_at = livekit_manager.issue_viewer_token(
            session_id=session_id,
            user_id=user["id"],
            display_name=display_name,
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail="Forbidden")
    return LiveKitTokenResponse(
        session_id=session_id,
        room_name=room_name,
        participant_identity=participant_identity,
        token=token,
        url=url,
        expires_at=expires_at,
    )


@app.post(
    "/livekit/session/{session_id}/transcripts",
    tags=["LiveKit"],
    summary="Ingest transcript message from LiveKit agent",
)
async def ingest_livekit_transcript(
    session_id: str,
    payload: LiveKitTranscriptIngest,
    request: Request,
):
    _require_agent(request)
    try:
        livekit_manager.get_session(session_id)
    except LiveKitSessionNotFound:
        raise HTTPException(status_code=404, detail="Session not found")
    timestamp: Optional[datetime.datetime] = None
    if payload.timestamp:
        try:
            ts = datetime.datetime.fromisoformat(payload.timestamp)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            timestamp = ts
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format")
    entry = await livekit_manager.append_transcript(
        session_id=session_id,
        role=payload.role,
        text=payload.text.strip(),
        timestamp=timestamp,
        metadata=payload.metadata or {},
    )
    return {"status": "ok", "entry": entry.to_dict()}


@app.post(
    "/livekit/session/{session_id}/query",
    tags=["LiveKit"],
    summary="Execute database query for LiveKit agent",
)
async def livekit_query(
    session_id: str,
    payload: LiveKitQueryRequest,
    request: Request,
):
    _require_agent(request)
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    try:
        livekit_manager.get_session(session_id)
    except LiveKitSessionNotFound:
        raise HTTPException(status_code=404, detail="Session not found")
    context = payload.context or livekit_manager.build_conversation_context(session_id)
    try:
        result = await asyncio.to_thread(
            processor.process_query_with_smart_visualization,
            question,
            context,
        )
    except Exception as exc:
        logger.error("Failed to process LiveKit query: %s", exc)
        raise HTTPException(status_code=500, detail="Query processing failed")
    return {"result": _to_json_safe(result)}

 

@app.websocket("/livekit/session/{session_id}/stream")
async def livekit_transcript_stream(websocket: WebSocket, session_id: str):
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4401, reason="Missing token")
        return
    payload = verify_token(token)
    if not payload:
        await websocket.close(code=4401, reason="Invalid token")
        return
    user_id = payload.get("sub")
    try:
        session = livekit_manager.get_session(session_id)
    except LiveKitSessionNotFound:
        await websocket.close(code=4404, reason="Session not found")
        return
    if session.user_id != user_id:
        await websocket.close(code=4403, reason="Forbidden")
        return
    try:
        await livekit_manager.register_websocket(session_id, websocket)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await livekit_manager.unregister_websocket(session_id, websocket)
    except Exception as exc:
        logger.warning("LiveKit websocket error for session %s: %s", session_id, exc)
        await livekit_manager.unregister_websocket(session_id, websocket)
        try:
            await websocket.close(code=1011, reason="Internal error")
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
