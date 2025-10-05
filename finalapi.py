
from typing import Dict, Any, List, Optional
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

from fastapi import FastAPI, HTTPException, status, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone, date, time


# Import your existing classes (assuming they're in main.py)
from groqnopenai import (
    SmartNL2SQLProcessor,
    DashboardQueryEngine,
    SilentVoiceInputAdapter,
    SilentVoiceOutputAdapter,
)
from mongo_store import MongoChatStore, new_transcript_id, new_chat_id
from voice_agent import VoiceAgent, OpenAIVoiceInputAdapter, OpenAIVoiceOutputAdapter

OPENAPI_TAGS = [
    {"name": "General", "description": "Utility and informational endpoints for service discovery."},
    {"name": "Health", "description": "Operational health and readiness checks."},
    {"name": "Authentication", "description": "User registration, login, and identity introspection."},
    {"name": "Query", "description": "Natural-language query processing and related data retrieval APIs."},
    {"name": "Transcripts", "description": "Transcript lifecycle management and chat retrieval."},
    {"name": "Dashboard", "description": "User dashboard graph management APIs."},
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
    html_content: str
    graph_type: Optional[str] = None
    file_path: Optional[str] = None

class GraphsResponse(BaseModel):
    transcript_id: str
    chat_id: str
    graphs: List[GraphItem]
    # Back-compat for old clients that expect a single graph
    html_content: Optional[str] = None
    graph_type: Optional[str] = None
    file_path: Optional[str] = None

processor = SmartNL2SQLProcessor()
dashboard_query_engine = DashboardQueryEngine(store=store)
VOICE_ENABLED = os.getenv("VOICE_ENABLED", "true").lower() not in {"false", "0", "no"}

if VOICE_ENABLED:
    try:
        voice_input_adapter = OpenAIVoiceInputAdapter()
        voice_output_adapter = OpenAIVoiceOutputAdapter()
    except Exception as voice_error:
        logger.warning("Falling back to silent voice adapters: %s", voice_error)
        voice_input_adapter = SilentVoiceInputAdapter()
        voice_output_adapter = SilentVoiceOutputAdapter()
else:
    voice_input_adapter = SilentVoiceInputAdapter()
    voice_output_adapter = SilentVoiceOutputAdapter()

voice_agent = VoiceAgent(
    store=store,
    processor=processor,
    dashboard_engine=dashboard_query_engine,
    voice_input=voice_input_adapter,
    voice_output=voice_output_adapter,
)


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


class VoiceSessionStartResponse(BaseModel):
    session_id: str
    greeting: str
    audio: Optional[str] = None


class VoiceAudioChunkRequest(BaseModel):
    audio: str
    end_of_utterance: Optional[bool] = None
    mime_type: Optional[str] = None


class VoiceAudioChunkResponse(BaseModel):
    type: str
    message: str
    transcript: str
    audio: str
    payload: Optional[Dict[str, Any]] = None

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


def _to_bson_safe(obj: Any) -> Any:
    """Recursively convert Python objects to Mongo/BSON-safe values.
    - Decimal -> Decimal128 to preserve precision
    - Lists/Dicts traversed recursively
    """
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


def _build_html_card_html(description: str, title: Optional[str] = None) -> str:
    safe_title = (title or "Insight").strip() or "Insight"
    safe_desc = (description or "").strip() or "No additional details provided."
    return f"""
<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{safe_title}</title>
    <style>
      :root {{ --bg:#0b1220; --card:#0f172a; --fg:#e2e8f0; --muted:#94a3b8; }}
      html,body{{margin:0;padding:0;background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,Segoe UI,Inter,Roboto,sans-serif}}
      .wrap{{padding:16px}}
      .card{{background:linear-gradient(180deg,rgba(20,184,166,.08),rgba(59,130,246,.08));border:1px solid rgba(255,255,255,.08);border-radius:14px;box-shadow:0 8px 24px rgba(0,0,0,.25);overflow:hidden}}
      .header{{display:flex;align-items:center;gap:10px;padding:14px 16px;border-bottom:1px solid rgba(255,255,255,.06);background:rgba(255,255,255,.02)}}
      .dot{{width:8px;height:8px;border-radius:9999px;background:#14b8a6;box-shadow:0 0 0 2px rgba(20,184,166,.25)}}
      .title{{font-size:14px;font-weight:600;letter-spacing:.2px}}
      .body{{padding:18px 16px 20px;line-height:1.6;font-size:14px}}
      .muted{{color:var(--muted)}}
    </style>
  </head>
  <body>
    <div class=\"wrap\"><div class=\"card\">
      <div class=\"header\"><div class=\"dot\"></div><div class=\"title\">{safe_title}</div></div>
      <div class=\"body\"><div>{safe_desc}</div></div>
    </div></div>
  </body>
</html>
"""


def _derive_graph_title_from_context(ctx: Dict[str, Any], fallback_prefix: str, index: int) -> str:
    html_blob = ctx.get("html") or ""
    if html_blob:
        match = re.search(r"<title>(.*?)</title>", html_blob, flags=re.IGNORECASE | re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate[:160]
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
            payloads.append({
                "title": title,
                "graph_type": ctx.get("chart_type") or result.get("graph_type"),
                "data_source": ctx.get("file_path"),
                "data": _match_data_for_context(ctx, result),
                "description": ctx.get("insight"),
                "html_content": ctx.get("html") or _load_graph_html(ctx.get("file_path")),
                "metadata": {
                    "query": ctx.get("query"),
                    "insight": ctx.get("insight"),
                    "file_path": ctx.get("file_path"),
                    "transcript_id": transcript_id,
                    "chat_id": chat_id,
                    "user_query": user_query,
                },
                "active": True,
            })
    elif result.get("execution_results"):
        title = (result.get("description") or result.get("user_request") or user_query or "Visualization").strip()
        payloads.append({
            "title": title[:160],
            "graph_type": result.get("graph_type"),
            "data_source": result.get("graph_file"),
            "data": result.get("execution_results"),
            "description": result.get("description"),
            "html_content": result.get("graph_html") or _load_graph_html(result.get("graph_file")),
            "metadata": {
                "query": result.get("executed_sub_query") or result.get("user_request") or user_query,
                "transcript_id": transcript_id,
                "chat_id": chat_id,
            },
            "active": True,
        })
    return payloads


def _load_graph_html(file_path: Optional[str]) -> Optional[str]:
    if not file_path:
        return None
    candidate = file_path
    if not os.path.isabs(candidate):
        candidate = os.path.abspath(os.path.join(os.getcwd(), candidate))
    try:
        with open(candidate, "r", encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        return None

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
        viz_list = result.get('visualizations') or []
        if isinstance(viz_list, dict):
            viz_list = [viz_list]

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

        def _append_context(chart_type: Optional[str], file_path: Optional[str], html: Optional[str], query: Optional[str], insight: Optional[str], sub_index: Optional[int] = None, title: Optional[str] = None) -> None:
            graph_contexts.append({
                "chart_type": chart_type,
                "file_path": file_path,
                "html": html,
                "query": query,
                "insight": insight,
                "sub_query_index": sub_index,
                "title": title,
            })

        if isinstance(viz_list, list) and viz_list:
            for viz in viz_list:
                if not isinstance(viz, dict):
                    continue
                file_path = viz.get('graph_file') or viz.get('file') or viz.get('file_path')
                chart_type = viz.get('graph_type') or viz.get('type')
                html_inline = viz.get('html') or viz.get('html_content')
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
                _append_context(chart_type, file_path, html_inline, base_query, insight_text, sub_index=sub_idx, title=viz.get('title'))

        if not graph_contexts and result.get('graph_file'):
            base_query = result.get('executed_sub_query') or result.get('original_query') or request.natural_language_query
            insight_text = result.get('description') or description_text
            _append_context(result.get('graph_type'), result.get('graph_file'), result.get('graph_html'), base_query, insight_text, sub_index=None, title=None)

        if graph_contexts:
            for ctx in graph_contexts:
                if not ctx.get('insight'):
                    ctx['insight'] = description_text or ctx.get('query') or request.natural_language_query

        for ctx in graph_contexts:
            content_items.append({
                "type": "graph",
                "payload": {
                    "chart_type": ctx.get('chart_type'),
                    "file_path": ctx.get('file_path'),
                    "html": ctx.get('html'),
                    "query": ctx.get('query'),
                    "insight": ctx.get('insight'),
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
            graph_type = payload.get("chart_type")
            html_inline = payload.get("html") or payload.get("html_content")
            graph_file = payload.get("file_path")
            if html_inline:
                graphs.append(GraphItem(html_content=html_inline, graph_type=graph_type, file_path=graph_file))
                continue
            if graph_file:
                try:
                    with open(graph_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    graphs.append(GraphItem(html_content=html_content, graph_type=graph_type, file_path=graph_file))
                except FileNotFoundError:
                    # Skip missing files rather than failing entire response
                    continue
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error reading graph file: {str(e)}")
    if not graphs:
        # Fallback to html_card from description so we never 404
        desc_text = ""
        for item in chat.get("content", []):
            if item.get("type") == "description":
                desc_text = (item.get("payload") or {}).get("text") or ""
                break
        html_card = _build_html_card_html(desc_text or "No visual required.")
        graphs = [GraphItem(html_content=html_card, graph_type="html_card", file_path=None)]
    # Build response with optional single-graph fields for older clients
    resp = GraphsResponse(transcript_id=transcript_id, chat_id=chat_id, graphs=graphs)
    if len(graphs) == 1:
        resp.html_content = graphs[0].html_content
        resp.graph_type = graphs[0].graph_type
        resp.file_path = graphs[0].file_path
    return resp


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


@app.post(
    "/voice/sessions",
    response_model=VoiceSessionStartResponse,
    dependencies=[Depends(auth_required)],
    tags=["Voice"],
    summary="Start a voice session",
    description="Initialise a voice-driven analytics session and return a session identifier.",
)
async def start_voice_session(http: Request = None):
    user = _require_user(http)
    user_id = user.get("id")
    session = voice_agent.start_session(user_id)
    user_doc = store.users.find_one({"user_id": user_id})
    user_name = user_doc.get("name") if user_doc else None
    greeting = f"Hello {user_name or 'there'}, I'm ready to help with your analytics."
    audio_b64: Optional[str] = None
    try:
        audio_bytes = voice_output_adapter.synthesize(greeting)
        if audio_bytes:
            audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    except NotImplementedError:
        audio_b64 = None
    except Exception as synth_err:
        logger.warning("Voice greeting synthesis failed: %s", synth_err)
        audio_b64 = None
    return VoiceSessionStartResponse(session_id=session.session_id, greeting=greeting, audio=audio_b64)


@app.post(
    "/voice/sessions/{session_id}/audio",
    response_model=VoiceAudioChunkResponse,
    dependencies=[Depends(auth_required)],
    tags=["Voice"],
    summary="Stream voice audio",
    description="Send a chunk of base64-encoded PCM audio; receives spoken response when end-of-utterance is detected.",
)
async def stream_voice_audio(session_id: str, payload: VoiceAudioChunkRequest, http: Request = None):
    user = _require_user(http)
    user_id = user.get("id")
    try:
        session = voice_agent.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Voice session not found")
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Voice session does not belong to the current user")

    try:
        response_payload = voice_agent.ingest_audio(session_id, payload.audio, payload.end_of_utterance, payload.mime_type)
    except ValueError as audio_err:
        raise HTTPException(status_code=400, detail=str(audio_err))

    if not response_payload:
        return VoiceAudioChunkResponse(
            type="listening",
            message="Listening...",
            transcript="",
            audio="",
        )

    return VoiceAudioChunkResponse(
        type=response_payload.get("type", "analytics"),
        message=response_payload.get("message", ""),
        transcript=response_payload.get("transcript", ""),
        audio=response_payload.get("audio", ""),
        payload=_to_json_safe(response_payload.get("payload")),
    )


@app.delete(
    "/voice/sessions/{session_id}",
    dependencies=[Depends(auth_required)],
    tags=["Voice"],
    summary="End a voice session",
    description="Terminate a voice session and release buffered audio.",
)
async def end_voice_session(session_id: str, http: Request = None):
    user = _require_user(http)
    user_id = user.get("id")
    try:
        session = voice_agent.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Voice session not found")
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Voice session does not belong to the current user")
    voice_agent.end_session(session_id)
    return {"status": "ended", "session_id": session_id}

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
            "GET /get_graph/{transcript_id}/{chat_id}": "Get graph HTML content",
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
            if key in {"html", "html_content", "svg", "static_html"} and isinstance(value, str):
                trimmed = value[:20000]
                sanitized["html"] = trimmed
                sanitized["html_content"] = trimmed
            elif key in {"data", "rows", "columns", "tables"}:
                continue
            else:
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
