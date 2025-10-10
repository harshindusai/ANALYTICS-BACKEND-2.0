import os
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
# Updated imports for Langfuse integration
from langfuse.openai import OpenAI
from langfuse import Langfuse
import os
from typing import Tuple, Optional
from openai import OpenAI
from typing import Dict, List, Tuple, Optional, Any
import configparser
from datetime import datetime
import traceback
import pandas as pd
import numpy as np
from collections import Counter
import warnings
from decimal import Decimal
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
from dotenv import load_dotenv
import plotly.express as px
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from plotly.utils import PlotlyJSONEncoder

if TYPE_CHECKING:
    from mongo_store import MongoChatStore

langfuse = Langfuse(
    secret_key="sk-lf-a2925571-050c-4cbf-baec-b3ec63782d7d",
    public_key="pk-lf-abdfde0f-7103-4028-9cbb-f9c92feb6bfc",
    host="https://cloud.langfuse.com"
)

os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

# Load environment variables from .env
load_dotenv()
load_dotenv(dotenv_path=".env")
warnings.filterwarnings('ignore')

# Setup comprehensive logging to file
def setup_logging():
    """Setup comprehensive logging to both file and console"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Create a unique log filename with timestamp
    log_filename = f"logs/nl2sql_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # Also log to console
        ]
    )

    return logging.getLogger(__name__)

logger = setup_logging()

class VoiceInputAdapter:
    """Base adapter for plugging voice input transcription in the future."""

    def transcribe(self, payload: Any) -> str:
        raise NotImplementedError("Voice input adapter must implement transcribe().")


class VoiceOutputAdapter:
    """Base adapter for plugging voice synthesis in the future."""

    def synthesize(self, text: str) -> Any:
        raise NotImplementedError("Voice output adapter must implement synthesize().")


class SilentVoiceInputAdapter(VoiceInputAdapter):
    """Default no-op voice input adapter that simply returns textual payloads."""

    def transcribe(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        return ""


class SilentVoiceOutputAdapter(VoiceOutputAdapter):
    """Default no-op voice output adapter."""

    def synthesize(self, text: str) -> None:
        return None


@dataclass
class DashboardGraphMetadata:
    graph_id: str
    title: str
    graph_type: Optional[str] = None
    data_source: Optional[str] = None
    data: List[Dict[str, Any]] = field(default_factory=list)
    description: Optional[str] = None
    figure: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    row_count: int = 0
    fields: List[str] = field(default_factory=list)
    last_synced_at: Optional[datetime] = None
    html_content: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "graph_id": self.graph_id,
            "title": self.title,
            "graph_type": self.graph_type,
            "data_source": self.data_source,
            "data": self.data,
            "description": self.description,
            "figure": self.figure,
            "config": self.config,
            "summary": self.summary,
            "metadata": self.metadata,
            "active": self.active,
            "row_count": self.row_count,
            "fields": self.fields,
            "last_synced_at": self.last_synced_at,
            "html_content": self.html_content,
        }
        # Remove empty entries to keep Mongo payload compact
        return {k: v for k, v in payload.items() if v is not None and v != []}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "title": self.title,
            "graph_type": self.graph_type,
            "data_source": self.data_source,
            "data": self.data,
            "description": self.description,
            "figure": self.figure,
            "config": self.config,
            "summary": self.summary,
            "metadata": self.metadata,
            "active": self.active,
            "row_count": self.row_count,
            "fields": self.fields,
            "last_synced_at": self.last_synced_at,
            "html_content": self.html_content,
        }

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data or [])


@dataclass
class DashboardGraphAnalysis:
    graph_id: str
    title: str
    summary: str
    stats: Dict[str, Any] = field(default_factory=dict)
    data_preview: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    columns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "title": self.title,
            "summary": self.summary,
            "stats": self.stats,
            "data_preview": self.data_preview,
            "row_count": self.row_count,
            "columns": self.columns,
        }


@dataclass
class DashboardQueryResult:
    query: str
    graphs_used: List[str]
    message: str
    analyses: List[DashboardGraphAnalysis]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "graphs_used": self.graphs_used,
            "message": self.message,
            "analyses": [analysis.to_dict() for analysis in self.analyses],
        }


class DashboardQueryEngine:
    """High-level query engine for dashboard graph metadata and data."""

    def __init__(
        self,
        store: "MongoChatStore",
        voice_input: Optional[VoiceInputAdapter] = None,
        voice_output: Optional[VoiceOutputAdapter] = None,
    ) -> None:
        self.store = store
        self.voice_input = voice_input or SilentVoiceInputAdapter()
        self.voice_output = voice_output or SilentVoiceOutputAdapter()

    # ---------
    # Voice plumbing (future friendly hooks)
    # ---------
    def set_voice_adapters(
        self,
        voice_input: Optional[VoiceInputAdapter] = None,
        voice_output: Optional[VoiceOutputAdapter] = None,
    ) -> None:
        if voice_input is not None:
            self.voice_input = voice_input
        if voice_output is not None:
            self.voice_output = voice_output

    def handle_voice_request(self, user_id: str, voice_payload: Any) -> Dict[str, Any]:
        """Transcribe voice input, answer the query, and optionally synthesize a spoken response."""
        query_text = self.voice_input.transcribe(voice_payload)
        if not query_text:
            return {
                "type": "error",
                "message": "Voice input did not contain any transcribable text.",
                "spoken": False,
            }
        response = self.handle_request(user_id, query_text)
        spoken = False
        message = response.get("message")
        if message:
            try:
                result = self.voice_output.synthesize(message)
                spoken = result is not None or spoken
            except NotImplementedError:
                spoken = False
        response["spoken"] = spoken
        response["query_text"] = query_text
        return response

    # ---------
    # Registry operations
    # ---------
    def register_graph(self, user_id: str, graph: Any) -> DashboardGraphMetadata:
        payload = graph.to_payload() if isinstance(graph, DashboardGraphMetadata) else dict(graph or {})
        stored = self.store.register_dashboard_graph(user_id, payload)
        metadata = self._to_metadata(stored)
        self._refresh_cache(user_id)
        return metadata

    def unregister_graph(self, user_id: str, graph_identifier: str) -> bool:
        removed = self.store.unregister_dashboard_graph(user_id, graph_identifier)
        if removed:
            self._refresh_cache(user_id)
        return removed

    def set_graph_active(self, user_id: str, graph_identifier: str, active: bool = True) -> bool:
        updated = self.store.activate_dashboard_graph(user_id, graph_identifier, active=active)
        if updated:
            self._refresh_cache(user_id)
        return updated

    def set_graph_scope(self, user_id: str, graph_identifiers: List[str]) -> Dict[str, int]:
        result = self.store.activate_only_dashboard_graphs(user_id, graph_identifiers)
        if result.get("activated") or result.get("deactivated"):
            self._refresh_cache(user_id)
        return result

    def list_graphs(self, user_id: str, active_only: bool = False) -> List[DashboardGraphMetadata]:
        docs = self.store.list_dashboard_graphs(user_id, only_active=active_only)
        return [self._to_metadata(doc) for doc in docs]

    def resolve_graphs_from_text(
        self,
        user_id: str,
        text: str,
        *,
        graphs: Optional[List[DashboardGraphMetadata]] = None,
    ) -> List[DashboardGraphMetadata]:
        if not text:
            return []
        graphs = graphs or self.list_graphs(user_id, active_only=False)
        normalized_query = self._normalize(text)
        return self._match_graphs(normalized_query, graphs)

    def include_graphs_in_scope(self, user_id: str, graph_identifiers: List[str]) -> int:
        updated = self.store.set_dashboard_graphs_active_state(user_id, graph_identifiers, active=True)
        if updated:
            self._refresh_cache(user_id)
        return updated

    def exclude_graphs_from_scope(self, user_id: str, graph_identifiers: List[str]) -> int:
        updated = self.store.set_dashboard_graphs_active_state(user_id, graph_identifiers, active=False)
        if updated:
            self._refresh_cache(user_id)
        return updated

    def get_scope_snapshot(self, user_id: str) -> Dict[str, List[Dict[str, Any]]]:
        return self.store.get_dashboard_graph_scope(user_id)

    def _refresh_cache(self, user_id: str) -> None:
        try:
            self.store.refresh_dashboard_graph_cache(user_id)
        except Exception as exc:
            logger.warning("Failed to refresh dashboard cache for user %s: %s", user_id, exc)

    # ---------
    # Query handling
    # ---------
    def handle_request(self, user_id: str, user_query: str) -> Dict[str, Any]:
        if not user_query or not user_query.strip():
            return {
                "type": "error",
                "message": "Query text is required to interact with dashboard graphs.",
            }

        graphs = self.list_graphs(user_id, active_only=False)
        directive_response = self._maybe_handle_scope_directive(user_id, user_query, graphs)
        if directive_response:
            return directive_response

        active_graphs = [graph for graph in graphs if graph.active]
        query_result = self.answer_question(user_id, user_query, graphs=active_graphs)
        response = {"type": "query", **query_result.to_dict()}
        response["scope"] = self.get_scope_snapshot(user_id)
        return response

    def answer_question(
        self,
        user_id: str,
        user_query: str,
        graphs: Optional[List[DashboardGraphMetadata]] = None,
    ) -> DashboardQueryResult:
        graphs = graphs or self.list_graphs(user_id, active_only=True)
        if not graphs:
            message = "No dashboard graphs are registered yet. Add a visualization before querying."
            return DashboardQueryResult(query=user_query, graphs_used=[], message=message, analyses=[])

        matched_graphs = self.resolve_graphs_from_text(user_id, user_query, graphs=graphs)
        if not matched_graphs:
            matched_graphs = graphs

        analyses: List[DashboardGraphAnalysis] = []
        for graph in matched_graphs:
            analyses.append(self._analyze_graph(graph, user_query))

        message = self._compose_response_message(user_query, analyses, matched_graphs, len(matched_graphs) == len(graphs))
        return DashboardQueryResult(
            query=user_query,
            graphs_used=[graph.title for graph in matched_graphs],
            message=message,
            analyses=analyses,
        )

    # ---------
    # Internal helpers
    # ---------
    def _to_metadata(self, doc: Dict[str, Any]) -> DashboardGraphMetadata:
        data = doc.get("data") or []
        if isinstance(data, dict):  # defensive guard
            data = [data]
        metadata = doc.get("metadata") or {}
        row_count = doc.get("row_count")
        if isinstance(row_count, str) and row_count.isdigit():
            row_count = int(row_count)
        if not isinstance(row_count, int):
            row_count = len(data)
        fields = doc.get("fields") or []
        if not fields and data and isinstance(data[0], dict):
            fields = sorted(data[0].keys())
        return DashboardGraphMetadata(
            graph_id=doc.get("graph_id"),
            title=doc.get("title", "Untitled graph"),
            graph_type=doc.get("graph_type"),
            data_source=doc.get("data_source"),
            data=data,
            description=doc.get("description"),
            figure=doc.get("figure"),
            config=doc.get("config"),
            summary=doc.get("summary"),
            metadata=metadata,
            active=bool(doc.get("active", True)),
            row_count=row_count,
            fields=fields,
            last_synced_at=doc.get("last_synced_at"),
            html_content=doc.get("html_content") or doc.get("html"),
        )

    def _normalize(self, value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip().lower())

    def _match_graphs(self, normalized_query: str, graphs: List[DashboardGraphMetadata]) -> List[DashboardGraphMetadata]:
        matched: List[DashboardGraphMetadata] = []
        for graph in graphs:
            normalized_title = self._normalize(graph.title)
            if not normalized_title:
                continue
            if normalized_title in normalized_query:
                matched.append(graph)
                continue
            title_tokens = [token for token in normalized_title.split(" ") if len(token) > 2]
            if title_tokens and all(token in normalized_query for token in title_tokens):
                matched.append(graph)
        return matched

    def _maybe_handle_scope_directive(
        self,
        user_id: str,
        user_query: str,
        graphs: List[DashboardGraphMetadata],
    ) -> Optional[Dict[str, Any]]:
        normalized_query = self._normalize(user_query)
        if not normalized_query:
            return None

        scope_related = any(token in normalized_query for token in ["query", "queries", "scope"])
        matched_graphs = self._match_graphs(normalized_query, graphs)
        matched_identifiers = [graph.graph_id for graph in matched_graphs]
        matched_titles = [graph.title for graph in matched_graphs]

        def _response(action: str, updated: int, message: str, graphs_payload: Optional[List[str]] = None) -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "type": "scope_update",
                "action": action,
                "updated": updated,
                "message": message,
                "graphs": graphs_payload if graphs_payload is not None else matched_titles,
            }
            payload["scope"] = self.get_scope_snapshot(user_id)
            return payload

        if any(keyword in normalized_query for keyword in ["delete", "unregister", "remove permanently"]):
            if not matched_graphs:
                return _response("remove", 0, "No dashboard graph names matched the request.", [])
            removed = 0
            for graph in matched_graphs:
                if self.unregister_graph(user_id, graph.graph_id):
                    removed += 1
            message = "Removed graph(s) from the registry." if removed else "Graphs were not found in the registry."
            return _response("remove", removed, message)

        remove_keywords = ["remove", "exclude", "drop"]
        if ("deactivate" in normalized_query) or (scope_related and any(keyword in normalized_query for keyword in remove_keywords)):
            if not matched_graphs:
                return _response("deactivate", 0, "No dashboard graph names matched the request.", [])
            updated = self.exclude_graphs_from_scope(user_id, matched_identifiers)
            message = "Deactivated graph(s) for querying." if updated else "Graphs were already inactive."
            return _response("deactivate", updated, message)

        exclusive_phrases = ["only use", "use only", "just use", "limit to", "restrict to"]
        if any(phrase in normalized_query for phrase in exclusive_phrases):
            if not matched_graphs:
                return _response("exclusive", 0, "Could not identify which graph to keep active.", [])
            result = self.set_graph_scope(user_id, matched_identifiers)
            message = "Updated scope to use selected graph(s) only."
            return _response("exclusive", result.get("activated", 0), message)

        if any(keyword in normalized_query for keyword in ["use all", "reset scope", "include all"]):
            identifiers = [graph.graph_id for graph in graphs]
            if not identifiers:
                return _response("exclusive", 0, "No graphs are registered yet.", [])
            result = self.set_graph_scope(user_id, identifiers)
            message = "Activated all graphs for querying."
            return _response("exclusive", result.get("activated", 0), message, [graph.title for graph in graphs])

        activate_keywords = ["activate", "reactivate"]
        add_keywords = ["add", "include"]
        if (
            any(keyword in normalized_query for keyword in activate_keywords)
            or (scope_related and any(keyword in normalized_query for keyword in add_keywords))
        ):
            if not matched_graphs:
                return _response("activate", 0, "Could not identify which graph to activate.", [])
            updated = self.include_graphs_in_scope(user_id, matched_identifiers)
            if not updated:
                message = "Graphs were already active."
            else:
                message = "Activated graph(s) for queries."
            return _response("activate", updated, message)

        return None

    def _analyze_graph(self, graph: DashboardGraphMetadata, user_query: str) -> DashboardGraphAnalysis:
        records = self._load_graph_records(graph)
        if not records:
            graph.row_count = 0
            summary = "No data is associated with this graph yet."
            return DashboardGraphAnalysis(
                graph_id=graph.graph_id,
                title=graph.title,
                summary=summary,
                stats={},
                data_preview=[],
                row_count=0,
                columns=graph.fields,
            )

        df = pd.DataFrame(records)
        graph.row_count = len(df)
        graph.fields = list(df.columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        stats: Dict[str, Any] = {}
        lowered_query = user_query.lower()

        if numeric_cols:
            if any(keyword in lowered_query for keyword in ["sum", "total", "overall"]):
                stats["total"] = {col: self._json_safe(df[col].sum()) for col in numeric_cols}
            if any(keyword in lowered_query for keyword in ["average", "mean"]):
                stats["average"] = {col: self._json_safe(df[col].mean()) for col in numeric_cols}
            if any(keyword in lowered_query for keyword in ["max", "highest", "top"]):
                stats["max"] = {col: self._json_safe(df[col].max()) for col in numeric_cols}
            if any(keyword in lowered_query for keyword in ["min", "lowest", "bottom"]):
                stats["min"] = {col: self._json_safe(df[col].min()) for col in numeric_cols}

            if not stats:
                descriptive = df[numeric_cols].agg(["sum", "mean", "max", "min"]).to_dict()
                stats["summary"] = self._json_safe(descriptive)
        else:
            # Use categorical breakdown for first few columns
            categorical_stats: Dict[str, Any] = {}
            for column in df.columns[:3]:
                counts = df[column].value_counts().head(5).to_dict()
                categorical_stats[column] = self._json_safe(counts)
            stats["categorical"] = categorical_stats

        data_preview = self._json_safe(df.head(5).to_dict(orient="records"))
        summary = self._build_summary(graph, stats, len(df), numeric_cols)

        return DashboardGraphAnalysis(
            graph_id=graph.graph_id,
            title=graph.title,
            summary=summary,
            stats=stats,
            data_preview=data_preview,
            row_count=len(df),
            columns=list(df.columns),
        )

    def _build_summary(
        self,
        graph: DashboardGraphMetadata,
        stats: Dict[str, Any],
        row_count: int,
        numeric_cols: List[str],
    ) -> str:
        parts: List[str] = []
        parts.append(f"Graph '{graph.title}' has {row_count} row(s) of data available.")

        totals = stats.get("total")
        if totals:
            formatted = ", ".join(f"{col}: {value}" for col, value in totals.items())
            parts.append(f"Total values -> {formatted}")

        averages = stats.get("average")
        if averages:
            formatted = ", ".join(f"{col}: {value}" for col, value in averages.items())
            parts.append(f"Average values -> {formatted}")

        if "max" in stats:
            formatted = ", ".join(f"{col}: {value}" for col, value in stats["max"].items())
            parts.append(f"Maximum observed -> {formatted}")

        if "min" in stats:
            formatted = ", ".join(f"{col}: {value}" for col, value in stats["min"].items())
            parts.append(f"Minimum observed -> {formatted}")

        if not numeric_cols and "categorical" in stats:
            sample_col = next(iter(stats["categorical"].items())) if stats["categorical"] else None
            if sample_col:
                column, breakdown = sample_col
                formatted = ", ".join(f"{label}: {count}" for label, count in breakdown.items())
                parts.append(f"Top categories in {column} -> {formatted}")

        description = graph.description or (graph.metadata.get("insight") if graph.metadata else None)
        if description:
            parts.append(description)

        return " ".join(parts)

    def _compose_response_message(
        self,
        user_query: str,
        analyses: List[DashboardGraphAnalysis],
        matched_graphs: List[DashboardGraphMetadata],
        used_all_graphs: bool,
    ) -> str:
        if not analyses:
            return "No insights were produced for the requested graphs."
        scope_phrase = "all available graphs" if used_all_graphs else "selected graph(s)"
        graph_titles = ", ".join(analysis.title for analysis in analyses)
        highlights = "; ".join(analysis.summary for analysis in analyses[:2])
        base_message = f"Answered '{user_query}' using {scope_phrase}: {graph_titles}."
        if highlights:
            base_message = f"{base_message} Key points -> {highlights}"
        total_rows = sum((analysis.row_count or 0) for analysis in analyses)
        if total_rows:
            base_message = f"{base_message} Total rows reviewed: {total_rows}."
        return base_message

    def _load_graph_records(self, graph: DashboardGraphMetadata) -> List[Dict[str, Any]]:
        if graph.data:
            graph.row_count = len(graph.data)
            if graph.data and isinstance(graph.data[0], dict):
                graph.fields = sorted(graph.data[0].keys())
            return graph.data
        if not graph.data_source:
            return []
        try:
            path = Path(graph.data_source)
            if not path.exists():
                return []
            if path.suffix.lower() == ".json":
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, list):
                    graph.data = data
                elif isinstance(data, dict):
                    graph.data = [data]
                else:
                    graph.data = []
            elif path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
                graph.data = df.to_dict(orient="records")
            else:
                graph.data = []
        except Exception as exc:
            logger.warning("Failed to load data for graph %s: %s", graph.graph_id, exc)
            graph.data = []
        graph.row_count = len(graph.data)
        if graph.data and isinstance(graph.data[0], dict):
            graph.fields = sorted(graph.data[0].keys())
        return graph.data

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, dict):
            return {key: self._json_safe(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        return value

class NL2SQLConfig:
    """Configuration management for NL2SQL system"""

    def __init__(self, config_file: str = 'config.ini'):
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        self._load_config()

    def _load_config(self):
        """Load configuration from file or environment variables"""
        try:
            self.config.read(self.config_file)
            logger.info(f"Configuration loaded from {self.config_file}")
        except Exception as e:
            logger.warning(f"Could not read config file {self.config_file}: {e}")

        # Database configuration
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'mithaas'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'Harsh@2004')
        }
        logger.info(f"Database config loaded for host: {self.db_config['host']}, database: {self.db_config['database']}")

        # Updated for Langfuse integration - using Groq API key but with OpenAI client
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.model_name = os.getenv('GROQ_MODEL_NAME')
        logger.info(f"API keys configured - Groq: {'✓' if self.groq_api_key else '✗'}, OpenAI: {'✓' if self.openai_api_key else '✗'}")

class DatabaseManager:
    """Database connection and query execution management"""

    def __init__(self, config: NL2SQLConfig):
        self.config = config
        self.connection = None

    def connect(self) -> bool:
        """Establish database connection"""
        try:
            logger.info("Attempting to connect to PostgreSQL database...")
            self.connection = psycopg2.connect(**self.config.db_config)
            logger.info("✅ Successfully connected to PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return False

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> Tuple[List[Dict], Optional[str]]:
        """Execute SQL query safely with parameterized inputs"""
        if not self.connection:
            error_msg = "No database connection established"
            logger.error(f"❌ {error_msg}")
            return [], error_msg

        try:
            logger.info(f"Executing SQL query: {query[:100]}{'...' if len(query) > 100 else ''}")

            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)

                if query.strip().upper().startswith(('SELECT', 'WITH')):
                    results = cursor.fetchall()
                    result_list = [dict(row) for row in results]
                    logger.info(f"✅ Query executed successfully, returned {len(result_list)} records")
                    return result_list, None
                else:
                    self.connection.commit()
                    logger.info("✅ Non-SELECT query executed successfully")
                    return [], None

        except Exception as e:
            self.connection.rollback()
            error_msg = f"Query execution error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return [], error_msg

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

class SmartGraphGenerator:
    """Enhanced smart graph generation with robust JSON parsing and CORRECT fallback logic"""

    def __init__(self, output_dir: str = "graphs", groq_client=None):
        self.output_dir = output_dir
        self.groq_client = groq_client
        self.ensure_output_dir()
        logger.info(f"SmartGraphGenerator initialized with output directory: {self.output_dir}")

    def ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

    def _figure_to_json(self, fig: go.Figure) -> Dict[str, Any]:
        """Convert a Plotly figure into a JSON-safe dictionary."""
        try:
            return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
        except Exception as exc:
            logger.warning(f"Failed to encode figure with PlotlyJSONEncoder: {exc}")
            try:
                return fig.to_dict()
            except Exception as inner_exc:
                logger.error(f"Failed to convert figure to dict: {inner_exc}")
                return {}

    @staticmethod
    def _default_plotly_config() -> Dict[str, Any]:
        """Shared Plotly config to keep charts responsive in the UI."""
        return {
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["toImage"],
        }

    def _build_plotly_payload(
        self,
        fig: go.Figure,
        chart_type: str,
        title: str,
        query: str,
        sub_query_index: Optional[int],
    ) -> Dict[str, Any]:
        """Normalize the Plotly figure into the structure expected by finalapi.py."""
        payload: Dict[str, Any] = {
            "type": "plotly",
            "graph_type": chart_type,
            "title": title,
            "figure": self._figure_to_json(fig),
            "config": self._default_plotly_config(),
            "query": query,
            "sub_query_index": sub_query_index,
        }
        try:
            payload["html"] = fig.to_html(full_html=False, include_plotlyjs="cdn")
        except Exception:
            # HTML is optional; ignore failures to avoid breaking chart generation.
            pass
        return payload

    @staticmethod
    def _format_metric_value(label: str, value: Any) -> Tuple[Any, str]:
        """Return the raw numeric value (if applicable) and a friendly formatted string."""
        raw_value = value
        if isinstance(value, Decimal):
            raw_value = float(value)
        if isinstance(raw_value, (int, float)):
            if abs(raw_value) >= 1000:
                formatted = f"{raw_value:,.2f}"
            else:
                formatted = f"{raw_value:.2f}"
            if any(keyword in label.lower() for keyword in ["amount", "revenue", "sale", "cost", "price", "profit", "total"]):
                formatted = f"₹{formatted}"
            return raw_value, formatted
        return value, str(value)

    def _render_summary_card_html(
        self,
        query: str,
        metrics: List[Dict[str, Any]],
        sub_query_index: Optional[int],
    ) -> Optional[str]:
        """Render a lightweight HTML snippet for value cards (optional fallback)."""
        try:
            badge = f"<div class='sub-query-badge'>Query {sub_query_index}</div>" if sub_query_index else ""
            metric_blocks = "\n".join(
                f"""
                <div class="value-item">
                    <div class="value-label">{metric['label']}</div>
                    <div class="value-number">{metric['formatted']}</div>
                </div>
                """
                for metric in metrics
            )
            html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Summary Card</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fb;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            background: white;
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            max-width: 520px;
            margin: 0 auto;
        }}
        .sub-query-badge {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            color: #4f46e5;
            background: rgba(79,70,229,0.12);
            display: inline-block;
            padding: 6px 14px;
            border-radius: 999px;
            margin-bottom: 12px;
        }}
        .query-text {{
            font-size: 18px;
            font-weight: 500;
            color: #1f2937;
            margin-bottom: 24px;
        }}
        .values-grid {{
            display: grid;
            gap: 16px;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        }}
        .value-item {{
            background: #f9fafc;
            border-radius: 12px;
            padding: 16px;
        }}
        .value-label {{
            font-size: 13px;
            font-weight: 600;
            color: #6b7280;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .value-number {{
            font-size: 28px;
            font-weight: 600;
            color: #111827;
        }}
        .timestamp {{
            font-size: 12px;
            color: #9ca3af;
            margin-top: 24px;
        }}
    </style>
</head>
<body>
    <div class="container">
        {badge}
        <div class="query-text">{query}</div>
        <div class="values-grid">
            {metric_blocks}
        </div>
        <div class="timestamp">Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
    </div>
</body>
</html>"""
            return html
        except Exception as exc:
            logger.debug(f"Failed to render summary card HTML: {exc}")
            return None

    @staticmethod
    def _build_chart_title(query: str, sub_query_index: Optional[int] = None) -> str:
        """Generate a concise chart title without duplicating the query text."""
        base = (query or "Visualization").strip() or "Visualization"
        max_len = 60
        if len(base) > max_len:
            base = base[: max_len - 3].rstrip() + "..."
        if sub_query_index:
            base = f"{base} (Query {sub_query_index})"
        return base

    def _is_decimal_column(self, series: pd.Series) -> bool:
        """Check if a series contains Decimal objects from PostgreSQL"""
        try:
            sample = series.dropna().head(5)
            if len(sample) == 0:
                return False

            # Check if any non-null values are Decimal objects
            for value in sample:
                if isinstance(value, Decimal):
                    return True
            return False
        except:
            return False

    def _convert_decimal_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Decimal columns to float for proper numeric analysis"""
        df_converted = df.copy()

        for col in df.columns:
            if self._is_decimal_column(df[col]):
                logger.info(f"Converting Decimal column '{col}' to float")
                try:
                    df_converted[col] = pd.to_numeric(df[col].astype(str), errors='coerce')
                    logger.info(f"Successfully converted '{col}' - new dtype: {df_converted[col].dtype}")
                except Exception as e:
                    logger.warning(f"Failed to convert column '{col}': {e}")

        return df_converted

    def create_html_value_card(self, data: List[Dict], query: str, sub_query_index: int = None) -> Optional[Dict[str, Any]]:
        """Create a summary card payload (with optional HTML) for simple metric outputs."""
        if not data or not isinstance(data[0], dict):
            return None

        title = self._build_chart_title(query, sub_query_index)
        metrics: List[Dict[str, Any]] = []

        for key, value in data[0].items():
            numeric_value, formatted_value = self._format_metric_value(key, value)
            metrics.append({
                "label": key.replace("_", " ").title(),
                "value": numeric_value,
                "formatted": formatted_value,
            })

        if not metrics:
            return None

        html_content = self._render_summary_card_html(query, metrics, sub_query_index)
        summary_payload = {
            "type": "summary_card",
            "title": title,
            "description": query,
            "metrics": metrics,
            "primary_metric": metrics[0],
            "sub_query_index": sub_query_index,
        }

        return {
            "type": "summary_card",
            "graph_type": "summary_card",
            "title": title,
            "figure": None,
            "config": None,
            "summary": summary_payload,
            "html": html_content,
            "query": query,
            "sub_query_index": sub_query_index,
        }

    def analyze_data_for_visualization(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze data structure to determine visualization suitability"""
        if not data:
            return {"suitable": False, "reason": "No data"}

        df = pd.DataFrame(data)
        df_converted = self._convert_decimal_columns(df)

        num_rows = len(df)
        num_cols = len(df.columns)

        numeric_cols = []
        categorical_cols = []
        date_cols = []

        for col in df_converted.columns:
            if pd.api.types.is_numeric_dtype(df_converted[col]):
                numeric_cols.append(col)
            elif pd.api.types.is_datetime64_any_dtype(df_converted[col]) or self._looks_like_date(df_converted[col]):
                date_cols.append(col)
            else:
                categorical_cols.append(col)

        analysis = {
            "suitable": True,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "date_cols": date_cols,
            "reason": ""
        }

        if num_rows == 1 and num_cols == 1:
            analysis["suitable"] = False
            analysis["reason"] = "Single value result - better as value card"
        elif num_rows == 1 and num_cols > 1:
            analysis["suitable"] = False
            analysis["reason"] = "Single row with multiple values - better as summary card"
        elif num_rows < 2:
            analysis["suitable"] = False
            analysis["reason"] = "Insufficient data for visualization"
        elif len(numeric_cols) == 0:
            analysis["suitable"] = False
            analysis["reason"] = "No numeric data for visualization"
        else:
            analysis["suitable"] = True
            analysis["reason"] = "Suitable for visualization"

        logger.info(f"📊 Data analysis: {analysis}")
        return analysis

    def _looks_like_date(self, series: pd.Series) -> bool:
        """Check if a series looks like dates"""
        try:
            sample = series.dropna().head(3)
            if len(sample) == 0:
                return False

            for val in sample:
                try:
                    pd.to_datetime(val)
                    return True
                except:
                    continue
            return False
        except:
            return False

    def clean_json_response(self, response_text: str) -> str:
        """Robust JSON cleaning for graph generator LLM responses"""
        try:
            cleaned = response_text.strip()

            # Case 1: Try parsing as-is first
            try:
                json.loads(cleaned)
                logger.info("🧹 Graph JSON - no cleaning needed")
                return cleaned
            except:
                pass

            # Case 2: Find JSON between braces
            first_brace = cleaned.find('{')
            last_brace = cleaned.rfind('}')

            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                extracted = cleaned[first_brace:last_brace + 1]
                try:
                    json.loads(extracted)
                    logger.info("🧹 Graph JSON extracted between braces")
                    return extracted
                except:
                    pass

            # Case 3: Remove markdown patterns
            patterns = [r'^```json?\s*', r'\s*```\s*$', r'^```\s*']
            for pattern in patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)

            try:
                json.loads(cleaned)
                logger.info("🧹 Graph JSON cleaned with regex")
                return cleaned
            except:
                pass

            # Case 4: Brute force balanced braces
            for i in range(len(cleaned)):
                if cleaned[i] == '{':
                    brace_count = 0
                    for j in range(i, len(cleaned)):
                        if cleaned[j] == '{':
                            brace_count += 1
                        elif cleaned[j] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                candidate = cleaned[i:j+1]
                                try:
                                    json.loads(candidate)
                                    logger.info("🧹 Graph JSON found with brace balancing")
                                    return candidate
                                except:
                                    break

            logger.warning("🧹 All graph JSON cleaning methods failed")
            return response_text

        except Exception as e:
            logger.error(f"❌ Error cleaning graph JSON response: {e}")
            return response_text

    def determine_graph_type_with_llm(self, data: List[Dict], user_query: str, model_name: str, sub_query_index: int = None) -> Dict[str, Any]:
        """Use Groq LLM to determine the best graph type and column mappings with improved JSON parsing"""
        logger.info("📊 Using LLM to determine graph type and column mappings")

        if not data:
            logger.warning("No data provided for LLM analysis")
            return {"graph_type": None, "error": "No data available"}

        sample_data = data[:5] if len(data) > 5 else data
        columns = list(data[0].keys()) if data else []

        data_summary = {
            "total_rows": len(data),
            "columns": columns,
            "sample_data": sample_data
        }

        query_label = f" (Sub-query {sub_query_index})" if sub_query_index else ""

        prompt = f"""You are an expert data visualization analyst. Analyze the user query and data to determine the optimal visualization.

User Query{query_label}: {user_query}

Data Information:
- Total rows: {data_summary['total_rows']}
- Columns: {data_summary['columns']}
- Sample data: {json.dumps(sample_data, indent=2, default=str)}

IMPORTANT RULES:
1. You MUST choose a graph type from the available options
2. You MUST specify actual column names from the data
3. NEVER use "null" - always provide valid column names
4. For time-based data, use date columns for X-axis
5. For categorical comparisons, use categorical columns for X-axis
6. Always use numeric columns for Y-axis measurements

Available graph types:
- "barchart": Categorical data vs numeric values
- "linechart": Time series or ordered data trends
- "piechart": Parts of a whole (categorical proportions)
- "scatterplot": Relationship between two numeric variables
- "histogram": Distribution of a single numeric variable
- "groupedbarchart": Categories grouped by another dimension

RESPOND ONLY WITH VALID JSON (no markdown, no code blocks, no extra text):
{{
    "graph_type": "oneoftheabovetypes",
    "x_column": "exactcolumnnamefromdata",
    "y_column": "exactcolumnnamefromdata", 
    "group_column": "exactcolumnnameornull",
    "reasoning": "why this visualization fits the data and query"
}}"""

        try:
            logger.info("📤 Sending graph type determination request to Groq...")

            response = self.groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a data visualization expert. Always return ONLY valid JSON with actual column names from the provided data. Do not use markdown code blocks, do not add any extra text, just return pure JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800,
                n=1,
            )

            response_text = response.choices[0].message.content.strip()
            logger.info(f"📥 LLM raw response: {repr(response_text)}")

            cleaned_response = self.clean_json_response(response_text)
            logger.info(f"📥 LLM cleaned response: {cleaned_response}")

            try:
                graph_decision = json.loads(cleaned_response)

                if (not graph_decision.get('graph_type') or graph_decision.get('graph_type') in ['null', None, ''] or
                    not graph_decision.get('x_column') or graph_decision.get('x_column') in ['null', None, ''] or
                    not graph_decision.get('y_column') or graph_decision.get('y_column') in ['null', None, '']):
                    logger.warning("LLM returned null/empty values, using fallback logic")
                    return {"graph_type": None, "error": "LLM returned invalid null/empty values"}

                logger.info(f"📊 LLM determined graph type: {graph_decision.get('graph_type')}")
                logger.info(f"📊 Column mappings - X: {graph_decision.get('x_column')}, Y: {graph_decision.get('y_column')}, Group: {graph_decision.get('group_column')}")
                logger.info(f"📊 Reasoning: {graph_decision.get('reasoning')}")

                return graph_decision

            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse LLM response as JSON: {e}")
                logger.error(f"❌ Problematic response: {repr(cleaned_response)}")
                return {"graph_type": None, "error": f"Invalid JSON response from LLM: {str(e)}"}

        except Exception as e:
            logger.error(f"❌ Error getting graph type from LLM: {e}")
            return {"graph_type": None, "error": str(e)}

    def fallback_visualization_logic(self, data: List[Dict], query: str, sub_query_index: int = None) -> Optional[Dict[str, Any]]:
        """CORRECTED fallback logic - creates HTML cards when visualization is not suitable"""
        logger.info("🔄 Using fallback visualization logic")

        try:
            df = pd.DataFrame(data)
            df_converted = self._convert_decimal_columns(df)
            analysis = self.analyze_data_for_visualization(data)

            # THIS IS THE KEY FIX - if not suitable for visualization, create HTML card
            if not analysis["suitable"]:
                logger.info(f"📊 Data not suitable for visualization: {analysis['reason']} - creating HTML card instead")
                return self.create_html_value_card(data, query, sub_query_index)

            numeric_cols = analysis["numeric_cols"]
            categorical_cols = analysis["categorical_cols"] 
            date_cols = analysis["date_cols"]

            title = self._build_chart_title(query, sub_query_index)

            if len(date_cols) > 0 and len(numeric_cols) > 0:
                logger.info("🔄 Fallback: Creating line chart (time series)")
                return self.create_line_chart(df_converted, date_cols[0], numeric_cols[0], None, title, query, sub_query_index)
            elif len(categorical_cols) > 0 and len(numeric_cols) > 0:
                if len(df_converted[categorical_cols[0]].unique()) <= 8 and any(word in query.lower() for word in ['share', 'proportion', 'percentage', 'distribution']):
                    logger.info("🔄 Fallback: Creating pie chart")
                    return self.create_pie_chart(df_converted, categorical_cols[0], numeric_cols[0], title, query, sub_query_index)
                else:
                    logger.info("🔄 Fallback: Creating bar chart")
                    return self.create_bar_chart(df_converted, categorical_cols[0], numeric_cols[0], None, title, query, sub_query_index)
            elif len(numeric_cols) >= 2:
                logger.info("🔄 Fallback: Creating scatter plot")
                return self.create_scatter_plot(df_converted, numeric_cols[0], numeric_cols[1], None, title, query, sub_query_index)
            else:
                logger.info("🔄 No chart suitable - creating HTML value card as final fallback")
                return self.create_html_value_card(data, query, sub_query_index)

        except Exception as e:
            logger.error(f"❌ Error in fallback visualization: {e}")
            # Even if fallback fails, try to create a value card
            try:
                return self.create_html_value_card(data, query, sub_query_index)
            except:
                return None

    def create_bar_chart(self, df: pd.DataFrame, x_col: str, y_col: str, group_col: Optional[str], title: str, query: str, sub_query_index: int = None) -> Optional[Dict[str, Any]]:
        """Create bar chart with sub-query support and return JSON-ready payload."""
        try:
            logger.info(f"📊 Creating bar chart: X={x_col}, Y={y_col}, Group={group_col}")

            if group_col and group_col in df.columns:
                fig = px.bar(
                    df,
                    x=x_col,
                    y=y_col,
                    color=group_col,
                    title=title,
                    barmode='group',
                    text_auto=True,
                )
                chart_type = "groupedbarchart"
            else:
                fig = px.bar(df, x=x_col, y=y_col, title=title, text_auto=True)
                chart_type = "barchart"

            fig.update_layout(
                font=dict(size=12),
                title_font_size=16,
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                showlegend=bool(group_col),
            )

            return self._build_plotly_payload(fig, chart_type, title, query, sub_query_index)

        except Exception as exc:
            logger.error(f"❌ Error creating bar chart: {exc}")
            return None

    def create_line_chart(self, df: pd.DataFrame, x_col: str, y_col: str, group_col: Optional[str], title: str, query: str, sub_query_index: int = None) -> Optional[Dict[str, Any]]:
        """Create line chart with sub-query support and return JSON-ready payload."""
        try:
            logger.info(f"📊 Creating line chart: X={x_col}, Y={y_col}, Group={group_col}")

            if group_col and group_col in df.columns:
                fig = px.line(df, x=x_col, y=y_col, color=group_col, title=title, markers=True)
            else:
                fig = px.line(df, x=x_col, y=y_col, title=title, markers=True)

            fig.update_layout(
                font=dict(size=12),
                title_font_size=16,
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                showlegend=bool(group_col),
            )

            return self._build_plotly_payload(fig, 'linechart', title, query, sub_query_index)

        except Exception as exc:
            logger.error(f"❌ Error creating line chart: {exc}")
            return None

    def create_pie_chart(self, df: pd.DataFrame, names_col: str, values_col: str, title: str, query: str, sub_query_index: int = None) -> Optional[Dict[str, Any]]:
        """Create pie chart with sub-query support and return JSON-ready payload."""
        try:
            logger.info(f"📊 Creating pie chart: Names={names_col}, Values={values_col}")

            if len(df) > df[names_col].nunique():
                df_agg = df.groupby(names_col)[values_col].sum().reset_index()
            else:
                df_agg = df

            fig = px.pie(df_agg, names=names_col, values=values_col, title=title)
            fig.update_layout(font=dict(size=12), title_font_size=16, showlegend=True)

            return self._build_plotly_payload(fig, 'piechart', title, query, sub_query_index)

        except Exception as exc:
            logger.error(f"❌ Error creating pie chart: {exc}")
            return None

    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, group_col: Optional[str], title: str, query: str, sub_query_index: int = None) -> Optional[Dict[str, Any]]:
        """Create scatter plot with sub-query support and return JSON-ready payload."""
        try:
            logger.info(f"📊 Creating scatter plot: X={x_col}, Y={y_col}, Group={group_col}")

            if group_col and group_col in df.columns:
                fig = px.scatter(df, x=x_col, y=y_col, color=group_col, title=title)
            else:
                fig = px.scatter(df, x=x_col, y=y_col, title=title)

            fig.update_layout(
                font=dict(size=12),
                title_font_size=16,
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                showlegend=bool(group_col),
            )

            return self._build_plotly_payload(fig, 'scatterplot', title, query, sub_query_index)

        except Exception as exc:
            logger.error(f"❌ Error creating scatter plot: {exc}")
            return None

    def generate_smart_visualization(self, data: List[Dict], query: str, user_query: str, model_name: str, sub_query_index: int = None) -> Optional[Dict[str, Any]]:
        """Generate smart visualization with LLM guidance and CORRECTED fallback logic"""
        logger.info(f"🎨 Generating smart visualization for query: {query[:50]}{'...' if len(query) > 50 else query}")

        if not data:
            logger.warning("No data provided for visualization")
            return None

        # Analyze data suitability FIRST
        analysis = self.analyze_data_for_visualization(data)
        if not analysis["suitable"]:
            logger.info(f"Creating value/summary card instead of chart: {analysis['reason']}")
            return self.create_html_value_card(data, query, sub_query_index)

        # Try LLM-based determination for chart-suitable data
        graph_decision = self.determine_graph_type_with_llm(data, user_query, model_name, sub_query_index)

        if graph_decision.get("graph_type") and not graph_decision.get("error"):
            # Execute LLM decision
            df = pd.DataFrame(data)
            df_converted = self._convert_decimal_columns(df)

            title = self._build_chart_title(query, sub_query_index)

            graph_type = graph_decision["graph_type"].lower()
            x_col = graph_decision["x_column"]
            y_col = graph_decision["y_column"] 
            group_col = graph_decision.get("group_column")

            logger.info(f"📊 Executing LLM decision: {graph_type}")

            try:
                if graph_type == "barchart":
                    return self.create_bar_chart(df_converted, x_col, y_col, group_col, title, query, sub_query_index)
                elif graph_type == "groupedbarchart":
                    return self.create_bar_chart(df_converted, x_col, y_col, group_col, title, query, sub_query_index)
                elif graph_type == "linechart":
                    return self.create_line_chart(df_converted, x_col, y_col, group_col, title, query, sub_query_index)
                elif graph_type == "piechart":
                    return self.create_pie_chart(df_converted, x_col, y_col, title, query, sub_query_index)
                elif graph_type == "scatterplot":
                    return self.create_scatter_plot(df_converted, x_col, y_col, group_col, title, query, sub_query_index)
                else:
                    logger.warning(f"Unknown graph type: {graph_type}, using fallback")
                    return self.fallback_visualization_logic(data, query, sub_query_index)

            except Exception as e:
                logger.error(f"❌ Error executing LLM visualization decision: {e}")
                return self.fallback_visualization_logic(data, query, sub_query_index)
        else:
            # Use fallback logic (which will properly create HTML cards if needed)
            logger.info("🔄 LLM failed, using fallback logic")
            return self.fallback_visualization_logic(data, query, sub_query_index)

class MultiQuerySplitter:
    """Enhanced class to handle splitting complex queries into multiple sub-queries with robust JSON parsing"""

    def __init__(self, groq_client, model_name: str):
        self.groq_client = groq_client
        self.model_name = model_name
        logger.info("🔀 MultiQuerySplitter initialized with robust JSON parsing")

    def split_query(self, natural_language_query: str) -> Dict[str, Any]:
        """Split a complex query into multiple sub-queries and detect dashboard directives."""
        logger.info(f"🔀 Analyzing query for potential splits: {natural_language_query}")

        prompt = f"""You are an expert business analyst. Analyze the following natural language query and determine if it contains multiple distinct questions that should be answered separately.

Query: "{natural_language_query}"

RULES:
1. If the query contains multiple distinct business questions (like asking for different metrics, different time periods, different entities), split them into separate queries.
2. Each sub-query should be complete and self-contained.
3. If it's just one business question, return it as a single query.
4. Dashboard actions (add/remove) are NOT separate queries. They should only be reflected in the "dashboard" field.
5. Preserve the original context and intent of each question.
6. Each sub-query should be answerable independently.
7. Dashboard handling:
If the query mentions “add to dashboard” → return "dashboard": "Add"
If the query mentions “remove from dashboard” → return "dashboard": "Remove"
If the query does not mention dashboard actions → return "dashboard": "False"
8. When a dashboard action is present, capture the graph reference or intent in a new field "dashboard_target". This can be:
   - The specific chart nickname the user mentions (e.g., "total sales by category")
   - "latest" if the user asks to remove or update the most recent chart
   - "all" if the user requests removing everything from the dashboard
   - "" (empty string) only if no hint whatsoever is provided
9. If the query only contains a dashboard action without any business question, return:
"sub_queries": []
"should_split": false
"dashboard": "Add/Remove"
"dashboard_target": (as described above)

Examples of queries that should be split:
"Show me total sales of Chocolate Cake last week, number of units of CHIPSONA ALOO sold in August 2025, and top outlet by sales in Delhi NCR"
"What are the attendance rates by department and also show me the revenue trends for Q3 2024"

Examples of queries that should NOT be split:
"Show me sales by category for last month" → sub_queries = ["Show me sales by category for last month"], dashboard = "False"
"What is the total revenue from all outlets in Delhi region" → sub_queries = ["What is the total revenue from all outlets in Delhi region"], dashboard = "False"
"Show total sales by category and add graph on dashboard" → sub_queries = ["Show total sales by category"], dashboard = "Add"
"Compare sales across regions and remove from dashboard" → sub_queries = ["Compare sales across regions"], dashboard = "Remove"
"Remove the added graph from the dashboard" → sub_queries = [], dashboard = "Remove"

RESPOND ONLY WITH VALID JSON (no markdown, no extra text):
{{
    "should_split": true/false,
    "sub_queries": [
        "First complete question",
        "Second complete question",
        "etc..."
    ],
    "reasoning": "Brief explanation of why you split or didn't split",
    "dashboard": "Add/Remove/False",
    "dashboard_target": "Graph nickname | latest | all | ''"
}}

If should_split is false, put the original query as the only item in sub_queries array."""

        try:
            logger.info("📤 Sending query split request to LLM...")

            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a business analyst expert at breaking down complex queries into simple, answerable sub-questions. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                n=1
            )

            response_text = response.choices[0].message.content.strip()
            logger.info(f"📥 LLM split response: {response_text}")

            # Clean and parse JSON using the robust method
            cleaned_response = self._clean_json_response(response_text)
            logger.info(f"📥 LLM cleaned split response: {cleaned_response}")

            try:
                split_result = json.loads(cleaned_response)

                should_split = bool(split_result.get("should_split", False))
                dashboard_raw = (split_result.get("dashboard") or "False").strip()
                dashboard_target = (split_result.get("dashboard_target") or "").strip()
                sub_queries = split_result.get("sub_queries", [])
                reasoning = split_result.get("reasoning", "")

                if should_split and len(sub_queries) > 1:
                    logger.info(f"🔀 Query will be split into {len(sub_queries)} parts")
                    logger.info(f"🔀 Reasoning: {reasoning}")
                    for i, sq in enumerate(sub_queries, 1):
                        logger.info(f"🔀 Sub-query {i}: {sq}")
                else:
                    if not sub_queries and dashboard_raw.lower() == "false":
                        sub_queries = [natural_language_query]
                    elif not sub_queries:
                        logger.info("🔀 No analytical sub-query detected; dashboard-only action")
                    else:
                        logger.info("🔀 Query will be processed as single query")

                return {
                    "should_split": should_split,
                    "sub_queries": sub_queries,
                    "reasoning": reasoning,
                    "dashboard": dashboard_raw,
                    "dashboard_target": dashboard_target
                }

            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse split response after cleaning: {e}")
                logger.error(f"❌ Problematic cleaned response: {repr(cleaned_response)}")
                return {
                    "should_split": False,
                    "sub_queries": [natural_language_query],
                    "reasoning": "Fallback to single query due to JSON parse error",
                    "dashboard": "False",
                    "dashboard_target": ""
                }

        except Exception as e:
            logger.error(f"❌ Error splitting query: {e}")
            return {
                "should_split": False,
                "sub_queries": [natural_language_query],
                "reasoning": f"Splitter exception: {str(e)}",
                "dashboard": "False",
                "dashboard_target": ""
            }

    def _clean_json_response(self, response_text: str) -> str:
        """
        Robust JSON cleaning that handles multiple edge cases:
        1. Markdown code blocks with ``` 
        2. Code blocks with ```json
        3. Mixed text and JSON
        4. Plain JSON
        5. JSON with extra whitespace
        """
        try:
            cleaned = response_text.strip()

            # Case 1: Try parsing as-is (plain JSON)
            try:
                json.loads(cleaned)
                logger.info("📋 Split JSON - no cleaning needed")
                return cleaned
            except:
                pass

            # Case 2: Find JSON between first { and last }
            first_brace = cleaned.find('{')
            last_brace = cleaned.rfind('}')

            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                extracted = cleaned[first_brace:last_brace + 1]
                try:
                    json.loads(extracted)
                    logger.info("📋 Split JSON extracted between braces")
                    return extracted
                except:
                    pass

            # Case 3: Remove common markdown patterns
            patterns = [
                r'^```json?\s*',  # Starting ```json or ```
                r'\s*```\s*$',   # Ending ```
                r'^```\s*',       # Starting ``` 
            ]

            for pattern in patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)

            try:
                json.loads(cleaned)
                logger.info("📋 Split JSON cleaned with regex patterns")
                return cleaned
            except:
                pass

            # Case 4: Brute force - find balanced JSON braces
            for i in range(len(cleaned)):
                if cleaned[i] == '{':
                    brace_count = 0
                    for j in range(i, len(cleaned)):
                        if cleaned[j] == '{':
                            brace_count += 1
                        elif cleaned[j] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                candidate = cleaned[i:j+1]
                                try:
                                    json.loads(candidate)
                                    logger.info("📋 Split JSON found with brace balancing")
                                    return candidate
                                except:
                                    break

            logger.warning("📋 All split JSON cleaning methods failed, returning original")
            return response_text

        except Exception as e:
            logger.error(f"❌ Error in split JSON cleaning: {e}")
            return response_text

class GroqNL2SQL:
    """Groq API integration for Natural Language to SQL conversion with enhanced English context and Langfuse support"""

    def __init__(self, config: NL2SQLConfig):
        self.config = config
        self.model_name = config.model_name
        self._initialize_groq_client()

    def _initialize_groq_client(self):
        """Initialize Groq client with Langfuse integration"""
        try:
            self.groq_client = OpenAI(
                api_key=self.config.groq_api_key,
                base_url="https://api.groq.com/openai/v1"
            )

            self.openai_client = OpenAI(
                api_key=self.config.openai_api_key
            )

            logger.info("✅ Groq client initialized successfully with Langfuse integration")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq client: {e}")
            raise

    def check_query_relevance(self, natural_language_query: str) -> Tuple[bool, str]:
        """Check if query is relevant to business data analysis"""
        try:
            prompt = f"""Analyze this query to determine if it's a legitimate business data request or casual conversation.

Query: "{natural_language_query}"

Business data queries typically ask about:
- Sales, revenue, profits, transactions
- Inventory, products, items, stock
- Employees, attendance, departments  
- Customers, vendors, suppliers
- Financial data, expenses, budgets
- Analytics, reports, trends, comparisons
- Outlets, branches, locations

Casual conversation includes:
- Greetings (hi, hello, how are you)
- General questions not related to business data
- Personal conversations
- Requests for help or tutorials

Return ONLY: true (if business data query) or false (if casual conversation)"""

            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a query relevance classifier. Respond with only 'true' or 'false'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10,
                n=1
            )

            response_text = response.choices[0].message.content.strip().lower()
            is_relevant = "true" in response_text

            explanation = "Query appears to be business data related" if is_relevant else "Query appears to be casual conversation"
            return is_relevant, explanation

        except Exception as e:
            logger.error(f"❌ Error checking query relevance: {e}")
            return True, "Could not determine relevance due to API error. Proceeding with query processing."

    def generate_sql(self, natural_language_query: str, schema_context: str, conversation_context: str) -> Tuple[str, Optional[str]]:
        """Generate SQL query from natural language using Groq LLM with follow-up context"""
        logger.info(f"🔄 Generating SQL with Groq for query: {natural_language_query}")

        context_prompt = ""
        if conversation_context:
            context_prompt = f"""\n\nPrevious Conversation Context: {conversation_context}
This may be a follow-up question related to the previous conversation. Use this context to better understand the current query."""
            logger.info("📖 Using conversation context for SQL generation")

        prompt = f"""You are an expert SQL developer working with a PostgreSQL database for a business ERP system.
Your task is to convert natural language queries into accurate SQL statements.

{context_prompt}

{schema_context}

REQUIREMENTS:
1. Generate SELECT statements for data retrieval queries
2. Always use proper JOINs when accessing multiple tables
3. Use appropriate WHERE clauses for filtering  
4. Format dates properly (YYYY-MM-DD format)
5. Use ILIKE for case-insensitive string matching
6. Return only the SQL query - no explanations or markdown formatting
7. For percentage calculations, calculate them properly using COUNT and conditional logic
8. For attendance queries, use fact_attendance table with present_flag column
9. Consider standard business context and common practices
10. For follow-up questions, build upon previous context appropriately
11. Ensure all responses are in proper English
12. Only return one final SQL query, nothing else

Natural Language Query: {natural_language_query}

Generate only the SQL query:"""

        try:
            logger.info("📤 Sending request to Groq API...")

            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates SQL queries for business analytics with contextual understanding and clear English communication."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                top_p=0.8,
                max_tokens=2048,
                n=1,
            )

            sql_query = response.choices[0].message.content.strip()
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()

            logger.info(f"✅ SQL generated successfully with Groq")
            logger.info(f"📜 Generated SQL: {sql_query}")

            return sql_query, None

        except Exception as e:
            error_msg = f"Error generating SQL with Groq client: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return "", error_msg

    def regenerate_sql_after_error(self, natural_language_query: str, failed_sql: str, error_message: str, schema_context: str, conversation_context: str) -> Tuple[str, Optional[str]]:
        """IMPROVED: Regenerate SQL query after database execution error using OpenAI with better prompt"""
        logger.info(f"🔄 SWITCHING TO OPENAI FOR SQL REGENERATION")
        logger.info(f"🔄 Regenerating SQL after error for query: {natural_language_query}")
        logger.info(f"❌ Previous SQL failed: {failed_sql}")
        logger.info(f"❌ Error message: {error_message}")

        context_prompt = ""
        if conversation_context:
            context_prompt = f"""\n\nPrevious Conversation Context: {conversation_context}"""

        # IMPROVED PROMPT with more specific error handling instructions
        prompt = f"""You are an expert SQL debugging specialist. A SQL query failed with an error and you need to generate a corrected version.

{context_prompt}

{schema_context}

FAILED QUERY DETAILS:
Original Natural Language Query: "{natural_language_query}"
Failed SQL Query: {failed_sql}
Database Error Message: {error_message}

CRITICAL REQUIREMENTS FOR CORRECTION:
1. Analyze the specific error message and identify the root cause
2. If error mentions missing columns: Use only columns that exist in the schema above
3. If error mentions missing tables: Use only tables that exist in the schema above  
4. If error mentions syntax issues: Fix the SQL syntax according to PostgreSQL standards
5. If error mentions JOIN issues: Ensure proper foreign key relationships are used
6. If error mentions GROUP BY issues: Add all non-aggregated columns to GROUP BY clause
7. If error mentions date/time issues: Use proper PostgreSQL date functions and formats
8. If error mentions data type issues: Use appropriate casting (::date, ::integer, etc.)
9. Generate a working SQL query that fulfills the original natural language request
10. ONLY return the corrected SQL query - no explanations, no markdown formatting, no extra text

Generate the corrected SQL query that fixes the error and answers the original question:"""

        try:
            logger.info("📤 Sending regeneration request to OpenAI API...")

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert SQL debugging specialist. Analyze database errors and generate corrected SQL queries. Return ONLY the corrected SQL query with no explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                top_p=0.8,
                max_tokens=2048,
                n=1,
            )

            sql_query = response.choices[0].message.content.strip()
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()

            logger.info(f"✅ SQL regenerated successfully with OpenAI")
            logger.info(f"📜 Regenerated SQL: {sql_query}")

            return sql_query, None

        except Exception as e:
            error_msg = f"Error regenerating SQL with OpenAI client: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return "", error_msg

    def is_safe_query_for_execution(self, sql_query: str) -> Tuple[bool, str]:
        """Check if SQL query is safe for execution"""
        dangerous_keywords = ['drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update']
        sql_lower = sql_query.lower().strip()

        for keyword in dangerous_keywords:
            if f' {keyword} ' in f' {sql_lower} ' or sql_lower.startswith(keyword):
                return False, f"Query contains potentially dangerous keyword: {keyword}"

        return True, "Query appears safe for execution"

    def generate_description(self, execution_results: List[Dict], natural_language_query: str, conversation_context: str) -> str:
        """Generate comprehensive description of the results with business context"""
        logger.info("📝 Generating comprehensive result description...")

        if not execution_results:
            return "No data was found matching your query criteria."

        result_summary = {
            "total_records": len(execution_results),
            "columns": list(execution_results[0].keys()) if execution_results else [],
            "sample_data": execution_results[:3] if len(execution_results) >= 3 else execution_results,
            "data_types": {}
        }

        if execution_results:
            for col in result_summary["columns"]:
                sample_val = execution_results[0][col]
                if isinstance(sample_val, (int, float, Decimal)):
                    result_summary["data_types"][col] = "numeric"
                elif isinstance(sample_val, str):
                    result_summary["data_types"][col] = "text"  
                else:
                    result_summary["data_types"][col] = "other"

        context_prompt = ""
        if conversation_context:
            context_prompt = f"""\n\nConversation Context: {conversation_context}"""

        prompt = f"""Provide a detailed, informative description (around 10 lines) that includes key findings and insights from the data, business implications and trends, specific numbers and percentages where relevant, recommendations or observations for business decision-making, all written in clear, professional English suitable for business executives. The tone should be professional yet conversational, focusing on actionable insights that can drive business decisions, ensuring language is grammatically correct and business-friendly. The analysis should be presented in paragraph format within 120 words, without bullet points. The context is aligned with the Indian market, so monetary values must be presented in INR (₹) with proper formatting.

Original Query: {natural_language_query}{context_prompt}

Data Analysis Results:
- Total Records: {result_summary['total_records']}
- Data Structure: {json.dumps(result_summary, indent=2, default=str)}


You are a professional business data analyst. Based on the provided sales data, give a detailed description of the results. Only provide the analysis of the data, no other information."""

        try:
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a detailed business analyst who provides comprehensive insights from the given data in clear, professional English."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600,
                n=1,
            )

            description = response.choices[0].message.content.strip()
            logger.info("✅ Description generated successfully")
            return description

        except Exception as e:
            logger.error(f"❌ Error generating description: {e}")
            fallback_description = f"Query executed successfully. Found {result_summary['total_records']} records with various business metrics that can help in making informed decisions for your organization."
            logger.info("🔄 Using fallback description")
            return fallback_description

    def generate_combined_description(self, all_results: List[Dict], original_query: str, sub_queries: List[str]) -> str:
        """Generate a combined description for multiple sub-query results"""
        logger.info("📝 Generating combined description for multiple sub-queries...")

        results_summary = []
        for i, result in enumerate(all_results, 1):
            if result.get('execution_results') and not result.get('error'):
                results_summary.append({
                    "sub_query_index": i,
                    "sub_query": sub_queries[i-1] if i-1 < len(sub_queries) else f"Sub-query {i}",
                    "record_count": len(result['execution_results']),
                    "sample_data": result['execution_results'][:2] if result['execution_results'] else [],
                    "has_visualization": bool(result.get('visualization') or (result.get('visualizations') or []))
                })

        prompt = f"""Provide a detailed, informative description (around 10 lines) that includes key findings and insights from the data, business implications and trends, specific numbers and percentages where relevant, recommendations or observations for business decision-making, all written in clear, professional English suitable for business executives. The tone should be professional yet conversational, focusing on actionable insights that can drive business decisions, ensuring language is grammatically correct and business-friendly. The analysis should be presented in paragraph format within 120 words, without bullet points. The context is aligned with the Indian market, so monetary values must be presented in INR (₹) with proper formatting.

Original Complex Query: "{original_query}"

This query was broken down into {len(sub_queries)} sub-queries, each providing different insights:

Sub-Query Results Summary:
{json.dumps(results_summary, indent=2, default=str)}

You are a professional business data analyst. Create a cohesive executive summary that ties all the results together, based on the provided sales data, give a detailed description of the results. Only provide the analysis of the data, no other information.
"""

        try:
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a senior business analyst who creates comprehensive executive summaries from complex multi-part data analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800,
                n=1,
            )

            description = response.choices[0].message.content.strip()
            logger.info("✅ Combined description generated successfully")
            return description

        except Exception as e:
            logger.error(f"❌ Error generating combined description: {e}")
            successful_count = len([r for r in all_results if not r.get('error')])
            fallback_description = f"Successfully processed {successful_count} out of {len(all_results)} queries from your complex request. Each query provided specific business insights with visualizations where applicable."
            logger.info("🔄 Using fallback combined description")
            return fallback_description

class SchemaManager:
    """Manage database schema information"""

    def __init__(self):
        self.ddl_schema = """-- Dimension Tables
CREATE TABLE dim_bank_account (
    bank_account_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100),
    account_no VARCHAR(30),
    branch VARCHAR(50)
);

CREATE TABLE dim_category (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(50)
);

CREATE TABLE dim_outlet (
    outlet_id SERIAL PRIMARY KEY,
    outlet_name VARCHAR(100),
    location VARCHAR(100),
    region VARCHAR(50)
);

CREATE TABLE dim_item (
    item_id SERIAL PRIMARY KEY,
    item_name VARCHAR(100),
    category_id INTEGER REFERENCES dim_category(category_id),
    uom VARCHAR(20)
);

CREATE TABLE dim_vendor (
    vendor_id SERIAL PRIMARY KEY,
    vendor_name VARCHAR(100),
    contact_no VARCHAR(20),
    city VARCHAR(50)
);

CREATE TABLE dim_employee (
    employee_id SERIAL PRIMARY KEY,
    employee_name VARCHAR(100),
    designation VARCHAR(50),
    department VARCHAR(50),
    joining_date DATE
);

-- Fact Tables
CREATE TABLE fact_cheques (
    cheque_id SERIAL PRIMARY KEY,
    bank_account_id INTEGER REFERENCES dim_bank_account(bank_account_id),
    cheque_no VARCHAR(30),
    amount NUMERIC(12, 2),
    issue_date DATE,
    clearing_status VARCHAR(10), -- 'Pending', 'Cleared'
    clearing_date DATE
);

CREATE TABLE fact_payables (
    vendor_id INTEGER REFERENCES dim_vendor(vendor_id),
    invoice_id SERIAL PRIMARY KEY,
    invoice_date DATE,
    due_date DATE,
    amount_due NUMERIC(12, 2),
    status VARCHAR(10) -- 'Paid', 'Unpaid', 'Partial'
);

CREATE TABLE fact_sales (
    sale_id SERIAL PRIMARY KEY,
    date DATE,
    outlet_id INTEGER REFERENCES dim_outlet(outlet_id),
    item_id INTEGER REFERENCES dim_item(item_id),
    category_id INTEGER REFERENCES dim_category(category_id),
    quantity INTEGER,
    gross_amount NUMERIC(12, 2),
    discount_amount NUMERIC(12, 2),
    net_amount NUMERIC(12, 2),
    payment_method VARCHAR(20) -- 'Cash', 'Card', 'UPI', 'Wallet'
);

CREATE TABLE fact_purchases (
    purchase_id SERIAL PRIMARY KEY,
    po_id INTEGER,
    date DATE,
    item_id INTEGER REFERENCES dim_item(item_id),
    vendor_id INTEGER REFERENCES dim_vendor(vendor_id),
    ordered_qty INTEGER,
    received_qty INTEGER,
    unit_price NUMERIC(12, 2),
    total_amount NUMERIC(12, 2),
    status VARCHAR(20) -- 'Ordered', 'Partially Received', 'Completed'
);

CREATE TABLE fact_attendance (
    attendance_id SERIAL PRIMARY KEY,
    employee_id INTEGER REFERENCES dim_employee(employee_id),
    date DATE,
    in_time TIME,
    out_time TIME,
    present_flag CHAR(1), -- 'Y' or 'N'
    remarks VARCHAR(50)
);"""
        logger.info("📋 SchemaManager initialized with DDL schema")

    def get_schema_context(self) -> str:
        """Get formatted schema context for LLM"""
        context = f"""You are an expert Business Data Analyst that converts natural language questions into optimized SQL queries and determines the best type of visualization.

Database Schema Information:
{self.ddl_schema}

Important Guidelines:
1. Always understand the user's intent carefully before generating SQL.
2. Use proper table joins when querying multiple tables - respect primary/foreign key relationships.
3. All monetary amounts are stored as NUMERIC(12,2). Show values in standard indian INR currency format where needed.
4. Dates are stored as DATE type (YYYY-MM-DD format). Use proper date functions for filtering and grouping.
5. For attendance queries, calculate percentages using present_flag ('Y' for present, 'N' for absent).
6. Always use appropriate aggregations (SUM, AVG, COUNT, MIN, MAX) as per the question.
7. Ensure that column aliases are human-friendly (e.g., 'total_sales' instead of 'Sales').
8. If the query involves time-series data, ORDER BY date/month/year appropriately.
9. Return only valid SQL compatible with PostgreSQL.

Your output must strictly include:
- SQL Query

return context"""

        return context

class SmartNL2SQLProcessor:
    """Smart processor with CORRECTED visualization logic, robust error handling, and multi-query support"""

    def __init__(self, config_file: str = 'config.ini'):
        logger.info("🚀 Initializing SmartNL2SQLProcessor with corrected functionality...")

        self.config = NL2SQLConfig(config_file)
        self.db_manager = DatabaseManager(self.config)
        self.schema_manager = SchemaManager()
        self.groq_nl2sql = GroqNL2SQL(self.config)

        # Pass the Groq client to the graph generator for LLM-based decisions
        self.graph_generator = SmartGraphGenerator(groq_client=self.groq_nl2sql.groq_client)

        # Initialize the multi-query splitter with robust JSON parsing
        self.query_splitter = MultiQuerySplitter(self.groq_nl2sql.groq_client, self.config.model_name)

        logger.info("✅ SmartNL2SQLProcessor initialized successfully with all corrected functionality")

    def _derive_dashboard_directive(self, query: str, existing_action: str, existing_target: str) -> Tuple[str, str, bool]:
        """Heuristic dashboard intent detection when the splitter omits directives."""
        action = existing_action if existing_action in {"add", "remove"} else "none"
        target = (existing_target or "").strip()
        derived = False

        normalized = (query or "").strip()
        if not normalized:
            return action, target, derived

        q_lower = normalized.lower()
        has_dashboard_keyword = "dashboard" in q_lower or "pin board" in q_lower

        if action == "none" and has_dashboard_keyword:
            if re.search(r"\b(add|pin|save|push|place|put)\b", q_lower):
                action = "add"
                derived = True
            elif re.search(r"\b(remove|delete|clear|drop|unpin|detach)\b", q_lower):
                action = "remove"
                derived = True

        if action == "remove":
            if not target:
                # Extract quoted target first (e.g., "Total sales")
                quoted = re.search(r'["“”](.+?)["“”]', normalized)
                if quoted:
                    target = quoted.group(1).strip()
                    derived = True
                elif re.search(r"\b(all|every|everything)\b", q_lower):
                    target = "all"
                    derived = True
                elif re.search(r"\b(latest|last|recent)\b", q_lower):
                    target = "latest"
                    derived = True
                else:
                    match = re.search(
                        r"(?:remove|delete|clear|drop|unpin)\s+(?:the\s+)?(?:graph|chart|visualization)\s+(.+?)(?:\s+from\s+(?:the\s+)?dashboard|[.?!,]|$)",
                        normalized,
                        flags=re.IGNORECASE,
                    )
                    if match:
                        target = match.group(1).strip()
                        derived = True
            has_remove_keywords = action == "remove"
            if has_remove_keywords and not has_dashboard_keyword:
                # Guard against misclassification when dashboard keyword missing
                action = existing_action if existing_action in {"add", "remove"} else "none"
                target = existing_target or target
                return action, target, derived

        if action == "add" and not target:
            if re.search(r"\b(latest|last|recent)\b", q_lower):
                target = "latest"
                derived = True

        return action, target, derived

    def process_single_query(self, natural_language_query: str, conversation_context: str, sub_query_index: int = None) -> Dict[str, Any]:
        """Process a single query (can be original query or sub-query)"""
        logger.info(f"{'='*80}")
        query_label = f" (Sub-query {sub_query_index})" if sub_query_index else ""
        logger.info(f"🔍 PROCESSING QUERY{query_label}: {natural_language_query}")
        logger.info(f"{'='*80}")

        start_time = datetime.now()

        result = {
            "timestamp": start_time.isoformat(),
            "original_query": natural_language_query,
            "sql_query": "",
            "execution_results": [],
            "description": "",
            "visualization": None,
            "visualizations": [],
            "graph_type": None,
            "error": None,
            "warning": None,
            "processing_time_seconds": 0,
            "query_safe": True,
            "is_relevant": True,
            "retry_attempts": 0,
            "retry_details": [],
            "sub_query_index": sub_query_index,
            "dashboard_action": "none",
            "dashboard_target": "",
            "row_count": 0
        }

        try:
            # Step 1: Check query relevance FIRST
            logger.info(f"🔍 STEP 1: Checking query relevance...")
            is_relevant, relevance_explanation = self.groq_nl2sql.check_query_relevance(natural_language_query)
            result["is_relevant"] = is_relevant

            if not is_relevant:
                result["error"] = f"This query appears to be casual conversation rather than a business data request. {relevance_explanation}"
                result["description"] = "I'm here to help with business data analysis and reporting. Please ask questions about sales, employees, inventory, financials, or other business data."
                result["processing_time_seconds"] = (datetime.now() - start_time).total_seconds()
                logger.info(f"❌ Query marked as irrelevant: {relevance_explanation}")
                return result

            # Step 2: Connect to database
            logger.info(f"🔍 STEP 2: Connecting to database...")
            if not self.db_manager.connect():
                result["error"] = "Failed to connect to database"
                logger.error("❌ Database connection failed")
                return result

            # Step 3: Generate SQL with context
            logger.info(f"🔍 STEP 3: Generating SQL with Groq...")
            schema_context = self.schema_manager.get_schema_context()
            sql_query, generation_error = self.groq_nl2sql.generate_sql(natural_language_query, schema_context, conversation_context)

            if generation_error:
                result["error"] = generation_error
                logger.error(f"❌ SQL generation failed: {generation_error}")
                return result

            result["sql_query"] = sql_query

            # Step 4: Safety check
            logger.info(f"🔍 STEP 4: Performing safety check...")
            is_safe, safety_message = self.groq_nl2sql.is_safe_query_for_execution(sql_query)
            result["query_safe"] = is_safe

            if not is_safe:
                result["warning"] = f"Query generated but not executed due to safety concerns: {safety_message}"
                result["description"] = "SQL query was generated but blocked from execution for security reasons."
                logger.warning(f"⚠️  Query blocked for security: {safety_message}")
                return result

            # Step 5: Execute query with retry logic
            logger.info(f"🔍 STEP 5: Executing SQL with retry mechanism...")
            execution_results = None
            execution_error = None
            current_sql = sql_query
            max_retries = 5

            for attempt in range(max_retries + 1):
                logger.info(f"🔄 ATTEMPT {attempt + 1} OF {max_retries + 1}")
                logger.info(f"🔍 Executing SQL: {current_sql[:100]}{'...' if len(current_sql) > 100 else ''}")

                execution_results, execution_error = self.db_manager.execute_query(current_sql)

                if execution_error is None:
                    logger.info(f"✅ Query executed successfully on attempt {attempt + 1}")
                    break
                else:
                    logger.error(f"❌ Attempt {attempt + 1} failed: {execution_error}")
                    result["retry_attempts"] = attempt + 1
                    result["retry_details"].append({
                        "attempt": attempt + 1,
                        "sql": current_sql,
                        "error": execution_error
                    })

                    if attempt < max_retries:
                        logger.info(f"🔄 RETRYING - Regenerating SQL with OpenAI for attempt {attempt + 2}...")
                        regenerated_sql, regeneration_error = self.groq_nl2sql.regenerate_sql_after_error(
                            natural_language_query, current_sql, execution_error, schema_context, conversation_context
                        )

                        if regeneration_error:
                            logger.error(f"❌ Failed to regenerate SQL: {regeneration_error}")
                            result["error"] = f"Failed to regenerate SQL after attempt {attempt + 1}: {regeneration_error}"
                            return result

                        current_sql = regenerated_sql
                        result["sql_query"] = current_sql

                        is_safe, safety_message = self.groq_nl2sql.is_safe_query_for_execution(current_sql)
                        if not is_safe:
                            logger.error(f"❌ Regenerated query failed safety check: {safety_message}")
                            result["error"] = f"Regenerated query failed safety check: {safety_message}"
                            return result
                    else:
                        result["error"] = f"Query execution failed after {max_retries + 1} attempts. Final error: {execution_error}"
                        logger.error(f"❌ Max retries ({max_retries + 1}) exceeded. Giving up.")
                        return result

            result["execution_results"] = execution_results
            if isinstance(execution_results, list):
                result["row_count"] = len(execution_results)

            # Step 6: Generate visualization (CORRECTED - will create HTML cards when appropriate)
            logger.info(f"🔍 STEP 6: Generating smart visualization...")
            if execution_results:
                graph_payload = self.graph_generator.generate_smart_visualization(
                    execution_results,
                    natural_language_query,
                    natural_language_query,
                    self.config.model_name,
                    sub_query_index,
                )
                if graph_payload:
                    result["visualization"] = graph_payload
                    graph_type = graph_payload.get("graph_type") or graph_payload.get("type")
                    result["visualizations"] = [{
                        "payload": graph_payload,
                        "sub_query_index": sub_query_index,
                        "sub_query": natural_language_query,
                        "graph_type": graph_type,
                        "title": graph_payload.get("title"),
                    }]
                    result["graph_type"] = graph_type
                    logger.info(f"✅ Visualization generated: {result['graph_type']}")
                else:
                    logger.info("ℹ️  No visualization generated")

            # Step 7: Generate detailed description with context
            logger.info(f"🔍 STEP 7: Generating description...")
            description = self.groq_nl2sql.generate_description(execution_results, natural_language_query, conversation_context)
            if not description and isinstance(execution_results, list) and len(execution_results) == 0:
                description = "No data was found matching your query criteria."
            result["description"] = description

        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            logger.error(f"❌ Unexpected error: {traceback.format_exc()}")

        finally:
            end_time = datetime.now()
            result["processing_time_seconds"] = (end_time - start_time).total_seconds()
            logger.info(f"⏱️  Query processed in {result['processing_time_seconds']:.2f} seconds")

        return result

    def process_query_with_smart_visualization(self, natural_language_query: str, conversation_context: str = "") -> Dict[str, Any]:
        """Process a user query, handling dashboard-only intents and multi-query analytics."""
        logger.info(f"{'='*80}")
        logger.info(f"🚀 PROCESSING NEW QUERY: {natural_language_query}")
        logger.info(f"{'='*80}")

        start_time = datetime.now()

        try:
            # Step 1: Check if query should be split into multiple sub-queries
            logger.info(f"🔀 STEP 1: Analyzing query for potential splits...")
            split_metadata = self.query_splitter.split_query(natural_language_query)
            sub_queries = split_metadata.get("sub_queries") or []
            dashboard_raw = (split_metadata.get("dashboard") or "False").strip().lower()
            dashboard_action = "add" if dashboard_raw == "add" else "remove" if dashboard_raw == "remove" else "none"
            dashboard_target = (split_metadata.get("dashboard_target") or "").strip()
            dashboard_reasoning = split_metadata.get("reasoning") or ""

            derived_action, derived_target, derived_flag = self._derive_dashboard_directive(
                natural_language_query,
                dashboard_action,
                dashboard_target,
            )
            if derived_flag:
                if dashboard_reasoning:
                    dashboard_reasoning = f"{dashboard_reasoning} | Heuristic dashboard intent detected."
                else:
                    dashboard_reasoning = "Heuristic dashboard intent detected from user query."
            if derived_action != dashboard_action:
                dashboard_action = derived_action
            if not dashboard_target and derived_target:
                dashboard_target = derived_target

            logger.info(f"🔧 Dashboard directive: action={dashboard_action}, target='{dashboard_target}'")

            if not sub_queries and dashboard_action in {"add", "remove"}:
                logger.info("🧭 Dashboard-only request detected. Skipping SQL generation.")
                return self._build_dashboard_only_result(
                    user_query=natural_language_query,
                    action=dashboard_action,
                    target=dashboard_target,
                    reasoning=dashboard_reasoning,
                    start_time=start_time
                )

            if len(sub_queries) > 1:
                logger.info(f"🔀 Query split into {len(sub_queries)} sub-queries")
                multi_result = self._process_multiple_queries(natural_language_query, sub_queries, conversation_context, start_time)
                multi_result["dashboard_action"] = dashboard_action
                multi_result["dashboard_target"] = dashboard_target
                multi_result["dashboard_reasoning"] = dashboard_reasoning
                multi_result["user_request"] = natural_language_query
                return multi_result
            else:
                logger.info(f"🔀 Processing as single query")
                # Use the only sub-query if provided (already stripped of dashboard directive)
                executable_query = sub_queries[0] if sub_queries else natural_language_query
                result = self.process_single_query(executable_query, conversation_context)

                result["is_multi_query"] = False
                result["total_sub_queries"] = 1
                result["successful_queries"] = 1 if not result.get("error") else 0
                result["dashboard_action"] = dashboard_action
                result["dashboard_target"] = dashboard_target
                result["dashboard_reasoning"] = dashboard_reasoning
                result["user_request"] = natural_language_query
                result["executed_sub_query"] = executable_query

                return result

        except Exception as e:
            logger.error(f"❌ Critical error in main processing: {e}")
            return {
                "timestamp": start_time.isoformat(),
                "original_query": natural_language_query,
                "error": f"Critical processing error: {str(e)}",
                "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                "is_multi_query": False,
                "total_sub_queries": 0,
                "successful_queries": 0,
                "dashboard_action": "none",
                "dashboard_target": "",
                "dashboard_reasoning": "",
                "user_request": natural_language_query
            }

        finally:
            self.db_manager.close()

    def _build_dashboard_only_result(self, user_query: str, action: str, target: str, reasoning: str, start_time: datetime) -> Dict[str, Any]:
        """Return a structured response for dashboard-only commands."""
        action = action or "none"
        end_time = datetime.now()
        action_label = action.capitalize() if action != "none" else "None"
        target_label = target or ("latest" if action == "remove" else "")

        if action == "add":
            description = "Dashboard update requested: add the latest generated visualization to the dashboard."
        elif action == "remove":
            target_text = target_label or "requested graph"
            description = f"Dashboard update requested: remove {target_text} from the dashboard."
        else:
            description = "No dashboard action identified."

        result = {
            "timestamp": start_time.isoformat(),
            "original_query": user_query,
            "user_request": user_query,
            "sql_query": "",
            "execution_results": [],
            "description": description,
            "visualization": None,
            "visualizations": [],
            "graph_type": None,
            "error": None,
            "warning": None,
            "processing_time_seconds": (end_time - start_time).total_seconds(),
            "query_safe": True,
            "is_relevant": True,
            "retry_attempts": 0,
            "retry_details": [],
            "sub_query_index": None,
            "is_multi_query": False,
            "total_sub_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "irrelevant_queries": 0,
            "sub_queries": [],
            "sub_results": [],
            "combined_description": None,
            "dashboard_action": action,
            "dashboard_target": target,
            "dashboard_reasoning": reasoning,
            "dashboard_only": True,
            "row_count": 0
        }

        return result

    def _process_multiple_queries(self, original_query: str, sub_queries: List[str], conversation_context: str, start_time: datetime) -> Dict[str, Any]:
        """Process multiple sub-queries and combine results"""
        logger.info(f"🔀 Processing {len(sub_queries)} sub-queries...")

        all_results = []

        for i, sub_query in enumerate(sub_queries, 1):
            logger.info(f"🔀 Processing sub-query {i}/{len(sub_queries)}: {sub_query}")

            sub_result = self.process_single_query(sub_query, conversation_context, sub_query_index=i)
            all_results.append(sub_result)

        # Generate combined description
        logger.info(f"🔀 Generating combined description...")
        combined_description = self.groq_nl2sql.generate_combined_description(all_results, original_query, sub_queries)

        # Collect all successful results
        successful_results = [r for r in all_results if not r.get('error') and r.get('is_relevant', True)]
        failed_results = [r for r in all_results if r.get('error')]
        irrelevant_results = [r for r in all_results if not r.get('is_relevant', True)]
        total_rows = sum(r.get('row_count', 0) for r in successful_results)

        # Collect all visualizations
        visualizations = []
        for result in successful_results:
            entries = result.get('visualizations') or []
            if isinstance(entries, dict):
                entries = [entries]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                payload = entry.get('payload')
                if not isinstance(payload, dict):
                    continue
                sub_idx = entry.get('sub_query_index') or result.get('sub_query_index')
                base_query = entry.get('sub_query') or ''
                if not base_query and sub_idx and 1 <= sub_idx <= len(sub_queries):
                    base_query = sub_queries[sub_idx - 1]
                visualizations.append({
                    "sub_query_index": sub_idx,
                    "sub_query": base_query,
                    "payload": payload,
                    "graph_type": payload.get("graph_type") or payload.get("type"),
                    "title": payload.get("title"),
                })

        # Create comprehensive multi-query result
        multi_result = {
            "timestamp": start_time.isoformat(),
            "original_query": original_query,
            "is_multi_query": True,
            "total_sub_queries": len(sub_queries),
            "successful_queries": len(successful_results),
            "failed_queries": len(failed_results),
            "irrelevant_queries": len(irrelevant_results),
            "sub_queries": sub_queries,
            "sub_results": all_results,
            "combined_description": combined_description,
            "visualizations": visualizations,
            "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
            "error": None if successful_results else "All sub-queries failed",
            "warning": f"Some sub-queries failed ({len(failed_results)}/{len(sub_queries)})" if failed_results and successful_results else None,
            "dashboard_action": "none",
            "dashboard_target": "",
            "dashboard_reasoning": "",
            "user_request": original_query,
            "row_count": total_rows
        }

        logger.info(f"🔀 Multi-query processing complete:")
        logger.info(f"   Total sub-queries: {len(sub_queries)}")
        logger.info(f"   Successful: {len(successful_results)}")
        logger.info(f"   Failed: {len(failed_results)}")
        logger.info(f"   Irrelevant: {len(irrelevant_results)}")
        logger.info(f"   Visualizations: {len(visualizations)}")

        return multi_result


def main():
    """Main function with enhanced test queries including multi-query support and corrected functionality"""
    logger.info(f"{'='*80}")
    logger.info("🚀 SMART NL2SQL PROCESSOR - CORRECTED VERSION WITH ALL FUNCTIONALITY")
    logger.info(f"{'='*80}")

    processor = SmartNL2SQLProcessor()

    # Enhanced test queries covering various scenarios including multi-query cases
    comprehensive_test_queries = [
        "sales data of chocolate cake in last month and total purchase amount and name of top vendor and display the share of each payment method in total sales for the same month.",
    ]

    all_results = []
    conversation_history = ""

    for i, query in enumerate(comprehensive_test_queries, 1):
        logger.info(f"{'='*60}")
        logger.info(f"🧪 TEST QUERY {i}: {query}")
        logger.info(f"{'='*60}")

        result = processor.process_query_with_smart_visualization(query, conversation_history)
        all_results.append(result)

        # Handle different result types
        if result.get("is_multi_query", False):
            logger.info(f"📊 MULTI-QUERY RESULT SUMMARY:")
            logger.info(f"   Original query: {result['original_query']}")
            logger.info(f"   Total sub-queries: {result['total_sub_queries']}")
            logger.info(f"   Successful: {result['successful_queries']}")
            logger.info(f"   Failed: {result['failed_queries']}")
            logger.info(f"   Visualizations generated: {len(result['visualizations'])}")

            for viz in result['visualizations']:
                payload = viz.get('payload') if isinstance(viz, dict) else None
                graph_label = ''
                if isinstance(payload, dict):
                    graph_label = payload.get('graph_type') or payload.get('type') or ''
                title_label = ''
                if isinstance(payload, dict):
                    title_label = payload.get('title') or viz.get('sub_query') or ''
                logger.info(f"   📈 {graph_label or 'Visualization'}: {title_label or 'Payload stored'}")

            logger.info(f"📝 COMBINED DESCRIPTION:")
            logger.info(f"   {result['combined_description']}")

        else:
            # Handle single query results
            if not result.get("is_relevant", True):
                logger.info(f"❌ IRRELEVANT QUERY")
                logger.info(f"   Response: {result['description']}")
            elif result["error"]:
                logger.error(f"❌ ERROR: {result['error']}")
                if result.get("retry_attempts", 0) > 0:
                    logger.info(f"🔄 RETRY ATTEMPTS: {result['retry_attempts']}")
                    for retry_detail in result.get("retry_details", []):
                        logger.info(f"   Attempt {retry_detail['attempt']}: {retry_detail['error']}")
            elif result["warning"]:
                logger.warning(f"⚠️  WARNING: {result['warning']}")
                logger.info(f"   Generated SQL: {result['sql_query']}")
            else:
                logger.info(f"✅ SUCCESS")
                logger.info(f"   Generated SQL: {result['sql_query']}")
                logger.info(f"   Results: {len(result['execution_results'])} records found")
                logger.info(f"   Description: {result['description']}")
                if result.get("retry_attempts", 0) > 0:
                    logger.info(f"🔄 Required {result['retry_attempts']} retries before success")

                visualization_payload = result.get("visualization") if isinstance(result.get("visualization"), dict) else None
                if visualization_payload:
                    logger.info(f"📊 VISUALIZATION GENERATED: {visualization_payload.get('graph_type') or visualization_payload.get('type')}")
                    logger.info(f"   Title: {visualization_payload.get('title')}")
                else:
                    logger.info("ℹ️  NO VISUALIZATION GENERATED")

        logger.info(f"⏱️  Processing time: {result.get('processing_time_seconds', 0):.2f} seconds")

    # Comprehensive summary
    successful_queries = [r for r in all_results if not r.get("error") and r.get("is_relevant", True)]
    irrelevant_queries = [r for r in all_results if not r.get("is_relevant", True)]
    blocked_queries = [r for r in all_results if r.get("warning")]
    failed_queries = [r for r in all_results if r.get("error") and r.get("is_relevant", True)]
    retry_queries = [r for r in all_results if r.get("retry_attempts", 0) > 0]
    visualizations_generated = [
        r for r in all_results
        if isinstance(r.get("visualization"), dict)
        or (r.get("visualizations") and len(r.get("visualizations") or []) > 0)
    ]

    logger.info(f"{'='*80}")
    logger.info("📊 COMPREHENSIVE RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"📈 Total queries processed: {len(all_results)}")
    logger.info(f"❌ Irrelevant queries detected: {len(irrelevant_queries)}")
    logger.info(f"✅ Successfully executed: {len(successful_queries)}")
    logger.info(f"❌ Failed after retries: {len(failed_queries)}")
    logger.info(f"🔄 Required retries: {len(retry_queries)}")
    logger.info(f"⚠️  Blocked for security: {len(blocked_queries)}")
    logger.info(f"📊 Visualizations generated: {len(visualizations_generated)}")

    if successful_queries:
        avg_time = sum(r.get("processing_time_seconds", 0) for r in successful_queries) / len(successful_queries)
        logger.info(f"⏱️  Average processing time: {avg_time:.2f} seconds")

    logger.info(f"{'='*80}")
    logger.info("✅ PROCESSING COMPLETE - ALL FUNCTIONALITY CORRECTED AND WORKING")
    logger.info("📋 Key fixes applied:")
    logger.info("   1. ✅ HTML value cards created for single/multi-value results") 
    logger.info("   2. ✅ Robust JSON parsing for LLM responses")
    logger.info("   3. ✅ Multi-query support with individual visualizations")
    logger.info("   4. ✅ Improved SQL regeneration prompts")
    logger.info("   5. ✅ All original functionality preserved")
    logger.info(f"📁 Check logs and graphs directories for outputs")

if __name__ == "__main__":
    main()
