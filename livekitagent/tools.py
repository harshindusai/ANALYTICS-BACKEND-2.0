
import asyncio
import json
import logging
import os
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Optional, List

import requests
from livekit.agents import RunContext, function_tool




@function_tool()
async def check_health_status(
    context: RunContext,  # type: ignore
) -> str:
    """
    Check the local health API at http://localhost:8000/health.

    Use this tool whenever the user asks about system health, diagnostics, uptime,
    or anything related to the service being healthy. Returns the raw response body
    if the endpoint responds with HTTP 200.
    """
    url = "http://localhost:8000/health"

    try:
        response = await asyncio.to_thread(requests.get, url, timeout=5)
        response.raise_for_status()
        body = response.text.strip()
        if not body:
            body = "Health endpoint returned an empty body but status was 200."
        logging.info("Health check successful: %s", body)
        return body
    except Exception as exc:
        logging.error("Health check failed: %s", exc)
        return "Health check failed; please check the service manually."


INDUS_BACKEND_URL = os.getenv("INDUS_BACKEND_URL")
FALLBACK_BEARER_TOKEN = os.getenv("INDUS_BACKEND_BEARER_TOKEN")
FALLBACK_TRANSCRIPT_ID = os.getenv("INDUS_BACKEND_TRANSCRIPT_ID")


def _summarize_text(text: Optional[str], max_sentences: int = 2) -> Optional[str]:
    if not text:
        return None
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return None
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    summary = " ".join(sentences[:max_sentences]).strip()
    return summary or cleaned


@function_tool()
async def process_user_query(
    context: RunContext,  # type: ignore
    query: str,
    title: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    conversation_context: Optional[str] = None,
) -> str:
    """
    Send a business or analytics question to the Process Query API to obtain SQL and metadata.

    Use this when the user asks for business insights, analytics, or database-backed answers.
    """
    if not query:
        return "Cannot process an empty query."
    backend_url = INDUS_BACKEND_URL or os.getenv("INDUS_BACKEND_URL")
    if not backend_url:
        logging.error("process_user_query missing INDUS_BACKEND_URL configuration")
        return "Process Query API call failed; backend is not configured."

    try:
        userdata = context.userdata  # type: ignore[assignment]
    except ValueError:
        userdata = {}
    if not isinstance(userdata, dict):
        userdata = {}

    session_metadata = userdata.get("metadata")
    if not isinstance(session_metadata, dict):
        session_metadata = {}

    auth_token = userdata.get("auth_token") or session_metadata.get("auth_token") or FALLBACK_BEARER_TOKEN
    transcript_id = (
        userdata.get("transcript_id")
        or session_metadata.get("transcript_id")
        or FALLBACK_TRANSCRIPT_ID
    )
    if not auth_token:
        logging.error("process_user_query missing authorization token in session metadata")
        return "Process Query API call failed; authorization is unavailable."
    if not transcript_id:
        logging.error("process_user_query missing transcript id in session metadata")
        return "Process Query API call failed; transcript context is unavailable."

    generated_title = title or "Sales Query"
    request_metadata: dict[str, Any] = {}
    if isinstance(metadata, dict):
        request_metadata = dict(metadata)
    if "source" not in request_metadata:
        request_metadata["source"] = "voice-agent"

    payload = {
        "natural_language_query": query,
        "transcript_id": transcript_id,
        "title": generated_title or "User query",
        "metadata": request_metadata,
        "conversation_context": conversation_context or "User asked a business query.",
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}",
    }
    process_query_url = backend_url.rstrip("/") + "/process_query"

    try:
        response = await asyncio.to_thread(
            requests.post,
            process_query_url,
            json=payload,
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError:
            data = response.text.strip()
            logging.warning("Process Query API returned non-JSON payload: %s", data)
            return str(data)

        description_text: Optional[str] = None
        supplemental_hint: Optional[str] = None

        if isinstance(data, dict):
            transcript_id = data.get("transcript_id")
            chat_id_assistant = data.get("chat_id_assistant")
            status = data.get("status")
            message = (data.get("message") or "").strip()
            if transcript_id and chat_id_assistant:
                description_url = backend_url.rstrip("/") + f"/get_description/{transcript_id}/{chat_id_assistant}"
                tables_url = backend_url.rstrip("/") + f"/get_tables/{transcript_id}/{chat_id_assistant}"
                try:
                    desc_response = await asyncio.to_thread(
                        requests.get,
                        description_url,
                        headers=headers,
                        timeout=10,
                    )
                    if desc_response.ok:
                        desc_json = desc_response.json()
                        description_text = _summarize_text(desc_json.get("description"))
                except Exception as exc:
                    logging.warning("Failed to fetch description for transcript %s chat %s: %s", transcript_id, chat_id_assistant, exc)

                try:
                    tables_response = await asyncio.to_thread(
                        requests.get,
                        tables_url,
                        headers=headers,
                        timeout=10,
                    )
                    if tables_response.ok:
                        tables_json = tables_response.json()
                        record_count = tables_json.get("record_count")
                        if isinstance(record_count, int):
                            supplemental_hint = f"{record_count} record{'s' if record_count != 1 else ''} returned."
                except Exception as exc:
                    logging.debug("Failed to fetch tables metadata: %s", exc)

            summary_parts: List[str] = []
            if description_text:
                summary_parts.append(description_text)
            if supplemental_hint:
                summary_parts.append(supplemental_hint)
            if not summary_parts and message:
                summary_parts.append(message)
            final_summary = " ".join(summary_parts) if summary_parts else "Query processed successfully."
            if len(final_summary) > 400:
                final_summary = final_summary[:397].rstrip() + "..."
            if status and status not in {"success", ""}:
                final_summary = f"{status.capitalize()}: {final_summary}"
            logging.info("Process Query API success")
            return final_summary

        logging.info("Process Query API success with non-dict response")
        return json.dumps(data)
    except Exception as exc:
        logging.error("Process Query API call failed: %s", exc)
        return "Process Query API call failed; please try again later."
