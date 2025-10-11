
import asyncio
import json
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Optional

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

        if isinstance(data, (dict, list)):
            result = json.dumps(data)
        else:
            result = str(data)

        logging.info("Process Query API success")
        return result
    except Exception as exc:
        logging.error("Process Query API call failed: %s", exc)
        return "Process Query API call failed; please try again later."
