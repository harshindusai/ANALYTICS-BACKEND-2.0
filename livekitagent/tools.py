
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


PROCESS_QUERY_URL = "https://unliquescent-bart-grizzled.ngrok-free.dev/process_query"
PROCESS_QUERY_TRANSCRIPT_ID = "transc_15d92c63-2f2d-496d-b83e-c20182ae5b9a"
PROCESS_QUERY_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": (
        "Bearer "
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyX2RmOTM0ZGIwMmU5NSIsImVtYWlsIjoia3VtYXdhdGhhcnNoMjAwNEBnbWFpbC5j"
        "b20iLCJleHAiOjE3NjAxNzQ3OTl9.t80FQIO5Rr9WVm6gl37VP8i1ITUrHJkWm9MAPvk3hb0"
    ),
}


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
    The `transcript_id` is preset and should not be changed.
    """
    if not query:
        return "Cannot process an empty query."

    generated_title = title or "Sales Query"
    payload = {
        "natural_language_query": query,
        "transcript_id": PROCESS_QUERY_TRANSCRIPT_ID,
        "title": generated_title or "User query",
        "metadata": metadata or {"source": "voice-agent"},
        "conversation_context": conversation_context or "User asked a business query.",
    }

    try:
        response = await asyncio.to_thread(
            requests.post,
            PROCESS_QUERY_URL,
            json=payload,
            headers=PROCESS_QUERY_HEADERS,
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

        logging.info("Process Query API success: %s", result)
        return result
    except Exception as exc:
        logging.error("Process Query API call failed: %s", exc)
        return "Process Query API call failed; please try again later."
