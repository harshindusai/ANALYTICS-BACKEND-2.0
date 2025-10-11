from dotenv import load_dotenv
import os
import logging
import asyncio
from typing import Any, Dict, Optional

import httpx
import requests

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, JobContext
from livekit.plugins import noise_cancellation, openai, silero, google

# ✅ import your custom tools here
from tools import process_user_query, check_health_status

from prompts import AGENT_INSTRUCTION, SESSION_INSTRUCTION, SESSION_GREETING

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

load_dotenv()

INDUS_BACKEND_URL = os.getenv("INDUS_BACKEND_URL")
AGENT_API_KEY = os.getenv("LIVEKIT_AGENT_API_KEY")
ROOM_PREFIX = os.getenv("LIVEKIT_ROOM_PREFIX", "indus")
BACKEND_AUTH_TOKEN = os.getenv("INDUS_BACKEND_BEARER_TOKEN", (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiJ1c2VyX2RmOTM0ZGIwMmU5NSIsImVtYWlsIjoia3VtYXdhdGhhcnNoMjAwNEBnbWFpbC5jb20iLCJleHAiOjE3NjAxNzIyNTB9."
    "MWJlqpkC7vovP9oqYhlfdrMXMC68X_nBeOuXiUBTcBI"
))
DEFAULT_TRANSCRIPT_ID = os.getenv("INDUS_BACKEND_TRANSCRIPT_ID", "transc_15d92c63-2f2d-496d-b83e-c20182ae5b9a")


def _extract_session_id(room_name: Optional[str]) -> Optional[str]:
    if not room_name:
        return None
    prefix = f"{ROOM_PREFIX}-"
    if room_name.startswith(prefix):
        return room_name[len(prefix):]
    return None


async def _agent_request(path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not INDUS_BACKEND_URL or not AGENT_API_KEY:
        logger.debug("Backend integration skipped; missing INDUS_BACKEND_URL or LIVEKIT_AGENT_API_KEY.")
        return None
    url = INDUS_BACKEND_URL.rstrip("/") + path
    headers = {"x-agent-key": AGENT_API_KEY, "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            if response.headers.get("content-type", "").startswith("application/json"):
                return response.json()
    except Exception as exc:
        logger.warning("Backend request %s failed: %s", path, exc)
    return None


async def run_backend_query(question: str, session_id: str, context: Optional[str] = None) -> str:
    """Invoke the FastAPI /process_query endpoint via requests in a worker thread."""
    if not INDUS_BACKEND_URL:
        logger.warning("Cannot submit backend query; INDUS_BACKEND_URL is unset.")
        return "Backend is not configured."

    api_url = f"{INDUS_BACKEND_URL.rstrip('/')}/process_query"
    conversation_context = context or "User asked about sales by category"
    payload: Dict[str, Any] = {
        "natural_language_query": question,
        "transcript_id": DEFAULT_TRANSCRIPT_ID,
        "title": "Sales by category query",
        "metadata": {},
        "conversation_context": conversation_context,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BACKEND_AUTH_TOKEN}",
    }

    def _call_process_query() -> Dict[str, Any]:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        logger.info("process_query status=%s", response.status_code)
        try:
            data = response.json()
        except ValueError:
            logger.error("process_query returned non-JSON payload: %s", response.text)
            response.raise_for_status()
            return {}
        logger.info("process_query response=%s", data)
        response.raise_for_status()
        return data

    try:
        data = await asyncio.to_thread(_call_process_query)
    except requests.RequestException as exc:
        logger.error("process_query request failed: %s", exc)
        return "Unable to reach analytics backend."

    result = data.get("result") if isinstance(data, dict) else None
    if isinstance(result, dict):
        for key in ("description", "message", "dashboard_result", "summary"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
    message = data.get("message") if isinstance(data, dict) else None
    if isinstance(message, str) and message.strip():
        return message
    return "Query processed."


async def post_transcript(role: str, text: str, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    if not text or not text.strip():
        return
    payload: Dict[str, Any] = {
        "role": role,
        "text": text,
    }
    if metadata:
        payload["metadata"] = metadata
    await _agent_request(f"/livekit/session/{session_id}/transcripts", payload)


# ------------------------------
# Assistant definition (Google Realtime LLM)
# ------------------------------
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=AGENT_INSTRUCTION,
            llm=google.beta.realtime.RealtimeModel(
                voice="Aoede",
                temperature=0.8,
            ),
            tools=[
                process_user_query,
                check_health_status,
            ],
        )


def _extract_reply_text(reply: Any) -> Optional[str]:
    if reply is None:
        return None
    if isinstance(reply, str):
        text = reply.strip()
        return text or None
    if isinstance(reply, (list, tuple)):
        parts = []
        for item in reply:
            item_text = _extract_reply_text(item)
            if item_text:
                parts.append(item_text)
        combined = " ".join(parts).strip()
        return combined or None
    if isinstance(reply, dict):
        for key in ("text", "message", "content"):
            if key in reply:
                text = _extract_reply_text(reply[key])
                if text:
                    return text
        return None
    for attr in ("text", "message", "content"):
        if hasattr(reply, attr):
            text = _extract_reply_text(getattr(reply, attr))
            if text:
                return text
    return None


async def entrypoint(ctx: agents.JobContext):
    logger.info("Starting Agent Session...")

    room_name = getattr(ctx.room, "name", None)
    session_id = _extract_session_id(room_name)
    if session_id:
        logger.info("Resolved session id %s for room %s", session_id, room_name)
    else:
        logger.warning("Unable to determine session id from room name %s", room_name)

    session = AgentSession(
        llm=google.beta.realtime.RealtimeModel(
            voice="Aoede",
            temperature=0.8,
        ),
        stt=openai.STT(model="gpt-4o-transcribe"),
        tts=openai.TTS(model="tts-1", voice="nova"),
        vad=silero.VAD.load(),
    )
    assistant = Assistant()

    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(
            video_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()
    logger.info("Connected to LiveKit room.")
    if session_id:
        await post_transcript("system", "Agent connected to LiveKit room.", session_id)

    # initial system instruction
    greeting_text = SESSION_GREETING.strip()
    try:
        initial_reply = await session.generate_reply(instructions=SESSION_INSTRUCTION)
        spoken_text = _extract_reply_text(initial_reply)
        if not spoken_text and greeting_text:
            logger.info("Initial reply was empty; falling back to deterministic greeting.")
            try:
                fallback_reply = await session.generate_reply(
                    instructions=f'Say exactly: "{greeting_text}"'
                )
                spoken_text = _extract_reply_text(fallback_reply) or greeting_text
            except Exception as fallback_exc:
                logger.warning("Fallback greeting synthesis failed: %s", fallback_exc)
                spoken_text = greeting_text
        if session_id and spoken_text:
            await post_transcript("assistant", spoken_text, session_id)
        logger.info("Initial reply generated.")
    except Exception as exc:
        logger.warning("Initial reply generation failed: %s", exc)
        if session_id and greeting_text:
            await post_transcript("assistant", greeting_text, session_id)
        if greeting_text:
            try:
                await session.generate_reply(instructions=f'Say exactly: "{greeting_text}"')
            except Exception as speech_exc:
                logger.warning("Failed to synthesize fallback greeting after error: %s", speech_exc)


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
