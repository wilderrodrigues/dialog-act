"""Placeholder for storing participant data.

The real storage is written later. Until then everything is only logged, so nothing is persisted. Replace the
bodies of these two functions; the API calls them with the data shown in the signatures.
"""
import logging
from typing import Any

logger = logging.getLogger("chatbot.storage")


def save_participant(session_id: str, profile: dict[str, Any]) -> None:
    logger.info("participant %s: %s", session_id, profile)


def save_turn(session_id: str, turn: dict[str, Any]) -> None:
    logger.info("turn %s: %s", session_id, turn)
