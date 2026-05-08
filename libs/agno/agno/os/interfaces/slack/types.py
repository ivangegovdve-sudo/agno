from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agno.run.requirement import PauseType

if TYPE_CHECKING:
    from agno.run.requirement import RunRequirement


def block_to_dict(block: Any) -> Dict[str, Any]:
    """Convert a Slack block (SDK model, dataclass, or dict) to a plain dict."""
    if hasattr(block, "to_dict"):
        return block.to_dict()
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True, mode="json")
    if is_dataclass(block) and not isinstance(block, type):
        return asdict(block)
    return block if isinstance(block, dict) else {}


@dataclass
class ParsedDecision:
    requirement_id: str
    pause_type: PauseType
    approved: Optional[bool] = None
    rejected_note: Optional[str] = None
    input_values: Optional[Dict[str, Any]] = None
    feedback_selections: Optional[Dict[str, List[str]]] = None
    external_result: Optional[str] = None


@dataclass
class ParseError:
    requirement_id: str
    field: str
    message: str


# Slack buttons have a 2000-char value limit; text fields have 3000-char limits
def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


# tool_execution may be None or lack tool_name if requirement is for user_input/feedback
def _tool_name(requirement: "RunRequirement") -> str:
    tool = requirement.tool_execution
    return getattr(tool, "tool_name", None) or "tool"


def _tool_args(requirement: "RunRequirement") -> Dict[str, Any]:
    tool = requirement.tool_execution
    # Empty dict fallback ensures JSON serialization never fails
    return getattr(tool, "tool_args", None) or {}
