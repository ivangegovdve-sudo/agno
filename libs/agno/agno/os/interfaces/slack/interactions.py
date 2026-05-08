from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agno.os.interfaces.slack.ids import (
    ACTION_EXTERNAL_RESULT,
    ACTION_REJECT_REASON,
    external_result_block_id,
    feedback_action_id,
    parse_row_block_id,
    reject_reason_block_id,
    user_feedback_block_id,
    user_input_action_id,
    user_input_block_id,
)
from agno.os.interfaces.slack.types import (
    ParsedDecision,
    ParseError,
    SlackBlocks,
    SlackState,
    extract_feedback_picks,
    extract_field_value,
    tool_args,
    tool_name,
    truncate,
)
from agno.run.requirement import RunRequirement

DECISION_TITLE_MAX = 120  # Longer titles wrap awkwardly in Slack plan block
DECISION_VALUE_MAX = 40  # Card body renders poorly with long values


# --- Slack state helpers ---


def _get_action_state(state: SlackState, block_id: str, action_id: str) -> Dict[str, Any]:
    return state.get(block_id, {}).get(action_id, {})


# --- Pause type parsers ---
# Each parser extracts user decisions from Slack payload for one pause_type.
# Returns ParsedDecision with the resolved values; appends ParseError for validation failures.


# Parses Approve/Deny toggle state from block_id + optional rejection reason from InputBlock
def _parse_confirmation(
    requirement: RunRequirement,
    blocks: SlackBlocks,
    errors: List[ParseError],
    state: Optional[SlackState] = None,
) -> ParsedDecision:
    req_id = requirement.id or ""
    state = state or {}
    decision = None

    # Decision is encoded in block_id when user clicks Approve/Deny toggle
    for block in blocks:
        parsed = parse_row_block_id(block.get("block_id", ""))
        if parsed and parsed.get("req_id") == req_id and parsed.get("kind") == "confirmation":
            if parsed.get("status") == "decided":
                decision = parsed.get("decided")
                break

    if decision is None:
        name = tool_name(requirement)
        errors.append(ParseError(requirement_id=req_id, field=name, message="Approval decision required"))
        return ParsedDecision(requirement_id=req_id, pause_type="confirmation", approved=None)

    # Extract optional rejection reason from InputBlock state
    rejected_note = None
    if decision == "deny":
        reason_state = _get_action_state(state, reject_reason_block_id(req_id), ACTION_REJECT_REASON)
        reason_text = (reason_state.get("value") or "").strip()
        if reason_text:
            rejected_note = reason_text

    return ParsedDecision(
        requirement_id=req_id,
        pause_type="confirmation",
        approved=(decision == "approve"),
        rejected_note=rejected_note,
    )


# Parses text/dropdown fields from user_input_schema
def _parse_user_input(
    requirement: RunRequirement, state: SlackState, errors: List[ParseError]
) -> ParsedDecision:
    req_id = requirement.id or ""
    values: Dict[str, Any] = {}

    for field in requirement.user_input_schema or []:
        action_state = _get_action_state(
            state, user_input_block_id(req_id, field.name), user_input_action_id(field.name)
        )
        values[field.name] = extract_field_value(action_state)
        if values[field.name] is None:
            errors.append(ParseError(requirement_id=req_id, field=field.name, message="This field is required"))

    return ParsedDecision(requirement_id=req_id, pause_type="user_input", input_values=values)


# Parses checkbox/dropdown selections from user_feedback_schema questions
def _parse_user_feedback(
    requirement: RunRequirement, state: SlackState, errors: List[ParseError]
) -> ParsedDecision:
    req_id = requirement.id or ""
    selections: Dict[str, List[str]] = {}

    for i, question in enumerate(requirement.user_feedback_schema or []):
        action_state = _get_action_state(state, user_feedback_block_id(req_id, i), feedback_action_id(i))
        picked = extract_feedback_picks(action_state)
        if not picked:
            errors.append(ParseError(requirement_id=req_id, field=question.question, message="No option selected"))
        selections[question.question] = picked

    return ParsedDecision(requirement_id=req_id, pause_type="user_feedback", feedback_selections=selections)


# Parses pasted execution result from external_execution text field
def _parse_external(
    requirement: RunRequirement,
    state: SlackState,
    errors: List[ParseError],
) -> ParsedDecision:
    req_id = requirement.id or ""
    action_state = _get_action_state(state, external_result_block_id(req_id), ACTION_EXTERNAL_RESULT)
    result = (action_state.get("value") or "").strip()

    if not result:
        errors.append(ParseError(requirement_id=req_id, field="result", message="Result must be non-empty"))

    return ParsedDecision(
        requirement_id=req_id,
        pause_type="external_execution",
        external_result=result or None,
    )


# --- Public API ---


# Entry point: routes each requirement to its pause_type parser
def parse_submit_payload(
    payload: Dict[str, Any],
    requirements: List[RunRequirement],
) -> tuple[List[ParsedDecision], List[ParseError]]:
    blocks: SlackBlocks = (payload.get("message") or {}).get("blocks") or []
    state: SlackState = (payload.get("state") or {}).get("values") or {}

    decisions: List[ParsedDecision] = []
    errors: List[ParseError] = []

    for requirement in requirements:
        kind = requirement.pause_type
        if kind == "confirmation":
            decisions.append(_parse_confirmation(requirement, blocks, errors, state))
        elif kind == "user_input":
            decisions.append(_parse_user_input(requirement, state, errors))
        elif kind == "user_feedback":
            decisions.append(_parse_user_feedback(requirement, state, errors))
        elif kind == "external_execution":
            decisions.append(_parse_external(requirement, state, errors))

    return decisions, errors


# Mutates RunRequirement objects with parsed decisions — agent polls these for resolution
def apply_decisions(decisions: List[ParsedDecision], requirements: List[RunRequirement]) -> None:
    by_id = {r.id: r for r in requirements if r.id}

    for decision in decisions:
        requirement = by_id.get(decision.requirement_id)
        if requirement is None:
            continue

        if decision.pause_type == "confirmation":
            if decision.approved is True:
                requirement.confirm()
            elif decision.approved is False:
                requirement.reject(decision.rejected_note)
            # approved=None means undecided — skip, validation error already recorded
        elif decision.pause_type == "user_input" and decision.input_values is not None:
            requirement.provide_user_input(decision.input_values)
        elif decision.pause_type == "user_feedback" and decision.feedback_selections is not None:
            requirement.provide_user_feedback(decision.feedback_selections)
        elif decision.pause_type == "external_execution" and decision.external_result is not None:
            requirement.set_external_execution_result(decision.external_result)


# Formats "Approved: tool_name(args)" or "Denied: tool_name(args)" for resolved cards
def format_decision_title(decision: ParsedDecision, requirement: RunRequirement) -> str:
    if decision.pause_type != "confirmation":
        raise ValueError("format_decision_title only supports confirmation decisions")

    verb = "Approved" if decision.approved else "Denied"
    name = tool_name(requirement)
    args_dict = tool_args(requirement)
    arg_parts = []
    for k, v in args_dict.items():
        try:
            rendered = v if isinstance(v, str) else json.dumps(v, default=str)
        except (TypeError, ValueError):
            rendered = str(v)
        # Collapse newlines so multi-line JSON renders as single-line in the card header
        rendered = truncate(rendered.replace("\n", " ").strip(), DECISION_VALUE_MAX)
        arg_parts.append(f"{k}={rendered}")
    args = ", ".join(arg_parts)
    title = f"{verb}: {name}({args})" if args else f"{verb}: {name}"
    # Slack plan block wraps awkwardly on long titles; truncate to keep it single-line
    return truncate(title, DECISION_TITLE_MAX)
