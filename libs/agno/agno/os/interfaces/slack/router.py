from __future__ import annotations

import json
from ssl import SSLContext
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from agno.agent import Agent, RemoteAgent
from agno.os.interfaces.slack.events import process_event
from agno.os.interfaces.slack.helpers import (
    BotNameResolver,
    build_run_metadata,
    download_event_files_async,
    extract_event_context,
    resolve_channel_name,
    resolve_slack_user,
    send_slack_message_async,
    should_respond,
    slack_error_code,
    strip_bot_mention,
    upload_response_media_async,
)
from agno.os.interfaces.slack.security import verify_slack_signature
from agno.os.interfaces.slack.state import StreamState, TaskStatus
from agno.os.interfaces.slack.ids import decode_row_button_value, decode_submit_button_value
from agno.os.interfaces.slack.builders import (
    append_submit_if_needed,
    build_submit_button,
    decision_marker,
    select_confirmation_row,
)
from agno.os.interfaces.slack.interactions import (
    confirmation_row_summary,
    extract_row_action_context,
    synthetic_submit_payload,
)
from agno.os.interfaces.slack.types import SubmitContext, block_to_dict
from agno.team import RemoteTeam, Team
from agno.tools.slack import SlackTools
from agno.utils.log import log_error, log_info
from agno.workflow import RemoteWorkflow, Workflow

# Slack sends lifecycle events for bots with these subtypes. Without this
# filter the router would try to process its own messages, causing infinite loops.
_IGNORED_SUBTYPES = frozenset(
    {
        "bot_message",
        "bot_add",
        "bot_remove",
        "bot_enable",
        "bot_disable",
        "message_changed",
        "message_deleted",
    }
)

# User-facing error message for failed requests
_ERROR_MESSAGE = "Sorry, there was an error processing your message."

# Slack caps streamed messages at ~40K total payload (text + task card blocks)
_STREAM_CHAR_LIMIT = 39000
_STREAM_CARD_LIMIT = 45


class SlackEventResponse(BaseModel):
    status: str = Field(default="ok")


class SlackChallengeResponse(BaseModel):
    challenge: str = Field(description="Challenge string to echo back to Slack")


def attach_routes(
    router: APIRouter,
    agent: Optional[Union[Agent, RemoteAgent]] = None,
    team: Optional[Union[Team, RemoteTeam]] = None,
    workflow: Optional[Union[Workflow, RemoteWorkflow]] = None,
    reply_to_mentions_only: bool = True,
    token: Optional[str] = None,
    signing_secret: Optional[str] = None,
    streaming: bool = True,
    loading_messages: Optional[List[str]] = None,
    task_display_mode: str = "plan",
    loading_text: str = "Thinking...",
    suggested_prompts: Optional[List[Dict[str, str]]] = None,
    ssl: Optional[SSLContext] = None,
    buffer_size: int = 100,
    max_file_size: int = 1_073_741_824,  # 1GB
    resolve_user_identity: bool = False,
) -> APIRouter:
    # Inner functions capture config via closure to keep each instance isolated
    entity = agent or team or workflow
    # entity_type drives event dispatch (agent vs team vs workflow events)
    entity_type: Literal["agent", "team", "workflow"] = "agent" if agent else "team" if team else "workflow"
    # Member HITL needs member runs embedded on Team run (member_responses).
    # Without this, continue_run cannot reliably reload member tool state from DB.
    if team is not None and not isinstance(team, RemoteTeam):
        team.store_member_responses = True
    raw_name = getattr(entity, "name", None)
    # entity_name labels task cards; entity_id namespaces session IDs
    entity_name = raw_name if isinstance(raw_name, str) else entity_type
    # Multiple Slack instances can be mounted on one FastAPI app (e.g. /research
    # and /analyst). op_suffix makes each operation_id unique to avoid collisions.
    op_suffix = entity_name.lower().replace(" ", "_")
    entity_id = getattr(entity, "id", None) or entity_name

    slack_tools = SlackTools(token=token, ssl=ssl, max_file_size=max_file_size)
    bot_name_resolver = BotNameResolver()

    @router.post(
        "/events",
        operation_id=f"slack_events_{op_suffix}",
        name="slack_events",
        description="Process incoming Slack events",
        response_model=Union[SlackChallengeResponse, SlackEventResponse],
        response_model_exclude_none=True,
        responses={
            200: {"description": "Event processed successfully"},
            400: {"description": "Missing Slack headers"},
            403: {"description": "Invalid Slack signature"},
        },
    )
    async def slack_events(request: Request, background_tasks: BackgroundTasks):
        # ACK immediately, process in background. Slack retries after ~3s if it
        # doesn't get a 200, so long-running agent calls must not block the response.
        body = await request.body()
        timestamp = request.headers.get("X-Slack-Request-Timestamp")
        slack_signature = request.headers.get("X-Slack-Signature", "")

        if not timestamp or not slack_signature:
            raise HTTPException(status_code=400, detail="Missing Slack headers")

        if not verify_slack_signature(body, timestamp, slack_signature, signing_secret=signing_secret):
            raise HTTPException(status_code=403, detail="Invalid signature")

        # Slack retries after ~3s if it doesn't get a 200. Since we ACK
        # immediately and process in background, retries are always duplicates.
        # Trade-off: if the server crashes mid-processing, the retried event
        # carrying the same payload won't be reprocessed — acceptable for chat.
        if request.headers.get("X-Slack-Retry-Num"):
            return SlackEventResponse(status="ok")

        data = await request.json()

        if data.get("type") == "url_verification":
            return SlackChallengeResponse(challenge=data.get("challenge"))

        if "event" in data:
            event = data["event"]
            event_type = event.get("type")
            # setSuggestedPrompts requires "Agents & AI Apps" mode (streaming UX only)
            if event_type == "assistant_thread_started" and streaming:
                background_tasks.add_task(_handle_thread_started, event)
            # Bot self-loop prevention: check bot_id at BOTH the top-level event
            # AND inside message_changed's nested "message" object. Slack puts
            # bot_id at different nesting levels depending on event shape — the
            # nested check catches edited bot messages that would otherwise be
            # reprocessed as new user events.
            elif (
                event.get("bot_id")
                or (event.get("message") or {}).get("bot_id")
                or event.get("subtype") in _IGNORED_SUBTYPES
            ):
                pass
            elif streaming:
                background_tasks.add_task(_stream_slack_response, data)
            else:
                background_tasks.add_task(_process_slack_event, data)

        return SlackEventResponse(status="ok")

    @router.post(
        "/interactions",
        operation_id=f"slack_interactions_{op_suffix}",
        name="slack_interactions",
        description="Handle Slack interactive components (HITL buttons / form submit)",
        response_model=SlackEventResponse,
        response_model_exclude_none=True,
        responses={
            200: {"description": "Interaction accepted"},
            400: {"description": "Malformed interaction payload"},
            403: {"description": "Invalid Slack signature"},
        },
    )
    async def slack_interactions(request: Request, background_tasks: BackgroundTasks):
        body = await request.body()
        timestamp = request.headers.get("X-Slack-Request-Timestamp")
        slack_signature = request.headers.get("X-Slack-Signature", "")
        if not timestamp or not slack_signature:
            raise HTTPException(status_code=400, detail="Missing Slack headers")
        if not verify_slack_signature(body, timestamp, slack_signature, signing_secret=signing_secret):
            raise HTTPException(status_code=403, detail="Invalid signature")

        # Pre-ack retry drop — Slack retries after ~3s if we don't ack. We ACK
        # below; any retry arriving before that gets the same 200 response.
        if request.headers.get("X-Slack-Retry-Num"):
            return SlackEventResponse(status="ok")

        # Slack sends interactive payloads as application/x-www-form-urlencoded
        # with a single form field `payload=<URL-encoded JSON>`.
        form = await request.form()
        payload_raw = form.get("payload")
        if not isinstance(payload_raw, str) or not payload_raw:
            raise HTTPException(status_code=400, detail="Missing payload")
        try:
            payload = json.loads(payload_raw)
        except Exception:
            raise HTTPException(status_code=400, detail="Malformed payload JSON")

        # Dispatch by action_id — only block_actions payloads carry HITL clicks.
        if payload.get("type") != "block_actions":
            return SlackEventResponse(status="ok")
        actions = payload.get("actions") or []
        if not actions:
            return SlackEventResponse(status="ok")
        action_id = actions[0].get("action_id", "")

        if action_id == "row_approve":
            background_tasks.add_task(_handle_row_approve, payload)
        elif action_id == "row_reject":
            background_tasks.add_task(_handle_row_reject_start, payload)
        elif action_id == "submit_pause":
            background_tasks.add_task(_handle_submit, payload)
        # Silently ignore unknown action_ids — a non-HITL Slack app sharing
        # the same endpoint might also post interactions here.

        return SlackEventResponse(status="ok")

    # --- Closure-dependent helpers (use slack_tools, entity, ssl from attach_routes scope) ---

    async def _safe_chat_update(
        channel: str,
        ts: str,
        text: str,
        blocks: List[Dict[str, Any]],
        log_context: str,
    ) -> bool:
        # Wraps chat_update with consistent error handling
        from slack_sdk.web.async_client import AsyncWebClient

        client = AsyncWebClient(token=slack_tools.token, ssl=ssl)
        try:
            await client.chat_update(channel=channel, ts=ts, text=text, blocks=blocks)
            return True
        except Exception as exc:
            log_error(f"[HITL] chat_update ({log_context}) failed for {ts}: {exc}")
            return False

    def _extract_submit_context(payload: Dict[str, Any]) -> Optional[SubmitContext]:
        # Extracts and validates context from submit button click payloads
        actions = payload.get("actions") or []
        if not actions:
            return None
        submit_block_id = actions[0].get("block_id") or ""
        if not submit_block_id.startswith("pause:"):
            return None
        run_id = submit_block_id.removeprefix("pause:")

        channel = (payload.get("channel") or {}).get("id")
        message = payload.get("message") or {}
        msg_ts = message.get("ts")
        if not (run_id and channel and msg_ts):
            return None

        thread_ts = message.get("thread_ts") or msg_ts
        button_value = actions[0].get("value") or ""
        _, awaiting_ts = decode_submit_button_value(button_value)

        return SubmitContext(
            run_id=run_id,
            channel=channel,
            msg_ts=msg_ts,
            thread_ts=thread_ts,
            session_id=f"{entity_id}:{thread_ts}",
            awaiting_ts=awaiting_ts,
            user_id=(payload.get("user") or {}).get("id", ""),
            team_id=(payload.get("team") or {}).get("id"),
            state_values=(payload.get("state") or {}).get("values") or {},
        )

    async def _delete_awaiting_indicator(channel: str, awaiting_ts: Optional[str]) -> None:
        # Silently deletes "Awaiting..." message, ignores message_not_found on retries
        if not awaiting_ts:
            return
        from slack_sdk.web.async_client import AsyncWebClient

        client = AsyncWebClient(token=slack_tools.token, ssl=ssl)
        try:
            await client.chat_delete(channel=channel, ts=awaiting_ts)
        except Exception as exc:
            if "message_not_found" not in str(exc):
                log_error(f"[HITL] chat_delete (awaiting indicator) failed for ts={awaiting_ts}: {exc}")

    async def _load_active_requirements(ctx: SubmitContext) -> List[Any]:
        # Fetches paused run's requirements, returns empty list on failure
        try:
            run_output = await entity.aget_run_output(run_id=ctx.run_id, session_id=ctx.session_id)  # type: ignore[union-attr]
        except Exception as exc:
            log_error(f"[HITL] aget_run_output failed for run={ctx.run_id}: {exc}")
            return []
        return list(getattr(run_output, "active_requirements", None) or []) if run_output else []

    async def _lock_submitted_form(ctx: SubmitContext, original_blocks: List[Dict[str, Any]], requirements: List[Any]) -> None:
        # Converts interactive inputs/cards to readonly display after submit
        has_inputs = any(b.get("type") == "input" for b in original_blocks)
        has_interactive_cards = any(b.get("type") == "card" and b.get("actions") for b in original_blocks)
        if not has_inputs and not has_interactive_cards:
            return

        from slack_sdk.web.async_client import AsyncWebClient

        from agno.os.interfaces.slack.builders import response_blocks

        readonly_blocks = response_blocks(original_blocks, ctx.state_values, requirements)
        client = AsyncWebClient(token=slack_tools.token, ssl=ssl)
        try:
            await client.chat_update(channel=ctx.channel, ts=ctx.msg_ts, text="Submitted", blocks=readonly_blocks)
        except Exception as exc:
            log_error(f"[HITL] chat_update (submit readonly) failed for {ctx.msg_ts}: {exc}")

    async def _validate_and_apply_decisions(
        ctx: SubmitContext,
        payload: Dict[str, Any],
        requirements: List[Any],
    ) -> Optional[List[Any]]:
        # Parses payload, posts ephemeral on errors, applies decisions. Returns decisions or None on error.
        from slack_sdk.web.async_client import AsyncWebClient

        from agno.os.interfaces.slack.interactions import apply_decisions, parse_submit_payload

        decisions, errors = parse_submit_payload(payload, requirements)
        if errors:
            detail = "\n".join(f"• {e.field}: {e.message}" for e in errors)
            client = AsyncWebClient(token=slack_tools.token, ssl=ssl)
            await _post_ephemeral(client, channel=ctx.channel, user=ctx.user_id, text=f"Please fix the following and submit again:\n{detail}")
            return None
        apply_decisions(decisions, requirements)
        return decisions

    async def _emit_denied_decision_cards(stream: Any, decisions: List[Any], requirements: List[Any], run_id: str) -> None:
        # Emits task cards for DENIED confirmations only (approved ones show via tool's call card)
        from agno.os.interfaces.slack.types import tool_args, tool_name, truncate

        requirements_by_id = {r.id: r for r in requirements if r.id}
        for decision in decisions:
            req = requirements_by_id.get(decision.requirement_id)
            if req is None or decision.pause_type != "confirmation" or decision.approved is True:
                continue
            name = tool_name(req)
            args_dict = tool_args(req)
            arg_parts = [f"{k}={truncate(str(v), 40)}" for k, v in args_dict.items()]
            args_str = ", ".join(arg_parts)
            title = truncate(f"Denied: {name}({args_str})" if args_str else f"Denied: {name}", 120)
            try:
                await stream.append(markdown_text="", chunks=[{
                    "type": "task_update",
                    "id": f"approval:{decision.requirement_id}",
                    "title": title,
                    "status": "complete",
                }])
            except Exception as exc:
                log_error(f"[HITL] decision_update append failed: run_id={run_id} slack_error={slack_error_code(exc)!r} | {exc}")

    async def _stream_continuation(
        ctx: SubmitContext,
        stream: Any,
        requirements: List[Any],
    ) -> StreamState:
        # Streams the continuation, returns final state with paused_event if re-paused
        state = StreamState(entity_name=entity_name, entity_type=entity_type)
        try:
            # requirements= forwards user's confirmations into the resumed run
            response_stream: Any = entity.acontinue_run(  # type: ignore[union-attr, call-arg, call-overload]
                run_id=ctx.run_id,
                requirements=requirements,
                session_id=ctx.session_id,
                stream=True,
                stream_events=True,
            )
        except Exception as exc:
            log_error(f"[HITL] acontinue_run (stream) failed for run={ctx.run_id}: {exc}")
            return state

        try:
            async for chunk in response_stream:
                state.collect_media(chunk)
                ev = getattr(chunk, "event", None)
                if ev and await process_event(ev, chunk, state, stream):
                    break
                if state.has_content():
                    content = state.flush()
                    if content and state.stream_chars_sent + len(content) <= _STREAM_CHAR_LIMIT:
                        await stream.append(markdown_text=content)
                        state.stream_chars_sent += len(content)
        except Exception as exc:
            log_error(f"[HITL] continuation append failed: run_id={ctx.run_id} slack_error={slack_error_code(exc)!r} | {exc}")
        return state

    async def _finalize_or_repause(
        ctx: SubmitContext,
        stream: Any,
        state: StreamState,
    ) -> None:
        # Handles stream finalization — posts new pause card if re-paused, otherwise stops stream
        from slack_sdk.web.async_client import AsyncWebClient

        client = AsyncWebClient(token=slack_tools.token, ssl=ssl)

        if state.paused_event is not None:
            requirements = list(getattr(state.paused_event, "active_requirements", None) or [])
            if requirements:
                from agno.os.interfaces.slack.pause import finalize_pause, post_pause_card

                new_awaiting_ts = await finalize_pause(
                    client=client,
                    stream=stream,
                    state=state,
                    run_id=ctx.run_id,
                    channel=ctx.channel,
                    thread_ts=ctx.thread_ts,
                    requirements=requirements,
                    log_prefix="re-",
                )
                try:
                    await post_pause_card(client, state.paused_event, ctx.channel, ctx.thread_ts, new_awaiting_ts)
                except Exception as exc:
                    log_error(f"[HITL] Failed to post Card block (re-pause): {exc}")
                return

        # Normal completion — stop the stream
        stop_kwargs: Dict[str, Any] = {}
        if state.has_content():
            stop_kwargs["markdown_text"] = state.flush()
        if state.task_cards:
            final_status: TaskStatus = state.terminal_status or "complete"
            completion_chunks = state.resolve_all_pending(final_status)
            if completion_chunks:
                stop_kwargs["chunks"] = completion_chunks
        try:
            await stream.stop(**stop_kwargs)
        except Exception as exc:
            log_error(f"[HITL] stream.stop after resume failed: run_id={ctx.run_id} slack_error={slack_error_code(exc)!r} | {exc}")

    async def _handle_row_approve(payload: Dict[str, Any]) -> None:
        # Toggle row to approved, auto-submit if all confirmation rows decided
        ctx = extract_row_action_context(payload)
        if ctx is None:
            return

        result = select_confirmation_row(ctx, selected="approve", include_reason_input=False)
        await _safe_chat_update(ctx.channel, ctx.card_ts, "Approval pending", result.blocks, "row approve")

        if result.should_auto_submit:
            await _handle_submit(synthetic_submit_payload(payload, ctx.run_id, ctx.awaiting_ts, result.blocks))

    async def _handle_row_reject_start(payload: Dict[str, Any]) -> None:
        # Toggle row to denied + add reason input, append Submit if all rows decided
        ctx = extract_row_action_context(payload)
        if ctx is None:
            return

        result = select_confirmation_row(ctx, selected="deny", include_reason_input=True)
        # Deny stays interactive so user can add optional reason before Submit
        blocks = append_submit_if_needed(result.blocks, ctx.run_id, ctx.awaiting_ts)
        await _safe_chat_update(ctx.channel, ctx.card_ts, "Rejection pending", blocks, "row reject")

    async def _handle_submit(payload: Dict[str, Any]) -> None:
        # Opens fresh stream — Slack's 5min timeout makes reusing pre-pause ts unreliable
        from slack_sdk.web.async_client import AsyncWebClient

        ctx = _extract_submit_context(payload)
        if ctx is None:
            return
        log_info(f"[HITL] submit received: run_id={ctx.run_id} channel={ctx.channel}")

        await _delete_awaiting_indicator(ctx.channel, ctx.awaiting_ts)

        requirements = await _load_active_requirements(ctx)
        if not requirements:
            client = AsyncWebClient(token=slack_tools.token, ssl=ssl)
            await _post_ephemeral(client, channel=ctx.channel, user=ctx.user_id, text="This approval is no longer active.")
            return

        decisions = await _validate_and_apply_decisions(ctx, payload, requirements)
        if decisions is None:
            return

        original_blocks = list((payload.get("message") or {}).get("blocks") or [])
        await _lock_submitted_form(ctx, original_blocks, requirements)

        # Open fresh continuation stream
        client = AsyncWebClient(token=slack_tools.token, ssl=ssl)
        stream = await client.chat_stream(
            channel=ctx.channel,
            thread_ts=ctx.thread_ts,
            recipient_team_id=ctx.team_id,
            recipient_user_id=ctx.user_id,
            task_display_mode=task_display_mode,
            buffer_size=buffer_size,
        )

        await _emit_denied_decision_cards(stream, decisions, requirements, ctx.run_id)
        state = await _stream_continuation(ctx, stream, requirements)
        await _finalize_or_repause(ctx, stream, state)

    async def _post_ephemeral(
        client: Any,
        *,
        channel: str,
        user: str,
        text: str,
    ) -> None:
        try:
            await client.chat_postEphemeral(channel=channel, user=user, text=text)
        except Exception as exc:
            log_error(f"[HITL] chat_postEphemeral failed: {exc}")

    async def _process_slack_event(data: dict):
        event = data["event"]
        if not should_respond(event, reply_to_mentions_only):
            return

        from slack_sdk.web.async_client import AsyncWebClient

        ctx = extract_event_context(event)
        async_client = AsyncWebClient(token=slack_tools.token, ssl=ssl)

        # Replace the bot's @mention with its Slack display name so the agent sees
        # "hi Scout" instead of "hi " when the user types "hi @Scout"
        bot_user_id = (data.get("authorizations") or [{}])[0].get("user_id")
        bot_name = await bot_name_resolver.resolve(async_client, bot_user_id) if bot_user_id else None
        ctx["message_text"] = strip_bot_mention(ctx["message_text"], bot_user_id, bot_name)

        # Namespace with entity_id so threads don't collide across mounted interfaces
        session_id = f"{entity_id}:{ctx['thread_id']}"

        try:
            await async_client.assistant_threads_setStatus(
                channel_id=ctx["channel_id"],
                thread_ts=ctx["thread_id"],
                status=loading_text,
            )
        except Exception:
            pass  # Best-effort UX — typing indicator failure doesn't block response

        try:
            # Resolve Slack user ID to email + display name when opted in
            resolved_user_id = ctx["user"]
            display_name = None
            if resolve_user_identity:
                resolved_user_id, display_name = await resolve_slack_user(async_client, ctx["user"])

            channel_name = await resolve_channel_name(async_client, ctx["channel_id"])

            files, images, videos, audio, skipped = await download_event_files_async(
                slack_tools.token, event, slack_tools.max_file_size
            )

            message_text = ctx["message_text"]
            if skipped:
                notice = "[Skipped files: " + ", ".join(skipped) + "]"
                message_text = f"{notice}\n{message_text}"
            run_kwargs: Dict[str, Any] = {
                "user_id": resolved_user_id,
                "session_id": session_id,
                "metadata": build_run_metadata(display_name, resolved_user_id, ctx),
                "dependencies": {
                    "Slack channel": f"#{channel_name}" if channel_name else ctx["channel_id"],
                    "Slack channel_id": ctx["channel_id"],
                    "Slack thread_ts": ctx["thread_id"],
                },
                "add_dependencies_to_context": True,
                "files": files or None,
                "images": images or None,
                "videos": videos or None,
                "audio": audio or None,
            }

            response = await entity.arun(message_text, **run_kwargs)  # type: ignore[union-attr]

            if response:
                if response.status == "ERROR":
                    log_error(f"Error processing message: {response.content}")
                    await send_slack_message_async(
                        async_client,
                        channel=ctx["channel_id"],
                        message=f"{_ERROR_MESSAGE} Please try again later.",
                        thread_ts=ctx["thread_id"],
                    )
                    return

                # HITL: Handle paused status in non-streaming mode
                if response.status == "PAUSED":
                    requirements = list(getattr(response, "active_requirements", None) or [])
                    run_id = getattr(response, "run_id", None)
                    if run_id and requirements:
                        from agno.os.interfaces.slack.pause import _PAUSE_LABELS, post_pause_card
                        from agno.os.interfaces.slack.types import tool_name

                        # Send the paused content message
                        content = str(response.content) if response.content else ""
                        if content:
                            await send_slack_message_async(
                                async_client,
                                channel=ctx["channel_id"],
                                message=content,
                                thread_ts=ctx["thread_id"],
                            )

                        # Post awaiting indicator
                        pause_labels = [_PAUSE_LABELS[r.pause_type].format(tool=tool_name(r)) for r in requirements]
                        awaiting_ts = None
                        if pause_labels:
                            try:
                                awaiting_resp = await async_client.chat_postMessage(
                                    channel=ctx["channel_id"],
                                    thread_ts=ctx["thread_id"],
                                    text="\n".join(pause_labels),
                                    mrkdwn=True,
                                )
                                awaiting_ts = awaiting_resp.get("ts")
                            except Exception as exc:
                                log_error(f"[HITL] Non-streaming awaiting indicator failed: {exc}")

                        # Post the HITL card with approve/reject buttons
                        try:
                            await post_pause_card(
                                async_client, response, ctx["channel_id"], ctx["thread_id"], awaiting_ts
                            )
                        except Exception as exc:
                            log_error(f"[HITL] Non-streaming pause card failed: {exc}")
                        return

                if hasattr(response, "reasoning_content") and response.reasoning_content:
                    rc = str(response.reasoning_content)
                    formatted = "*Reasoning:*\n> " + rc.replace("\n", "\n> ")
                    await send_slack_message_async(
                        async_client,
                        channel=ctx["channel_id"],
                        message=formatted,
                        thread_ts=ctx["thread_id"],
                    )

                content = str(response.content) if response.content else ""
                await send_slack_message_async(
                    async_client,
                    channel=ctx["channel_id"],
                    message=content,
                    thread_ts=ctx["thread_id"],
                )
                await upload_response_media_async(async_client, response, ctx["channel_id"], ctx["thread_id"])
        except Exception as e:
            log_error(f"Error processing slack event: {str(e)}")
            await send_slack_message_async(
                async_client,
                channel=ctx["channel_id"],
                message=_ERROR_MESSAGE,
                thread_ts=ctx["thread_id"],
            )
        finally:
            # Clear "Thinking..." status. In streaming mode stream.stop() handles
            # this automatically, but the non-streaming path must clear explicitly.
            try:
                await async_client.assistant_threads_setStatus(
                    channel_id=ctx["channel_id"], thread_ts=ctx["thread_id"], status=""
                )
            except Exception:
                pass  # Best-effort UX — clearing status indicator is cosmetic

    async def _stream_slack_response(data: dict):
        from slack_sdk.web.async_client import AsyncWebClient

        event = data["event"]
        if not should_respond(event, reply_to_mentions_only):
            return

        ctx = extract_event_context(event)

        async_client = AsyncWebClient(token=slack_tools.token, ssl=ssl)

        # Replace the bot's @mention with its Slack display name so the agent sees
        # "hi Scout" instead of "hi " when the user types "hi @Scout"
        bot_user_id = (data.get("authorizations") or [{}])[0].get("user_id")
        bot_name = await bot_name_resolver.resolve(async_client, bot_user_id) if bot_user_id else None
        ctx["message_text"] = strip_bot_mention(ctx["message_text"], bot_user_id, bot_name)

        session_id = f"{entity_id}:{ctx['thread_id']}"

        # Not consistently placed across Slack event envelope shapes
        team_id = data.get("team_id") or event.get("team")
        # CRITICAL: recipient_user_id must be the HUMAN user, not the bot.
        # event["user"] = human who sent the message. data["authorizations"][0]["user_id"]
        # = the bot's own user ID. Using the bot ID causes Slack to stream content
        # to an invisible recipient, resulting in a blank bubble until stopStream.
        user_id = ctx["user"]
        state = StreamState(entity_type=entity_type, entity_name=entity_name)
        stream = None

        try:
            try:
                status_kwargs: Dict[str, Any] = {
                    "channel_id": ctx["channel_id"],
                    "thread_ts": ctx["thread_id"],
                    "status": loading_text,
                }
                if loading_messages:
                    status_kwargs["loading_messages"] = loading_messages
                await async_client.assistant_threads_setStatus(**status_kwargs)
            except Exception:
                pass  # Best-effort UX — typing indicator failure doesn't block response

            # Resolve Slack user ID to email + display name when opted in
            resolved_user_id = ctx["user"]
            display_name = None
            if resolve_user_identity:
                resolved_user_id, display_name = await resolve_slack_user(async_client, ctx["user"])

            channel_name = await resolve_channel_name(async_client, ctx["channel_id"])

            files, images, videos, audio, skipped = await download_event_files_async(
                slack_tools.token, event, slack_tools.max_file_size
            )

            message_text = ctx["message_text"]
            if skipped:
                notice = "[Skipped files: " + ", ".join(skipped) + "]"
                message_text = f"{notice}\n{message_text}"
            run_kwargs: Dict[str, Any] = {
                "stream": True,
                # Enables event-level chunks for task card and tool lifecycle rendering
                "stream_events": True,
                "user_id": resolved_user_id,
                "session_id": session_id,
                "metadata": build_run_metadata(display_name, resolved_user_id, ctx),
                "dependencies": {
                    "Slack channel": f"#{channel_name}" if channel_name else ctx["channel_id"],
                    "Slack channel_id": ctx["channel_id"],
                    "Slack thread_ts": ctx["thread_id"],
                },
                "add_dependencies_to_context": True,
                "files": files or None,
                "images": images or None,
                "videos": videos or None,
                "audio": audio or None,
            }

            response_stream = entity.arun(message_text, **run_kwargs)  # type: ignore[union-attr]

            if response_stream is None:
                try:
                    await async_client.assistant_threads_setStatus(
                        channel_id=ctx["channel_id"], thread_ts=ctx["thread_id"], status=""
                    )
                except Exception:
                    pass  # Best-effort UX — clearing status indicator is cosmetic
                return

            # Deferred so "Thinking..." indicator stays visible during file
            # download and agent startup (opening earlier shows a blank bubble)
            stream = await async_client.chat_stream(
                channel=ctx["channel_id"],
                thread_ts=ctx["thread_id"],
                recipient_team_id=team_id,
                recipient_user_id=user_id,
                task_display_mode=task_display_mode,
                buffer_size=buffer_size,
            )

            async def _rotate_stream(pending_text: str = ""):
                # Close current bubble at Slack's payload limit, open a fresh one.
                # Preserves in_progress task cards across the rotation so the user
                # sees continuity rather than orphaned open cards in the old bubble.
                nonlocal stream
                assert stream is not None  # Caller only invokes after stream is opened
                in_progress = [(k, v.title) for k, v in state.task_cards.items() if v.status == "in_progress"]
                rotate_stop: Dict[str, Any] = {}
                if state.task_cards:
                    rotate_stop["chunks"] = state.resolve_all_pending("complete")
                await stream.stop(**rotate_stop)
                new_stream = await async_client.chat_stream(
                    channel=ctx["channel_id"],
                    thread_ts=ctx["thread_id"],
                    recipient_team_id=team_id,
                    recipient_user_id=user_id,
                    task_display_mode=task_display_mode,
                    buffer_size=buffer_size,
                )
                # Only mutate state after both async ops succeed
                state.task_cards.clear()
                state.stream_chars_sent = 0
                stream = new_stream
                # Re-open in-progress cards so the user sees continuity
                for key, card_title in in_progress:
                    state.track_task(key, card_title)
                    await stream.append(
                        markdown_text="",
                        chunks=[{"type": "task_update", "id": key, "title": card_title, "status": "in_progress"}],
                    )
                if pending_text:
                    continued = "_(continued)_\n" + pending_text
                    await stream.append(markdown_text=continued)
                    state.stream_chars_sent = len(continued)

            async for chunk in response_stream:
                state.collect_media(chunk)

                ev = getattr(chunk, "event", None)
                if ev:
                    if await process_event(ev, chunk, state, stream):
                        break

                # Card overflow: rotate before Slack rejects the payload
                if len(state.task_cards) >= _STREAM_CARD_LIMIT:
                    await _rotate_stream(state.flush() if state.has_content() else "")

                if state.has_content():
                    if not state.title_set:
                        state.title_set = True
                        title = ctx["message_text"][:50].strip() or "New conversation"
                        try:
                            await async_client.assistant_threads_setTitle(
                                channel_id=ctx["channel_id"], thread_ts=ctx["thread_id"], title=title
                            )
                        except Exception:
                            pass  # Best-effort UX — title update failure doesn't block response

                    content = state.flush()
                    content_len = len(content)
                    if state.stream_chars_sent + content_len <= _STREAM_CHAR_LIMIT:
                        await stream.append(markdown_text=content)
                        state.stream_chars_sent += content_len
                    else:
                        await _rotate_stream(content)
            # Default to complete when no terminal error/cancel event arrived
            final_status: TaskStatus = state.terminal_status or "complete"
            completion_chunks = state.resolve_all_pending(final_status) if state.task_cards else []
            stop_kwargs: Dict[str, Any] = {}
            if state.has_content():
                stop_kwargs["markdown_text"] = state.flush()
            if completion_chunks:
                stop_kwargs["chunks"] = completion_chunks

            # HITL pause: stop the pre-pause stream cleanly and post the
            # approval Card as a separate message. In-progress task cards
            # flip to "pending" — Slack renders these as neutral waiting
            # indicators (not the red errored state it would otherwise
            # assign to in_progress cards whose stream has ended). The
            # resume handler opens a fresh chat_stream for the continuation
            # in a new bubble when the human clicks.
            if state.paused_event is not None:
                pause_run_id = getattr(state.paused_event, "run_id", None)
                requirements = list(getattr(state.paused_event, "active_requirements", None) or [])
                if pause_run_id and requirements:
                    from agno.os.interfaces.slack.pause import finalize_pause, post_pause_card

                    awaiting_ts = await finalize_pause(
                        client=async_client,
                        stream=stream,
                        state=state,
                        run_id=pause_run_id,
                        channel=ctx["channel_id"],
                        thread_ts=ctx["thread_id"],
                        requirements=requirements,
                    )
                    try:
                        await post_pause_card(
                            async_client, state.paused_event, ctx["channel_id"], ctx["thread_id"], awaiting_ts
                        )
                    except Exception as exc:
                        log_error(f"[HITL] Failed to post Card block (pause): {exc}")
                    return

            await stream.stop(**stop_kwargs)

            await upload_response_media_async(async_client, state, ctx["channel_id"], ctx["thread_id"])

        except Exception as e:
            # Check structured response first (cheap); fall back to str(e) only if needed
            slack_resp = getattr(e, "response", None)
            slack_body = slack_resp.data if slack_resp else None
            slack_error = slack_body.get("error", "") if isinstance(slack_body, dict) else ""
            is_msg_too_long = "msg_too_long" in slack_error or "msg_blocks_too_long" in slack_error
            if not is_msg_too_long:
                is_msg_too_long = "msg_too_long" in str(e)
            if not is_msg_too_long:
                log_error(
                    f"Error streaming slack response [channel={ctx['channel_id']}, thread={ctx['thread_id']}, user={user_id}]: {e}"
                )
            try:
                await async_client.assistant_threads_setStatus(
                    channel_id=ctx["channel_id"], thread_ts=ctx["thread_id"], status=""
                )
            except Exception:
                pass  # Best-effort UX — clearing status indicator is cosmetic
            # Clean up open stream so Slack doesn't show stuck progress indicators
            if stream is not None:
                try:
                    stop_kwargs_err: Dict[str, Any] = {}
                    if state.task_cards:
                        stop_kwargs_err["chunks"] = state.resolve_all_pending(
                            "complete" if is_msg_too_long else "error"
                        )
                    await stream.stop(**stop_kwargs_err)
                except Exception:
                    pass  # Best-effort cleanup — stream may already be closed
            if not is_msg_too_long:
                await send_slack_message_async(
                    async_client,
                    channel=ctx["channel_id"],
                    message=_ERROR_MESSAGE,
                    thread_ts=ctx["thread_id"],
                )

    async def _handle_thread_started(event: dict):
        from slack_sdk.web.async_client import AsyncWebClient

        async_client = AsyncWebClient(token=slack_tools.token, ssl=ssl)
        thread_info = event.get("assistant_thread", {})
        channel_id = thread_info.get("channel_id", "")
        thread_ts = thread_info.get("thread_ts", "")
        if not channel_id or not thread_ts:
            return

        prompts = suggested_prompts or [
            {"title": "Help", "message": "What can you help me with?"},
            {"title": "Search", "message": "Search the web for..."},
        ]
        try:
            await async_client.assistant_threads_setSuggestedPrompts(
                channel_id=channel_id, thread_ts=thread_ts, prompts=prompts
            )
        except Exception as e:
            log_error(f"Failed to set suggested prompts: {str(e)}")

    return router
