"""Tests for streaming API interaction.

Verifies T2.2: Stream response parsing with mock events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from longline.api.claude import stream_response
from longline.core.events import ErrorEvent, TextDelta, ToolUseStart, TurnComplete


@dataclass
class MockContentBlock:
    type: str
    id: str = ""
    name: str = ""
    text: str = ""
    data: str = ""


@dataclass
class MockDelta:
    type: str
    text: str = ""
    partial_json: str = ""
    thinking: str = ""
    stop_reason: str | None = None


@dataclass
class MockUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class MockMessage:
    usage: MockUsage
    model: str = ""


@dataclass
class MockEvent:
    type: str
    index: int = 0
    content_block: MockContentBlock | None = None
    delta: MockDelta | None = None
    message: MockMessage | None = None
    usage: MockUsage | None = None


def make_text_stream_events() -> list[MockEvent]:
    """Create mock events for a simple text response."""
    return [
        MockEvent(type="message_start", message=MockMessage(usage=MockUsage())),
        MockEvent(type="content_block_start", index=0, content_block=MockContentBlock(type="text")),
        MockEvent(type="content_block_delta", index=0, delta=MockDelta(type="text_delta", text="Hello ")),
        MockEvent(type="content_block_delta", index=0, delta=MockDelta(type="text_delta", text="world!")),
        MockEvent(type="content_block_stop", index=0),
        MockEvent(
            type="message_delta",
            delta=MockDelta(type="message_delta", stop_reason="end_turn"),
            usage=MockUsage(output_tokens=10),
        ),
    ]


def make_tool_use_stream_events() -> list[MockEvent]:
    """Create mock events for a text + tool_use response."""
    return [
        MockEvent(type="message_start", message=MockMessage(usage=MockUsage())),
        # Text block
        MockEvent(type="content_block_start", index=0, content_block=MockContentBlock(type="text")),
        MockEvent(type="content_block_delta", index=0, delta=MockDelta(type="text_delta", text="Let me check.")),
        MockEvent(type="content_block_stop", index=0),
        # Tool use block
        MockEvent(
            type="content_block_start",
            index=1,
            content_block=MockContentBlock(type="tool_use", id="tu_1", name="bash"),
        ),
        MockEvent(
            type="content_block_delta",
            index=1,
            delta=MockDelta(type="input_json_delta", partial_json='{"comma'),
        ),
        MockEvent(
            type="content_block_delta",
            index=1,
            delta=MockDelta(type="input_json_delta", partial_json='nd": "ls"}'),
        ),
        MockEvent(type="content_block_stop", index=1),
        MockEvent(
            type="message_delta",
            delta=MockDelta(type="message_delta", stop_reason="tool_use"),
            usage=MockUsage(output_tokens=20),
        ),
    ]


class MockStream:
    """Mock async context manager + async iterator for stream events."""

    def __init__(self, events: list[MockEvent]) -> None:
        self.events = events

    async def __aenter__(self) -> MockStream:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    def __aiter__(self) -> AsyncIterator[MockEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[MockEvent]:
        for event in self.events:
            yield event


class TestStreamResponseTextOnly:
    async def test_text_deltas_yielded(self) -> None:
        """Mock: pure text response → TextDelta events + TurnComplete."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=MockStream(make_text_stream_events()))

        events = [e async for e in stream_response(
            mock_client,
            messages=[{"role": "user", "content": "hello"}],
            system="You are helpful.",
            model="test-model",
        )]

        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_deltas) == 2
        assert text_deltas[0].text == "Hello "
        assert text_deltas[1].text == "world!"

        turns = [e for e in events if isinstance(e, TurnComplete)]
        assert len(turns) == 1
        assert turns[0].stop_reason == "end_turn"


class TestStreamResponseToolUse:
    async def test_tool_use_detected(self) -> None:
        """Mock: text + tool_use → TextDelta + ToolUseStart + TurnComplete(tool_use)."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=MockStream(make_tool_use_stream_events()))

        events = [e async for e in stream_response(
            mock_client,
            messages=[{"role": "user", "content": "list files"}],
            system="You are helpful.",
            model="test-model",
        )]

        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_deltas) == 1
        assert text_deltas[0].text == "Let me check."

        tool_starts = [e for e in events if isinstance(e, ToolUseStart)]
        assert len(tool_starts) == 1
        assert tool_starts[0].tool_name == "bash"
        assert tool_starts[0].tool_id == "tu_1"
        assert tool_starts[0].input == {"command": "ls"}

        turns = [e for e in events if isinstance(e, TurnComplete)]
        assert len(turns) == 1
        assert turns[0].stop_reason == "tool_use"

    async def test_partial_json_accumulated(self) -> None:
        """Tool input JSON is correctly accumulated from partial chunks."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=MockStream(make_tool_use_stream_events()))

        events = [e async for e in stream_response(
            mock_client,
            messages=[{"role": "user", "content": "test"}],
            system="test",
        )]

        tool_starts = [e for e in events if isinstance(e, ToolUseStart)]
        assert tool_starts[0].input == {"command": "ls"}


class TestServedModel:
    """`TurnComplete.served_model` is the RESPONSE's model, not the request's.

    The distinction is not pedantic: the Anthropic-compatible gateway this
    project runs against (cc-switch at `http://127.0.0.1:15721`) answers every
    request with `model="deepseek-flash"` regardless of the `model` field it was
    sent. Recording the requested name would have made every eval artifact
    describe a model that never ran.
    """

    async def test_served_model_is_taken_from_the_response(self) -> None:
        """FAILS ON: reading `params["model"]`, which the gateway ignores."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=MockStream([
            MockEvent(type="message_start", message=MockMessage(usage=MockUsage(), model="deepseek-flash")),
            MockEvent(type="content_block_start", index=0, content_block=MockContentBlock(type="text")),
            MockEvent(type="content_block_delta", index=0, delta=MockDelta(type="text_delta", text="hi")),
            MockEvent(type="content_block_stop", index=0),
            MockEvent(
                type="message_delta",
                delta=MockDelta(type="message_delta", stop_reason="end_turn"),
                usage=MockUsage(output_tokens=10),
            ),
        ]))

        events = [e async for e in stream_response(
            mock_client,
            messages=[{"role": "user", "content": "hello"}],
            system="test",
            model="claude-sonnet-4-20250514",  # deliberately NOT what comes back
        )]

        turns = [e for e in events if isinstance(e, TurnComplete)]
        assert len(turns) == 1
        assert turns[0].served_model == "deepseek-flash"

    async def test_a_silent_transport_reports_none_not_the_request(self) -> None:
        """FAILS ON: defaulting `served_model` to the requested model.

        A transport that says nothing has told us nothing. Substituting the
        request value would turn "unknown" into a confident claim, and the
        report would publish a model name with no evidence behind it.
        """
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=MockStream([
            MockEvent(type="message_start", message=MockMessage(usage=MockUsage())),  # model=""
            MockEvent(type="content_block_start", index=0, content_block=MockContentBlock(type="text")),
            MockEvent(type="content_block_delta", index=0, delta=MockDelta(type="text_delta", text="hi")),
            MockEvent(type="content_block_stop", index=0),
        ]))

        events = [e async for e in stream_response(
            mock_client,
            messages=[{"role": "user", "content": "hello"}],
            system="test",
            model="claude-sonnet-4-20250514",
        )]

        turns = [e for e in events if isinstance(e, TurnComplete)]
        assert turns[0].served_model is None


class TestTransientGatewayErrorsAreRetryable:
    """A 5xx from the gateway is an outage, not a verdict about the case.

    `is_recoverable=True` is what makes `query_loop` back off and retry
    (2s..10s, capped by `max_retry`) instead of ending the case. Without it, a
    gateway blip writes a row that measured nothing -- and a long paid run can
    collect hundreds of them. This was not hypothetical: a live suite died on
    503 "Upstream request failed: Endpoint is unavailable" on every case.
    """

    @staticmethod
    def _status_error(status: int, body: dict) -> Any:
        import httpx

        import anthropic

        return anthropic.APIStatusError(
            f"Error code: {status} - {body}",
            response=httpx.Response(status, request=httpx.Request("POST", "http://x")),
            body=body,
        )

    @pytest.mark.parametrize("status", [429, 529, 502, 503, 504])
    async def test_transient_statuses_are_recoverable(self, status: int) -> None:
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=self._status_error(status, {"error": {"type": "server_error"}})
        )

        events = [e async for e in stream_response(
            mock_client, messages=[{"role": "user", "content": "hi"}], system="t",
        )]

        errors = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(errors) == 1
        assert errors[0].is_recoverable is True

    async def test_a_client_error_stays_fatal(self) -> None:
        """FAILS ON: treating every 4xx/5xx as retryable.

        A 400/404/413 is a fact about the REQUEST; retrying it five times just
        spends 30 seconds per case re-sending something the server has already
        rejected on its merits.
        """
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=self._status_error(400, {"error": {"type": "invalid_request_error"}})
        )

        events = [e async for e in stream_response(
            mock_client, messages=[{"role": "user", "content": "hi"}], system="t",
        )]

        errors = [e for e in events if isinstance(e, ErrorEvent)]
        assert errors[0].is_recoverable is False


class TestInputTokensFromEitherEvent:
    """A gateway may report input_tokens in `message_delta`, not `message_start`.

    Anthropic's convention puts `input_tokens` in `message_start.message.usage`
    and only `output_tokens` in `message_delta`. The gateway these eval runs go
    through does the opposite: it sends `message_start.usage.input_tokens = 0`
    and the real count in `message_delta`. Reading only the documented field
    recorded `input_tokens = 0` on every row -- a token bill that is missing its
    entire input half, while the output half looks perfectly healthy.
    """

    @staticmethod
    def _stream(start_in: int, delta_in: int | None) -> MockStream:
        delta_usage = MockUsage(input_tokens=delta_in if delta_in is not None else 0)
        events = [
            MockEvent(type="message_start", message=MockMessage(
                usage=MockUsage(input_tokens=start_in), model="omen-alpha")),
            MockEvent(type="content_block_start", index=0, content_block=MockContentBlock(type="text")),
            MockEvent(type="content_block_delta", index=0, delta=MockDelta(type="text_delta", text="hi")),
            MockEvent(type="content_block_stop", index=0),
            MockEvent(type="message_delta", delta=MockDelta(
                type="message_delta", stop_reason="end_turn"), usage=delta_usage),
        ]
        return MockStream(events)

    async def _usage(self, stream: MockStream) -> Any:
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=stream)
        events = [e async for e in stream_response(
            mock_client, messages=[{"role": "user", "content": "hi"}], system="t",
        )]
        return [e for e in events if isinstance(e, TurnComplete)][0].usage

    async def test_a_gateway_that_reports_input_in_the_delta_is_believed(self) -> None:
        """FAILS ON: trusting `message_start` alone, which reads 0 here."""
        usage = await self._usage(self._stream(start_in=0, delta_in=38))
        assert usage.input_tokens == 38

    async def test_the_documented_field_still_wins_when_the_delta_is_silent(self) -> None:
        """Anthropic sends no `input_tokens` in `message_delta`; nothing changes."""
        usage = await self._usage(self._stream(start_in=100, delta_in=None))
        assert usage.input_tokens == 100

    async def test_a_zero_in_the_delta_never_clobbers_a_real_count(self) -> None:
        """FAILS ON: unconditional assignment from whichever event arrives last.

        Some gateways send `input_tokens=0` in the delta as a placeholder. Taking
        the last value rather than the last NON-ZERO value would turn a correct
        `message_start` reading into a zero.
        """
        usage = await self._usage(self._stream(start_in=100, delta_in=0))
        assert usage.input_tokens == 100


class TestGatewayWrappedUpstreamFailures:
    """The gateway can wrap an upstream outage in HTTP 400.

    A live baseline re-run died because the gateway served
    `400 - {'error': {'type': 'server_error', 'message': 'Upstream request
    failed: Model is unavailable.'}}`. A 400 is normally a verdict about the
    REQUEST and fatal by design -- but this body is the gateway reporting ITS
    upstream, not judging our request. Retrying it is recovery, and treating it
    as fatal wrote 54 rows that measured nothing (again zero tokens).
    """

    @staticmethod
    def _wrapped(status: int, error_type: str, message: str) -> Any:
        import httpx

        import anthropic

        body = {"error": {"type": error_type, "message": message}}
        return anthropic.APIStatusError(
            f"Error code: {status} - {body}",
            response=httpx.Response(status, request=httpx.Request("POST", "http://x")),
            body=body,
        )

    async def test_a_400_from_the_gateway_reporting_its_upstream_is_retryable(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=self._wrapped(
            400, "server_error", "Upstream request failed: Model is unavailable.",
        ))

        events = [e async for e in stream_response(
            mock_client, messages=[{"role": "user", "content": "hi"}], system="t",
        )]

        errors = [e for e in events if isinstance(e, ErrorEvent)]
        assert errors[0].is_recoverable is True

    async def test_a_real_400_is_still_fatal(self) -> None:
        """A client-shaped 400 (invalid request) must not be retried."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=self._wrapped(
            400, "invalid_request_error", "max_tokens: field required",
        ))

        events = [e async for e in stream_response(
            mock_client, messages=[{"role": "user", "content": "hi"}], system="t",
        )]

        errors = [e for e in events if isinstance(e, ErrorEvent)]
        assert errors[0].is_recoverable is False
